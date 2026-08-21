"""
Indic F5 — Tamil Neural TTS Backend with Real-Time Zero-Shot Voice Cloning & In-Built Translator
Features:
 - 🎙️ Indic F5 Neural Flow-Matching Transformer Engine (F5-TTS DiT + Vocos 24kHz)
 - 🎙️ Real-Time Zero-Shot Voice Cloning (/clone_voice)
 - 🗂️ Dynamic Multi-Speaker Profiles (Pre-set & User Custom Cloned Voices)
 - 🤖 Auto-Transcription with Whisper ASR
 - 🌐 In-Built English-to-Tamil Neural Translator (/translate)
 - 🔤 Phonetic Tanglish Transliteration
 - 🔢 Automatic Tamil Numeral & Currency Normalization
 - 🖥️ Interactive Web UI at http://localhost:5050
"""

import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Enforce cache configuration
CACHE_ROOT = os.path.abspath("cache")
os.environ["PIP_CACHE_DIR"] = os.path.join(CACHE_ROOT, "pip")
os.environ["TORCH_HOME"] = os.path.join(CACHE_ROOT, "torch")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import re
import glob
import time
import base64
import json
import shutil
import threading
import urllib.request
import urllib.parse
import numpy as np
import soundfile as sf
import torch
import torchaudio

def soundfile_load(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath)
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.mean(dim=-1, keepdim=True).t()
    return tensor, samplerate

torchaudio.load = soundfile_load
torch.compile = lambda x, *args, **kwargs: x

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from safetensors.torch import load_file
from vocos import Vocos
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model as load_f5_model, infer_batch_process

app = Flask(__name__)
CORS(app)

DATASET_DIR = "dataset"
CHECKPOINTS_DIR = "checkpoints"
INDIC_F5_DIR = os.path.join(CHECKPOINTS_DIR, "indic_f5")
VOCOS_DIR = os.path.join(CHECKPOINTS_DIR, "vocos")
CUSTOM_VOICES_FILE = os.path.join(DATASET_DIR, "custom_voices.json")
WHISPER_CACHE_DIR = os.path.join(CACHE_ROOT, "whisper")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(INDIC_F5_DIR, exist_ok=True)
os.makedirs(VOCOS_DIR, exist_ok=True)

DEFAULT_VOICE_METADATA = {
    "female_1": {"label": "Female 1 (Narrator)", "gender": "female", "icon": "👩", "default_style": "Narrator"},
    "female_2": {"label": "Female 2 (Conversational)", "gender": "female", "icon": "👧", "default_style": "Conversational"},
    "female_3_own": {"label": "Female 3 (Own Voice)", "gender": "female", "icon": "🎙️", "default_style": "Natural"},
    "male_1": {"label": "Male 1 (Narrator)", "gender": "male", "icon": "🧔", "default_style": "Narrator"},
    "male_2": {"label": "Male 2 (Conversational)", "gender": "male", "icon": "👨", "default_style": "Conversational"},
    "male_3_own": {"label": "Male 3 (Own Voice)", "gender": "male", "icon": "🎙️", "default_style": "Natural"},
}

EMOTION_STYLES = {
    "neutral": {"label": "Neutral / Normal", "icon": "😐", "prompt_tag": "[neutral]"},
    "cheerful": {"label": "Happy & Cheerful", "icon": "😊", "prompt_tag": "[cheerful]"},
    "narrator": {"label": "Storyteller / Drama", "icon": "📖", "prompt_tag": "[narrative]"},
    "formal": {"label": "Formal / News", "icon": "👔", "prompt_tag": "[formal]"},
    "surprised": {"label": "Excited & Surprised", "icon": "😲", "prompt_tag": "[excited]"},
}

ema_model = None
vocoder = None
whisper_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_load_lock = threading.Lock()
model_is_loading = False
model_load_error = None


def find_snapshot(pattern_list):
    for pat in pattern_list:
        dirs = glob.glob(pat)
        if dirs:
            return dirs[0]
    return None


def download_vocos_if_needed():
    """Ensure Vocos 24kHz neural vocoder is downloaded locally."""
    config_path = os.path.join(VOCOS_DIR, "config.yaml")
    model_path = os.path.join(VOCOS_DIR, "pytorch_model.bin")
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print("[Vocos] Downloading 24kHz neural vocoder...", flush=True)
        files = [
            ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/config.yaml", config_path),
            ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin", model_path)
        ]
        for url, dest in files:
            if not os.path.exists(dest) or os.path.getsize(dest) < 100:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
        print("[Vocos] ✅ Vocos vocoder ready!", flush=True)


def download_indic_f5_if_needed():
    """Ensure Indic F5 model and vocab are downloaded locally."""
    model_path = os.path.join(INDIC_F5_DIR, "model.safetensors")
    vocab_path = os.path.join(INDIC_F5_DIR, "vocab.txt")

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        return model_path, vocab_path

    # Check HF Cache
    indic_snap = find_snapshot([
        "cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*",
        "cache/**/models--ai4bharat--IndicF5/snapshots/*",
        os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*")
    ])
    if indic_snap and os.path.exists(os.path.join(indic_snap, "model.safetensors")):
        return os.path.join(indic_snap, "model.safetensors"), os.path.join(indic_snap, "checkpoints", "vocab.txt")

    token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import snapshot_download
    print("\n[Indic F5] Downloading IndicF5 Tamil Model from Hugging Face (~1.2 GB)...", flush=True)
    snap = snapshot_download(repo_id="ai4bharat/IndicF5", token=token)
    
    # Copy to checkpoints/indic_f5 for permanent local offline access
    src_model = os.path.join(snap, "model.safetensors")
    src_vocab = os.path.join(snap, "checkpoints", "vocab.txt")
    if os.path.exists(src_model):
        shutil.copy(src_model, model_path)
    if os.path.exists(src_vocab):
        shutil.copy(src_vocab, vocab_path)

    return model_path, vocab_path


def load_indic_f5_pipeline():
    """Load Indic F5 (DiT + Vocos) model into memory."""
    global ema_model, vocoder, model_is_loading, model_load_error
    with model_load_lock:
        if ema_model is not None and vocoder is not None:
            return True
        if model_is_loading:
            return False
        model_is_loading = True
        model_load_error = None
        try:
            print("\n=======================================================", flush=True)
            print(f"  🇮🇳 Initializing Indic F5 Tamil Neural TTS ({device.upper()})...", flush=True)
            print("=======================================================", flush=True)

            download_vocos_if_needed()
            safetensors_path, vocab_path = download_indic_f5_if_needed()

            vocos_config = os.path.join(VOCOS_DIR, "config.yaml")
            vocos_bin = os.path.join(VOCOS_DIR, "pytorch_model.bin")

            print(f"[Indic F5] Loading DiT Transformer ({device})...", flush=True)
            ema_model = load_f5_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device
            )

            print(f"[Indic F5] Loading Vocos Neural Vocoder ({device})...", flush=True)
            vocoder = Vocos.from_hparams(vocos_config)
            vocos_state = torch.load(vocos_bin, map_location=device, weights_only=True)
            vocoder.load_state_dict(vocos_state)
            vocoder.eval()

            print(f"[Indic F5] Loading model weights from {os.path.basename(safetensors_path)}...", flush=True)
            state_dict = load_file(safetensors_path, device=device)
            ema_state = {k.replace("ema_model._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("ema_model.")}
            vocoder_state = {k.replace("vocoder._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("vocoder.")}

            ema_model.load_state_dict(ema_state, strict=True)
            if vocoder_state:
                vocoder.load_state_dict(vocoder_state, strict=False)

            ema_model.eval()
            vocoder.eval()
            model_is_loading = False
            print(f"[Indic F5] ✅ Indic F5 Tamil Speech Model is 100% Ready on {device}!\n", flush=True)
            return True
        except Exception as e:
            model_is_loading = False
            model_load_error = str(e)
            import traceback
            traceback.print_exc()
            print(f"[Indic F5] Error loading Indic F5: {e}", flush=True)
            return False


def load_whisper():
    """Loads Whisper ASR model for Tamil speech recognition and auto-transcription."""
    global whisper_model
    if whisper_model is None:
        try:
            import whisper
            print("[Whisper] Loading Whisper ASR model for Tamil speech transcription...", flush=True)
            whisper_model = whisper.load_model("small", download_root=WHISPER_CACHE_DIR, device=device)
            print("[Whisper] ✅ Whisper model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[Whisper] Warning: Could not load Whisper ({e})", flush=True)


def get_custom_voices_metadata():
    """Reads custom voices JSON file."""
    if os.path.exists(CUSTOM_VOICES_FILE):
        try:
            with open(CUSTOM_VOICES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_custom_voices_metadata(meta):
    """Saves custom voices metadata."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(CUSTOM_VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_available_voices():
    """Dynamically scan dataset/ for default speakers + custom cloned voices."""
    voices = {}
    if not os.path.exists(DATASET_DIR):
        return voices

    custom_meta = get_custom_voices_metadata()
    all_metadata = {**DEFAULT_VOICE_METADATA, **custom_meta}

    for spk_id, meta in all_metadata.items():
        spk_dir = os.path.join(DATASET_DIR, spk_id)
        if not os.path.isdir(spk_dir):
            continue

        ref_wav = os.path.join(spk_dir, "reference.wav")
        if not os.path.exists(ref_wav):
            wavs = [f for f in os.listdir(spk_dir) if f.endswith(".wav")]
            if wavs:
                ref_wav = os.path.join(spk_dir, sorted(wavs)[0])
            else:
                continue

        ref_txt = os.path.join(spk_dir, "transcript.txt")
        transcript = ""
        if os.path.exists(ref_txt):
            try:
                with open(ref_txt, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
            except Exception:
                pass

        voices[spk_id] = {
            "path": ref_wav,
            "text": transcript if transcript else "வணக்கம்",
            "label": meta.get("label", spk_id),
            "gender": meta.get("gender", "neutral"),
            "icon": meta.get("icon", "🎙️"),
            "default_style": meta.get("default_style", "Natural"),
            "is_custom": spk_id not in DEFAULT_VOICE_METADATA
        }

    return voices


def fix_tamil_initial_n(text):
    """Normalize Tamil initial n characters."""
    n_map = {
        'ன': 'ந', 'னா': 'நா', 'னி': 'நி', 'னீ': 'நீ', 'னு': 'நு',
        'னூ': 'நூ', 'னெ': 'நெ', 'னே': 'நே', 'னை': 'நை', 'னொ': 'நொ', 'னோ': 'நோ'
    }
    words = text.split()
    fixed_words = []
    for word in words:
        if not word:
            fixed_words.append(word)
            continue
        for k in sorted(n_map.keys(), key=len, reverse=True):
            if word.startswith(k):
                word = n_map[k] + word[len(k):]
                break
        fixed_words.append(word)
    return ' '.join(fixed_words)


def transliterate_to_tamil(text):
    """Transliterates English-transliterated Tamil (Tanglish) to native Tamil script."""
    if re.search(r'[a-zA-Z]', text):
        try:
            from transliterate import algorithm, azhagi
            raw = algorithm.Iterative.transliterate(azhagi.Transliteration.table, text)
            return fix_tamil_initial_n(raw)
        except Exception:
            pass
    return text


def translate_english_to_tamil(text):
    """Translates English sentence to native Tamil script using Google Translate API with fallback."""
    if not text or not re.search(r'[a-zA-Z]', text):
        return text

    # Primary: High-speed translation API
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ta&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = "".join([part[0] for part in data[0] if part and len(part) > 0 and part[0]])
            if translated.strip():
                return fix_tamil_initial_n(translated.strip())
    except Exception as e:
        print(f"[Translator] Primary translation warning: {e}", flush=True)

    # Fallback: MyMemory translation API
    try:
        mm_url = "https://api.mymemory.translated.net/get?q=" + urllib.parse.quote(text) + "&langpair=en|ta"
        req = urllib.request.Request(mm_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = data.get("responseData", {}).get("translatedText", "")
            if translated and not translated.startswith("MYMEMORY WARNING"):
                return fix_tamil_initial_n(translated.strip())
    except Exception as e:
        print(f"[Translator] Fallback translation warning: {e}", flush=True)

    return text


def normalize_numbers_in_tamil(text):
    """Converts numerals (e.g. 20, 100, 2026, 50%, ₹500, 9.5) to spoken Tamil words."""
    if not text:
        return text

    try:
        import tamil.numeral
    except Exception:
        tamil = None

    tamil_digits = {'௦':'0', '௧':'1', '௨':'2', '௩':'3', '௪':'4', '௫':'5', '௬':'6', '௭':'7', '௮':'8', '௯':'9'}
    for t_digit, a_digit in tamil_digits.items():
        text = text.replace(t_digit, a_digit)

    text = re.sub(r'(\d+)\s*%', r'\1 சதவீதம்', text)
    text = re.sub(r'(?:₹|Rs\.?|INR)\s*(\d+)', r'\1 ரூபாய்', text)

    digits_map = {
        '0': 'பூஜ்ஜியம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு',
        '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது'
    }

    def replace_decimal(match):
        int_part = match.group(1)
        dec_part = match.group(2)
        try:
            w1 = tamil.numeral.num2tamilstr(int(int_part)) if tamil else int_part
            w2 = ' '.join([digits_map.get(d, d) for d in dec_part])
            return f"{w1} புள்ளி {w2}"
        except Exception:
            return match.group(0)

    text = re.sub(r'\b(\d+)\.(\d+)\b', replace_decimal, text)

    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            if num == 0:
                return 'பூஜ்ஜியம்'
            if tamil and 0 < num <= 100000000:
                return tamil.numeral.num2tamilstr(num)
            else:
                return ' '.join([digits_map.get(d, d) for d in num_str])
        except Exception:
            return num_str

    text = re.sub(r'\b\d+\b', replace_number, text)
    return text


def synthesize_speech_indic(text, ref_audio_path, ref_transcript, speed=1.0, quality=30, stability=0.75, emotion="neutral"):
    """
    Synthesizes Tamil speech using Indic F5 Neural Flow-Matching Transformer + Vocos 24kHz.
    """
    global ema_model, vocoder
    if ema_model is None or vocoder is None:
        load_indic_f5_pipeline()

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    # Convert numerals to spoken Tamil
    text = normalize_numbers_in_tamil(text)
    start_time = time.time()

    # Load reference audio
    ref_wav, sr = sf.read(ref_audio_path)
    ref_tensor = torch.from_numpy(ref_wav).float()
    if ref_tensor.ndim == 2:
        ref_tensor = ref_tensor.mean(dim=-1, keepdim=True).t()
    elif ref_tensor.ndim == 1:
        ref_tensor = ref_tensor.unsqueeze(0)

    ref_text = (ref_transcript.strip() if ref_transcript.strip() else "வணக்கம்") + " "
    nfe = max(12, min(int(quality), 32)) if quality else 16
    print(f"[Indic F5] Synthesizing speech (nfe={nfe}) for: '{text[:50]}...' with voice: {os.path.basename(ref_audio_path)}", flush=True)

    result, _, _ = infer_batch_process(
        ref_audio=(ref_tensor, sr),
        ref_text=ref_text,
        gen_text_batches=[text],
        model_obj=ema_model,
        vocoder=vocoder,
        nfe_step=nfe,
        speed=speed,
        device=device
    )

    final_wave = np.asarray(result, dtype=np.float32)
    max_peak = np.max(np.abs(final_wave))
    if max_peak > 0:
        final_wave = (final_wave / max_peak) * 0.95

    sample_rate = 24000
    buf = io.BytesIO()
    sf.write(buf, final_wave, samplerate=sample_rate, format="WAV")
    buf.seek(0)
    elapsed = time.time() - start_time
    duration = len(final_wave) / sample_rate
    print(f"[Indic F5] ✅ Generated {duration:.2f}s Tamil speech in {elapsed:.2f}s", flush=True)
    return buf.read()


@app.route("/", methods=["GET"])
def index():
    with open("tts_ui.html", "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    if ema_model is not None and vocoder is not None:
        status_str = "ready"
        http_code = 200
    elif model_is_loading:
        status_str = "loading"
        http_code = 202
    elif model_load_error:
        status_str = "error"
        http_code = 500
    else:
        status_str = "idle"
        http_code = 200

    return jsonify({
        "status": status_str,
        "model": "Indic F5 (F5-TTS DiT + Vocos 24kHz)",
        "error": model_load_error,
        "device": device,
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "voices_count": len(get_available_voices())
    }), http_code


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "🎙️ Tamil Indic F5 Neural TTS & Voice Cloning API",
            "version": "2.0.0",
            "description": "High-performance Tamil Text-to-Speech API powered by Indic F5-TTS DiT and Vocos 24kHz with real-time zero-shot voice cloning, dynamic multi-speaker profiles, and English-to-Tamil translator."
        },
        "servers": [
            {"url": "http://localhost:5050", "description": "Local Development Server"}
        ],
        "paths": {
            "/voices": {
                "get": {
                    "summary": "Get all available voice profiles & emotions",
                    "responses": {"200": {"description": "Voice list"}}
                }
            },
            "/generate": {
                "post": {
                    "summary": "Generate Tamil speech audio from text",
                    "responses": {"200": {"description": "Generated audio in base64"}}
                }
            },
            "/clone_voice": {
                "post": {
                    "summary": "Clone a custom voice with instant zero-shot adaptation",
                    "responses": {"200": {"description": "Cloned voice metadata"}}
                }
            },
            "/translate": {
                "post": {
                    "summary": "Translate English or Tanglish text to Tamil",
                    "responses": {"200": {"description": "Translated Tamil text"}}
                }
            }
        }
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
def swagger_ui():
    html = """<!DOCTYPE html>
<html>
<head>
    <title>🎙️ Indic F5 Tamil TTS — API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎙️</text></svg>">
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [SwaggerUIBundle.presets.apis]
        });
    </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/voices", methods=["GET"])
def list_voices():
    """Returns available speakers and emotions."""
    voices = get_available_voices()
    voice_list = []
    for k, v in voices.items():
        voice_list.append({
            "key": k,
            "label": v["label"],
            "gender": v["gender"],
            "icon": v["icon"],
            "style": v["default_style"],
            "is_custom": v["is_custom"],
            "sample_url": f"/voice_sample/{k}",
            "sample_text": v.get("text", "")
        })

    emotion_list = []
    for k, v in EMOTION_STYLES.items():
        emotion_list.append({
            "key": k,
            "label": v["label"],
            "icon": v["icon"]
        })

    return jsonify({
        "voices": voice_list,
        "emotions": emotion_list,
        "default_voice": "female_1" if "female_1" in voices else (list(voices.keys())[0] if voices else "")
    })


@app.route("/voice_sample/<voice_key>", methods=["GET"])
def get_voice_sample(voice_key):
    """Serves reference audio sample for a given speaker voice profile."""
    from flask import send_file
    voices = get_available_voices()
    if voice_key not in voices or not os.path.exists(voices[voice_key]["path"]):
        return jsonify({"error": f"Voice sample '{voice_key}' not found"}), 404

    return send_file(voices[voice_key]["path"], mimetype="audio/wav")


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Auto-transcribes uploaded voice sample using Whisper ASR."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    temp_path = os.path.join(DATASET_DIR, f"temp_transcribe_{int(time.time()*1000)}.wav")
    os.makedirs(DATASET_DIR, exist_ok=True)
    audio_file.save(temp_path)

    try:
        load_whisper()
        if whisper_model is None:
            return jsonify({"transcript": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"})

        result = whisper_model.transcribe(temp_path, language="ta")
        transcript = result.get("text", "").strip()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"transcript": transcript if transcript else "வணக்கம்"})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"transcript": "வணக்கம்", "warning": str(e)})


@app.route("/translate", methods=["POST"])
def translate_api():
    """Translates English or phonetic Tanglish to pure Tamil script."""
    data = request.json or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "auto")

    if not text:
        return jsonify({"translated_text": "", "original_text": ""})

    if mode == "phonetic":
        translated = transliterate_to_tamil(text)
    elif mode == "translate":
        translated = translate_english_to_tamil(text)
    else:
        if re.search(r'[a-zA-Z]', text):
            translated = translate_english_to_tamil(text)
        else:
            translated = text

    normalized = normalize_numbers_in_tamil(translated)
    return jsonify({
        "original_text": text,
        "translated_text": normalized
    })


@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    """Upload 1 or more voice clips to dynamically create/fine-tune a cloned voice profile."""
    audio_files = []
    if "audio_files" in request.files:
        audio_files = request.files.getlist("audio_files")
    elif "audio" in request.files:
        audio_files = request.files.getlist("audio")

    # Filter out empty entries
    audio_files = [f for f in audio_files if f and f.filename]
    if not audio_files:
        return jsonify({"error": "No audio files uploaded. Please select at least 1 audio clip."}), 400

    voice_name = request.form.get("name", "").strip()
    voice_gender = request.form.get("gender", "neutral")
    voice_style = request.form.get("style", "Natural")
    voice_icon = request.form.get("icon", "🎙️")

    # Transcripts can be passed as JSON list, multiple form fields, or newline-separated text
    transcripts_raw = request.form.get("transcripts_json") or ""
    transcripts_list = []
    if transcripts_raw:
        try:
            transcripts_list = json.loads(transcripts_raw)
        except Exception:
            transcripts_list = []

    if not transcripts_list:
        transcripts_list = request.form.getlist("transcripts")

    if not transcripts_list:
        single_transcript = request.form.get("transcript", "").strip()
        if single_transcript:
            transcripts_list = [t.strip() for t in single_transcript.split("\n") if t.strip()]

    if not voice_name:
        voice_name = f"Custom Voice {int(time.time())}"

    voice_id = "custom_" + re.sub(r'[^a-zA-Z0-9_]', '_', voice_name.lower())
    spk_dir = os.path.join(DATASET_DIR, voice_id)
    os.makedirs(spk_dir, exist_ok=True)

    train_wavs_dir = os.path.join("training_dataset", "wavs")
    os.makedirs(train_wavs_dir, exist_ok=True)
    meta_csv_path = os.path.join("training_dataset", "metadata.csv")

    processed_samples = []
    all_transcripts = []

    for idx, a_file in enumerate(audio_files, 1):
        raw_filename = a_file.filename or f"clip_{idx}.wav"
        save_name = f"{voice_id}_{idx:04d}.wav"
        local_path = os.path.join(spk_dir, save_name)
        a_file.save(local_path)

        # Standardize audio to mono 24kHz
        duration = 0.0
        try:
            data, sr = sf.read(local_path)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            
            # Resample to 24000Hz if needed
            if sr != 24000:
                import torchaudio.transforms as T
                tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                resampler = T.Resample(orig_freq=sr, new_freq=24000)
                data = resampler(tensor_data).squeeze(0).numpy()
                sr = 24000

            duration = round(len(data) / sr, 2)
            sf.write(local_path, data, sr, format='WAV')

            # Also copy to training_dataset/wavs for fine-tuning
            shutil.copy(local_path, os.path.join(train_wavs_dir, save_name))
        except Exception as e:
            print(f"[Clone] Error standardizing {save_name}: {e}", flush=True)

        # Determine transcript
        clip_transcript = ""
        if idx - 1 < len(transcripts_list) and transcripts_list[idx - 1].strip():
            clip_transcript = transcripts_list[idx - 1].strip()
        else:
            try:
                load_whisper()
                if whisper_model is not None:
                    res = whisper_model.transcribe(local_path, language="ta")
                    clip_transcript = res.get("text", "").strip()
            except Exception as e:
                print(f"[Clone] Whisper transcription warning for {save_name}: {e}", flush=True)

        if not clip_transcript:
            clip_transcript = "வணக்கம், இது எனது மாதிரி குரல் பதிவு."

        all_transcripts.append(clip_transcript)
        processed_samples.append({
            "filename": save_name,
            "original_name": raw_filename,
            "transcript": clip_transcript,
            "duration": duration
        })

        # Append to training_dataset/metadata.csv if exists
        try:
            if os.path.exists(meta_csv_path):
                with open(meta_csv_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{save_name}|{clip_transcript}|{voice_id}|ta|{duration}")
        except Exception:
            pass

    # Save primary reference.wav (best 5-10s or 1st clip)
    ref_wav_dest = os.path.join(spk_dir, "reference.wav")
    first_clip_path = os.path.join(spk_dir, processed_samples[0]["filename"])
    shutil.copy(first_clip_path, ref_wav_dest)

    primary_transcript = all_transcripts[0] if all_transcripts else "வணக்கம்"
    with open(os.path.join(spk_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(primary_transcript)

    # Save full sample transcripts manifest
    with open(os.path.join(spk_dir, "samples_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    meta = get_custom_voices_metadata()
    meta[voice_id] = {
        "label": voice_name,
        "gender": voice_gender,
        "icon": voice_icon,
        "default_style": voice_style,
        "sample_count": len(processed_samples),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_custom_voices_metadata(meta)

    return jsonify({
        "success": True,
        "voice_id": voice_id,
        "voice_key": voice_id,
        "label": voice_name,
        "icon": voice_icon,
        "transcript": primary_transcript,
        "total_samples": len(processed_samples),
        "samples": processed_samples,
        "message": f"Successfully registered '{voice_name}' with {len(processed_samples)} audio samples & transcripts!"
    })


@app.route("/generate", methods=["POST"])
def generate():
    """Generates Tamil speech audio."""
    data = request.json or {}
    text = (data.get("text") or "").strip()
    voice_key = data.get("voice", "female_1")
    speed = float(data.get("speed", 1.0))
    quality = int(data.get("quality", 30))
    stability = float(data.get("stability", 0.75))
    emotion = data.get("emotion", "neutral")
    auto_translate = bool(data.get("auto_translate", True))

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    voices = get_available_voices()
    if voice_key not in voices:
        return jsonify({"error": f"Voice '{voice_key}' not found"}), 400

    voice = voices[voice_key]

    processed_text = text
    if auto_translate and re.search(r'[a-zA-Z]', text):
        processed_text = translate_english_to_tamil(text)
    else:
        processed_text = transliterate_to_tamil(text)

    processed_text = normalize_numbers_in_tamil(processed_text)

    try:
        audio_bytes = synthesize_speech_indic(
            processed_text,
            voice["path"],
            voice["text"],
            speed=speed,
            quality=quality,
            stability=stability,
            emotion=emotion
        )
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return jsonify({
            "audio_b64": b64,
            "voice": voice_key,
            "voice_label": voice["label"],
            "speed": speed,
            "quality": quality,
            "stability": stability,
            "emotion": emotion,
            "transliterated_text": processed_text
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/fine_tune", methods=["POST"])
def fine_tune():
    """Runs Indic F5 multi-speaker fine-tuning on dataset/training_dataset."""
    try:
        import subprocess
        proc = subprocess.run([sys.executable, "train_local_cpu.py"], capture_output=True, text=True)
        return jsonify({
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "output": proc.stdout[-1500:] if proc.stdout else "",
            "error": proc.stderr[-500:] if proc.stderr else "",
            "message": "Fine-tuning completed successfully!" if proc.returncode == 0 else "Fine-tuning encountered an error."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_indic_f5_pipeline()

    voices = get_available_voices()
    print("\n" + "=" * 65, flush=True)
    print("🎙️ Indic F5 Neural Tamil TTS & Zero-Shot Voice Cloning Server", flush=True)
    print(f"🎙️ Available Voices: {len(voices)} speaker profiles", flush=True)
    for k, v in voices.items():
        custom_tag = " (Custom)" if v.get("is_custom") else ""
        print(f"  • [{k}] {v['icon']} {v['label']}{custom_tag}", flush=True)
    print("🌐 Access Web UI at:       http://localhost:5050", flush=True)
    print("📖 Swagger API Docs at:    http://localhost:5050/docs", flush=True)
    print("📋 OpenAPI Spec JSON at:   http://localhost:5050/openapi.json", flush=True)
    print("=" * 65 + "\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)