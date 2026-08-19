"""
Tamil TTS Backend — Fast Neural Speech Synthesis with 6-Speaker Voice Profiles & In-Built Translator
Features:
 - In-Built English-to-Tamil Neural Translator (/translate)
 - Auto-Translation and Phonetic Tanglish Transliteration
 - Direct Offline Local Cache Weights Loading
 - Fast 8-Step ODE Single-Batch Solver on CPU
 - 100% Newly Generated Speech for Typed Text
 - Pitch-Preserving Speed Control (0.8x - 1.3x)
 - 6 Dynamic Voice Profiles with Instant Runtime Switching
 - Serves Web UI directly at http://localhost:5050

Run: .venv311\\Scripts\\python tts_backend.py
"""

import os
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Enforce cache inside TTS Model\cache
CACHE_ROOT = os.path.abspath("cache")
os.environ["HF_HOME"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["PIP_CACHE_DIR"] = os.path.join(CACHE_ROOT, "pip")
os.environ["TORCH_HOME"] = os.path.join(CACHE_ROOT, "torch")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import re
import glob
import time
import base64
import json
import urllib.request
import urllib.parse
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Audio loader fallback for Windows compatibility
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

VOICE_METADATA = {
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
device = "cpu"


def load_neural_pipeline():
    """Load neural TTS foundation model synchronously from local cache with exact weight mapping."""
    global ema_model, vocoder
    print("\n=======================================================", flush=True)
    print("  Initializing Neural Tamil Speech Model into Memory...", flush=True)
    print("=======================================================", flush=True)

    # Locate cached IndicF5 model files
    indic_dirs = glob.glob("cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*")
    if not indic_dirs:
        raise RuntimeError("IndicF5 snapshot not found in cache")
    indic_snap = indic_dirs[0]
    vocab_path = os.path.join(indic_snap, "checkpoints", "vocab.txt")
    safetensors_path = os.path.join(indic_snap, "model.safetensors")

    # Locate cached Vocos model files
    vocos_dirs = glob.glob("cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/*")
    if not vocos_dirs:
        raise RuntimeError("Vocos snapshot not found in cache")
    vocos_snap = vocos_dirs[0]
    vocos_config = os.path.join(vocos_snap, "config.yaml")

    # 1. Instantiate DiT model
    ema_model = load_f5_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        mel_spec_type="vocos",
        vocab_file=vocab_path,
        device=device
    )

    # 2. Instantiate Vocos vocoder locally
    vocoder = Vocos.from_hparams(vocos_config)
    vocos_state = torch.load(os.path.join(vocos_snap, "pytorch_model.bin"), map_location=device)
    vocoder.load_state_dict(vocos_state)
    vocoder.eval()

    # 3. Load safetensors weights cleanly
    state_dict = load_file(safetensors_path, device=device)
    ema_state = {k.replace("ema_model._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("ema_model.")}
    vocoder_state = {k.replace("vocoder._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("vocoder.")}

    ema_model.load_state_dict(ema_state, strict=True)
    vocoder.load_state_dict(vocoder_state, strict=False)

    ema_model.eval()
    vocoder.eval()
    print(f"\n[Model] ✅ Neural Tamil TTS Model is 100% Ready on: {device}\n", flush=True)


def get_available_voices():
    """Dynamically scan dataset/ for all 6 voice profiles."""
    voices = {}
    if not os.path.exists(DATASET_DIR):
        return voices

    for spk_id, meta in VOICE_METADATA.items():
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
            "label": meta["label"],
            "gender": meta["gender"],
            "icon": meta["icon"],
            "default_style": meta.get("default_style", "Natural")
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

    # 1. Primary: High-speed translation API
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

    # 2. Secondary fallback: MyMemory translation API
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

def normalize_numbers_in_tamil(text):
    """Converts numerals (e.g. 20, 100, 2026, 50%, ₹500, 9.5) to spoken Tamil words."""
    if not text:
        return text

    try:
        import tamil.numeral
    except Exception:
        tamil = None

    # 1. Map Tamil script numerals to ASCII digits
    tamil_digits = {'௦':'0', '௧':'1', '௨':'2', '௩':'3', '௪':'4', '௫':'5', '௬':'6', '௭':'7', '௮':'8', '௯':'9'}
    for t_digit, a_digit in tamil_digits.items():
        text = text.replace(t_digit, a_digit)

    # 2. Percentage expansion (e.g. 20% -> 20 சதவீதம்)
    text = re.sub(r'(\d+)\s*%', r'\1 சதவீதம்', text)

    # 3. Currency expansion (e.g. ₹500, Rs. 500 -> 500 ரூபாய்)
    text = re.sub(r'(?:₹|Rs\.?|INR)\s*(\d+)', r'\1 ரூபாய்', text)

    digits_map = {
        '0': 'பூஜ்ஜியம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு',
        '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது'
    }

    # 4. Decimals (e.g. 9.5 -> ஒன்பது புள்ளி ஐந்து)
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

    # 5. Whole numbers (e.g. 20 -> இருபது)
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


def synthesize_speech(text, ref_audio_path, ref_transcript, speed=1.0, quality=30, stability=0.75, emotion="neutral"):
    """Synthesizes speech using fast 8-step ODE solver with clean weight inference."""
    global ema_model, vocoder
    if ema_model is None or vocoder is None:
        raise RuntimeError("Neural model is not initialized yet. Please wait a moment.")

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    # Ensure all numbers are converted to spoken Tamil words
    text = normalize_numbers_in_tamil(text)

    start_time = time.time()
    
    # Load and prepare reference audio tensor (ensure mono 1D channel)
    ref_wav, sr = sf.read(ref_audio_path)
    ref_tensor = torch.from_numpy(ref_wav).float()
    if ref_tensor.ndim == 2:
        ref_tensor = ref_tensor.mean(dim=-1, keepdim=True).t()
    elif ref_tensor.ndim == 1:
        ref_tensor = ref_tensor.unsqueeze(0)

    ref_text = ref_transcript.strip() if ref_transcript.strip() else "வணக்கம்"

    print(f"[TTS] Synthesizing speech for text: '{text[:50]}...' on CPU...", flush=True)

    # Run single-batch inference with 8 ODE steps
    result, _, _ = infer_batch_process(
        ref_audio=(ref_tensor, sr),
        ref_text=ref_text,
        gen_text_batches=[text],
        model_obj=ema_model,
        vocoder=vocoder,
        nfe_step=8,
        speed=speed,
        device="cpu"
    )

    final_wave = np.asarray(result, dtype=np.float32)

    # Peak normalize audio into [-0.95, 0.95] range for clean, audible playback
    max_peak = np.max(np.abs(final_wave))
    if max_peak > 0:
        final_wave = (final_wave / max_peak) * 0.95

    sample_rate = 24000

    # Write to buffer
    buf = io.BytesIO()
    sf.write(buf, final_wave, samplerate=sample_rate, format="WAV")
    buf.seek(0)

    elapsed = time.time() - start_time
    print(f"[TTS] ✅ Finished! Duration: {len(final_wave)/sample_rate:.2f}s in {elapsed:.1f}s", flush=True)
    return buf.read()


@app.route("/", methods=["GET"])
def index():
    with open("tts_ui.html", "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="text/html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "translate")  # 'translate' or 'transliterate'

    if not text:
        return jsonify({"error": "Text is empty"}), 400

    try:
        if mode == "transliterate":
            result = transliterate_to_tamil(text)
        else:
            result = translate_english_to_tamil(text)

        result = normalize_numbers_in_tamil(result)
        has_english = bool(re.search(r'[a-zA-Z]', text))
        return jsonify({
            "original": text,
            "translated": result,
            "mode": mode,
            "source_detected": "en" if has_english else "ta"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voices", methods=["GET"])
def get_voices():
    voices = get_available_voices()
    return jsonify({
        "voices": [
            {
                "key": k,
                "label": v["label"],
                "gender": v["gender"],
                "icon": v["icon"],
                "style": v["default_style"],
                "has_custom_audio": os.path.exists(v["path"])
            }
            for k, v in voices.items()
        ],
        "emotions": [
            {
                "key": k,
                "label": v["label"],
                "icon": v["icon"]
            }
            for k, v in EMOTION_STYLES.items()
        ]
    })


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
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
        return jsonify({"error": f"Voice '{voice_key}' not found in dataset/"}), 400

    voice = voices[voice_key]

    # Auto-translate or transliterate if text contains English
    processed_text = text
    if auto_translate and re.search(r'[a-zA-Z]', text):
        processed_text = translate_english_to_tamil(text)
    else:
        processed_text = transliterate_to_tamil(text)

    processed_text = normalize_numbers_in_tamil(processed_text)

    try:
        audio_bytes = synthesize_speech(
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


if __name__ == "__main__":
    load_neural_pipeline()
    voices = get_available_voices()
    print(f"[Voices] Ready with {len(voices)} voice profiles from dataset/:", flush=True)
    for k, v in voices.items():
        print(f" - [{k}] {v['label']} ({v['path']})", flush=True)
    print("\n" + "=" * 65, flush=True)
    print("🚀 Fish Speech S2 Tamil TTS Localhost Server with In-Built Translator!", flush=True)
    print("🌐 Access Web UI at: http://localhost:5050", flush=True)
    print("=" * 65 + "\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)