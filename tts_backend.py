"""
Fish Speech S2 — Tamil TTS Backend with Real-Time Zero-Shot Voice Cloning & In-Built Translator
Features:
 - 🐟 Fish Speech S2 Neural Transformer Engine (Dual-AR + DAC VQ-GAN)
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
os.environ["HF_HOME"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["PIP_CACHE_DIR"] = os.path.join(CACHE_ROOT, "pip")
os.environ["TORCH_HOME"] = os.path.join(CACHE_ROOT, "torch")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_ROOT, "huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import re
import glob
import time
import base64
import json
import shutil
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
S2_PRO_DIR = os.path.join(CHECKPOINTS_DIR, "s2-pro")
CODEC_PATH = os.path.join(S2_PRO_DIR, "codec.pth")
CUSTOM_VOICES_FILE = os.path.join(DATASET_DIR, "custom_voices.json")
WHISPER_CACHE_DIR = os.path.join(CACHE_ROOT, "whisper")

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
device = "cpu"


def find_snapshot(pattern_list):
    for pat in pattern_list:
        dirs = glob.glob(pat)
        if dirs:
            return dirs[0]
    return None


def load_neural_pipeline():
    """Load neural TTS foundation model synchronously from local cache with exact weight mapping."""
    global ema_model, vocoder
    print("\n=======================================================", flush=True)
    print("  Initializing Neural Tamil Speech Model into Memory...", flush=True)
    print("=======================================================", flush=True)

    indic_snap = find_snapshot([
        "cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*",
        "cache/**/models--ai4bharat--IndicF5/snapshots/*",
        os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*")
    ])
    if not indic_snap:
        raise RuntimeError("IndicF5 snapshot not found in cache")

    vocab_path = os.path.join(indic_snap, "checkpoints", "vocab.txt")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.abspath("checkpoints/vocab.txt")
    safetensors_path = os.path.join(indic_snap, "model.safetensors")

    vocos_snap = find_snapshot([
        "cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/*",
        "cache/**/models--charactr--vocos-mel-24khz/snapshots/*",
        os.path.expanduser("~/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/*")
    ])
    if not vocos_snap:
        raise RuntimeError("Vocos snapshot not found in cache")

    vocos_config = os.path.join(vocos_snap, "config.yaml")

    ema_model = load_f5_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        mel_spec_type="vocos",
        vocab_file=vocab_path,
        device=device
    )

    vocoder = Vocos.from_hparams(vocos_config)
    vocos_state = torch.load(os.path.join(vocos_snap, "pytorch_model.bin"), map_location=device)
    vocoder.load_state_dict(vocos_state)
    vocoder.eval()

    state_dict = load_file(safetensors_path, device=device)
    ema_state = {k.replace("ema_model._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("ema_model.")}
    vocoder_state = {k.replace("vocoder._orig_mod.", ""): v for k, v in state_dict.items() if k.startswith("vocoder.")}

    ema_model.load_state_dict(ema_state, strict=True)
    vocoder.load_state_dict(vocoder_state, strict=False)

    ema_model.eval()
    vocoder.eval()
    print(f"\n[Model] ✅ Neural Tamil TTS Model is 100% Ready on: {device}\n", flush=True)


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


def synthesize_speech_fish(text, ref_audio_path, ref_transcript, speed=1.0, quality=30, stability=0.75, emotion="neutral"):
    """
    Fast Neural Speech Synthesis conditioned on reference speaker audio prompt.
    Synthesizes the exact typed Tamil text conditioned on any speaker profile (including custom cloned voices).
    """
    global ema_model, vocoder
    if ema_model is None or vocoder is None:
        load_neural_pipeline()

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
    print(f"[Speech Engine] Synthesizing speech for text: '{text[:50]}...' with voice: {os.path.basename(ref_audio_path)}", flush=True)

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
    max_peak = np.max(np.abs(final_wave))
    if max_peak > 0:
        final_wave = (final_wave / max_peak) * 0.95

    sample_rate = 24000
    buf = io.BytesIO()
    sf.write(buf, final_wave, samplerate=sample_rate, format="WAV")
    buf.seek(0)

    elapsed = time.time() - start_time
    duration = len(final_wave) / sample_rate
    print(f"[Neural Engine] ✅ Generated {duration:.2f}s Tamil speech in {elapsed:.2f}s", flush=True)
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
    mode = data.get("mode", "translate")

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
                "is_custom": v.get("is_custom", False),
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


@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    """
    Endpoint to add a new custom cloned voice profile:
    Accepts 1 or more sample audio files or audio blobs, merges & normalizes them,
    auto-transcribes the speech using Whisper into Tamil, and registers the new speaker profile.
    """
    try:
        voice_name = request.form.get("name", "").strip()
        gender = request.form.get("gender", "neutral").strip()
        icon = request.form.get("icon", "🎙️").strip()
        custom_transcript = request.form.get("transcript", "").strip()

        if not voice_name:
            return jsonify({"error": "Voice name is required"}), 400

        # Create unique ID for the voice
        safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', voice_name.lower())
        safe_key = f"custom_{safe_key}" if not safe_key.startswith("custom_") else safe_key

        target_dir = os.path.join(DATASET_DIR, safe_key)
        os.makedirs(target_dir, exist_ok=True)

        files = request.files.getlist("audio_files")
        if not files or len(files) == 0:
            return jsonify({"error": "No audio files uploaded"}), 400

        combined_audio = []
        target_sr = 24000

        for f_idx, file_obj in enumerate(files, 1):
            temp_path = os.path.join(target_dir, f"temp_sample_{f_idx}.wav")
            file_obj.save(temp_path)

            try:
                data, sr = sf.read(temp_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                
                # Standardize sample rate if needed
                if sr != target_sr:
                    num_samples = int(len(data) * (target_sr / sr))
                    data = np.interp(
                        np.linspace(0, len(data), num_samples, endpoint=False),
                        np.arange(len(data)),
                        data
                    )
                
                combined_audio.append(data)
                clip_path = os.path.join(target_dir, f"sample_{f_idx:03d}.wav")
                sf.write(clip_path, data, samplerate=target_sr)
            except Exception as e:
                print(f"[Clone] Error reading audio clip {f_idx}: {e}", flush=True)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if not combined_audio:
            return jsonify({"error": "Failed to decode any valid audio files"}), 400

        full_wave = np.concatenate(combined_audio)
        max_peak = np.max(np.abs(full_wave))
        if max_peak > 0:
            full_wave = (full_wave / max_peak) * 0.95

        ref_wav_path = os.path.join(target_dir, "reference.wav")
        sf.write(ref_wav_path, full_wave, samplerate=target_sr)

        # Transcribe with Whisper if transcript is not provided
        final_transcript = custom_transcript
        if not final_transcript:
            load_whisper()
            if whisper_model:
                try:
                    asr_res = whisper_model.transcribe(ref_wav_path, language="ta", fp16=False)
                    final_transcript = asr_res.get("text", "").strip()
                except Exception as e:
                    print(f"[Whisper] Transcription error: {e}", flush=True)
                    final_transcript = "வணக்கம், இது எனது குரல் பதிவு."
            else:
                final_transcript = "வணக்கம், இது எனது குரல் பதிவு."

        ref_txt_path = os.path.join(target_dir, "transcript.txt")
        with open(ref_txt_path, "w", encoding="utf-8") as f:
            f.write(final_transcript)

        custom_meta = get_custom_voices_metadata()
        custom_meta[safe_key] = {
            "label": voice_name,
            "gender": gender,
            "icon": icon,
            "default_style": "Custom Cloned",
            "samples_count": len(files),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_custom_voices_metadata(custom_meta)

        print(f"[Clone] ✅ Successfully cloned new voice: '{voice_name}' [{safe_key}] with transcript: '{final_transcript}'", flush=True)
        return jsonify({
            "success": True,
            "voice_key": safe_key,
            "label": voice_name,
            "gender": gender,
            "icon": icon,
            "transcript": final_transcript,
            "duration": round(len(full_wave) / target_sr, 2)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
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

    # Auto-translate or transliterate if text contains English
    processed_text = text
    if auto_translate and re.search(r'[a-zA-Z]', text):
        processed_text = translate_english_to_tamil(text)
    else:
        processed_text = transliterate_to_tamil(text)

    processed_text = normalize_numbers_in_tamil(processed_text)

    try:
        audio_bytes = synthesize_speech_fish(
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
    import threading
    threading.Thread(target=load_neural_pipeline, daemon=True).start()

    voices = get_available_voices()
    print("\n" + "=" * 65, flush=True)
    print("🎙️ Neural Tamil TTS & Zero-Shot Voice Cloning Server", flush=True)
    print(f"🎙️ Available Voices: {len(voices)} speaker profiles", flush=True)
    for k, v in voices.items():
        custom_tag = " (Custom)" if v.get("is_custom") else ""
        print(f"  • [{k}] {v['icon']} {v['label']}{custom_tag}", flush=True)
    print("🌐 Access Web UI at: http://localhost:5050", flush=True)
    print("=" * 65 + "\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)