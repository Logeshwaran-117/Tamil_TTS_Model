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

whisper_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Official Fish Speech S2 State
fish_model = None
fish_decode_func = None
fish_codec = None
using_fish_s2 = False


def load_codec_model_robust(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding robustly across any fish-speech version."""
    try:
        from fish_speech.models.text2semantic.inference import load_codec_model
        return load_codec_model(codec_checkpoint_path, device, precision)
    except Exception:
        pass

    try:
        from fish_speech.models.dac.inference import load_model as load_dac_model
        return load_dac_model("modded_dac_vq", codec_checkpoint_path, device=device)
    except Exception:
        pass

    import fish_speech
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    config_path = os.path.join(fish_speech.__path__[0], "configs", "modded_dac_vq.yaml")
    cfg = OmegaConf.load(config_path)
    codec = instantiate(cfg)
    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)
    return codec


def encode_audio_robust(audio_path, codec, device):
    """Encode audio into VQ prompt tokens."""
    try:
        from fish_speech.models.text2semantic.inference import encode_audio
        return encode_audio(audio_path, codec, device)
    except Exception:
        pass

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)
    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]


def decode_to_audio_robust(codes, codec):
    """Decode VQ tokens to audio waveform."""
    try:
        from fish_speech.models.text2semantic.inference import decode_to_audio
        return decode_to_audio(codes, codec)
    except Exception:
        pass
    audio = codec.from_indices(codes[None])
    return audio[0, 0]


def patch_llama_for_fish_qwen3_omni():
    """Patches older fish-speech llama module to support fish_qwen3_omni architecture."""
    try:
        import dataclasses
        from pathlib import Path
        import fish_speech.models.text2semantic.llama as llama_mod
        orig_from_pretrained = llama_mod.BaseModelArgs.from_pretrained

        def patched_from_pretrained(path):
            p = Path(path)
            if p.is_dir():
                p = p / "config.json"
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model_type") == "fish_qwen3_omni" and not hasattr(llama_mod.BaseModelArgs, "_from_fish_qwen3_omni"):
                tc = data.get("text_config", {})
                adc = data.get("audio_decoder_config", {})
                flat = dict(
                    model_type="dual_ar",
                    vocab_size=tc.get("vocab_size", 32000),
                    n_layer=tc.get("n_layer", 32),
                    n_head=tc.get("n_head", 32),
                    n_local_heads=tc.get("n_local_heads", -1),
                    head_dim=tc.get("head_dim"),
                    dim=tc.get("dim", 2560),
                    intermediate_size=tc.get("intermediate_size"),
                    rope_base=tc.get("rope_base", 10000),
                    norm_eps=tc.get("norm_eps", 1e-5),
                    max_seq_len=tc.get("max_seq_len", 2048),
                    dropout=tc.get("dropout", 0.0),
                    tie_word_embeddings=tc.get("tie_word_embeddings", True),
                    attention_qkv_bias=tc.get("attention_qkv_bias", False),
                    attention_o_bias=tc.get("attention_o_bias", False),
                    attention_qk_norm=tc.get("attention_qk_norm", False),
                    use_gradient_checkpointing=tc.get("use_gradient_checkpointing", True),
                    initializer_range=tc.get("initializer_range", 0.02),
                    semantic_begin_id=data.get("semantic_start_token_id", 0),
                    semantic_end_id=data.get("semantic_end_token_id", 0),
                    scale_codebook_embeddings=True,
                    norm_fastlayer_input=True,
                    audio_embed_dim=adc.get("text_dim", tc.get("dim", 2560)),
                    codebook_size=adc.get("vocab_size", 4096),
                    num_codebooks=adc.get("num_codebooks", 10),
                    n_fast_layer=adc.get("n_layer", 4),
                    fast_dim=adc.get("dim"),
                    fast_n_head=adc.get("n_head"),
                    fast_n_local_heads=adc.get("n_local_heads"),
                    fast_head_dim=adc.get("head_dim"),
                    fast_intermediate_size=adc.get("intermediate_size"),
                    fast_attention_qkv_bias=adc.get("attention_qkv_bias"),
                    fast_attention_qk_norm=adc.get("attention_qk_norm"),
                    fast_attention_o_bias=adc.get("attention_o_bias"),
                )
                valid_keys = {f.name for f in dataclasses.fields(llama_mod.DualARModelArgs)}
                flat = {k: v for k, v in flat.items() if k in valid_keys}
                return llama_mod.DualARModelArgs(**flat)
            return orig_from_pretrained(path)

        llama_mod.BaseModelArgs.from_pretrained = staticmethod(patched_from_pretrained)
    except Exception:
        pass


model_load_lock = threading.Lock()
model_is_loading = False


def load_fish_speech_s2_pipeline():
    """Load official Fish Speech S2 Dual-AR Transformer + DAC VQ-GAN vocoder."""
    global fish_model, fish_decode_func, fish_codec, using_fish_s2, model_is_loading
    with model_load_lock:
        if using_fish_s2 and fish_model is not None and fish_codec is not None:
            return True
        if model_is_loading:
            return False
        model_is_loading = True
        try:
            patch_llama_for_fish_qwen3_omni()
            from fish_speech.models.text2semantic.inference import init_model
            print("\n=======================================================", flush=True)
            print(f"  🐟 Initializing Official Fish Speech S2 Dual-AR Model ({device.upper()})...", flush=True)
            print("=======================================================", flush=True)

            if not os.path.exists(os.path.join(S2_PRO_DIR, "codec.pth")):
                from huggingface_hub import snapshot_download
                print("[Fish S2] Downloading s2-pro weights from Hugging Face...", flush=True)
                snapshot_download(repo_id="fishaudio/s2-pro", local_dir=S2_PRO_DIR, local_dir_use_symlinks=False)

            precision = torch.bfloat16 if device == "cuda" else torch.float32
            fish_model, fish_decode_func = init_model(S2_PRO_DIR, device=device, precision=precision, compile=False)
            with torch.device(device):
                fish_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=min(fish_model.config.max_seq_len, 2048),
                    dtype=next(fish_model.parameters()).dtype
                )
            codec_ckpt = os.path.join(S2_PRO_DIR, "codec.pth")
            fish_codec = load_codec_model_robust(codec_ckpt, device=device, precision=precision)
            using_fish_s2 = True
            model_is_loading = False
            print(f"[Fish S2] ✅ Official Fish Speech S2 Dual-AR Engine is 100% Ready on {device}!\n", flush=True)
            return True
        except Exception as e:
            model_is_loading = False
            import traceback
            traceback.print_exc()
            print(f"[Fish S2] Error loading Fish Speech S2: {e}", flush=True)
            return False


def load_whisper():
    """Loads Whisper ASR model for Tamil speech recognition and auto-transcription."""
    global whisper_model
    if whisper_model is None:
        try:
            # pyrefly: ignore [missing-import]
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
    Synthesizes speech using official Fish Speech S2 Dual-AR Transformer + DAC VQ-GAN vocoder.
    """
    global fish_model, fish_codec, fish_decode_func, using_fish_s2
    if not using_fish_s2 or fish_model is None or fish_codec is None:
        load_fish_speech_s2_pipeline()

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    # Ensure all numbers are converted to spoken Tamil words
    text = normalize_numbers_in_tamil(text)
    start_time = time.time()

    # pyrefly: ignore [missing-import]
    from fish_speech.models.text2semantic.inference import generate_long
    print(f"[Fish Speech S2] Dual-AR Synthesis for: '{text[:50]}...' with voice: {os.path.basename(ref_audio_path)}", flush=True)
    
    prompt_tokens = [encode_audio_robust(ref_audio_path, fish_codec, device).cpu()]
    prompt_text = [ref_transcript.strip() if ref_transcript.strip() else "வணக்கம்"]
    
    temperature = max(0.2, min(float(stability), 1.2))
    generator = generate_long(
        model=fish_model,
        device=device,
        decode_one_token=fish_decode_func,
        text=text,
        temperature=temperature,
        top_p=0.85,
        top_k=30,
        compile=False,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens
    )
    
    codes = []
    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
        elif response.action == "next" and codes:
            merged_codes = torch.cat(codes, dim=1)
            audio_tensor = decode_to_audio_robust(merged_codes.to(device), fish_codec)
            final_wave = audio_tensor.cpu().float().numpy()
            max_peak = np.max(np.abs(final_wave))
            if max_peak > 0:
                final_wave = (final_wave / max_peak) * 0.95
            
            buf = io.BytesIO()
            sf.write(buf, final_wave, samplerate=fish_codec.sample_rate, format="WAV")
            buf.seek(0)
            elapsed = time.time() - start_time
            duration = len(final_wave) / fish_codec.sample_rate
            print(f"[Fish Speech S2] ✅ Generated {duration:.2f}s Tamil speech in {elapsed:.2f}s", flush=True)
            return buf.read()
    
    raise RuntimeError("Fish Speech S2 did not generate audio output.")


@app.route("/", methods=["GET"])
def index():
    with open("tts_ui.html", "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if using_fish_s2 and fish_model is not None else ("loading" if model_is_loading else "idle"),
        "device": device,
        "cuda": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "voices_count": len(get_available_voices())
    })


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "🎙️ Tamil Neural TTS & Voice Cloning API",
            "version": "2.0.0",
            "description": "High-performance Tamil Text-to-Speech API with real-time zero-shot voice cloning, dynamic multi-speaker profiles, English-to-Tamil neural translation, and phonetic transliteration."
        },
        "servers": [
            {"url": "http://localhost:5050", "description": "Local Development Server"}
        ],
        "tags": [
            {"name": "Speech Synthesis", "description": "Generate Tamil speech audio from text"},
            {"name": "Voice Management", "description": "Fetch available voices and clone custom voices"},
            {"name": "Translation", "description": "Translate English or Tanglish text to Tamil"}
        ],
        "paths": {
            "/voices": {
                "get": {
                    "tags": ["Voice Management"],
                    "summary": "Get all available voice profiles & emotions",
                    "description": "Returns preset speaker profiles, custom cloned voices, and available emotion styles.",
                    "responses": {
                        "200": {
                            "description": "List of voices and emotions",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "voices": [
                                            {"key": "female_1", "label": "Female 1 (Narrator)", "gender": "female", "icon": "👩", "style": "Narrator", "is_custom": False},
                                            {"key": "male_1", "label": "Male 1 (Narrator)", "gender": "male", "icon": "🧔", "style": "Narrator", "is_custom": False}
                                        ],
                                        "emotions": [
                                            {"key": "neutral", "label": "Neutral / Normal", "icon": "😐"},
                                            {"key": "cheerful", "label": "Happy & Cheerful", "icon": "😊"},
                                            {"key": "narrator", "label": "Storyteller / Drama", "icon": "📖"},
                                            {"key": "formal", "label": "Formal / News", "icon": "👔"},
                                            {"key": "surprised", "label": "Excited & Surprised", "icon": "😲"}
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/generate": {
                "post": {
                    "tags": ["Speech Synthesis"],
                    "summary": "Generate Tamil speech from text",
                    "description": "Synthesizes natural Tamil speech using Fish Speech S2 / F5-TTS with custom voice and emotion parameters.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "example": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
                                            "description": "Tamil text, Tanglish, or English to synthesize"
                                        },
                                        "voice": {
                                            "type": "string",
                                            "default": "female_1",
                                            "example": "female_1",
                                            "description": "Speaker key (e.g. female_1, male_1, or custom cloned key)"
                                        },
                                        "emotion": {
                                            "type": "string",
                                            "default": "neutral",
                                            "enum": ["neutral", "cheerful", "narrator", "formal", "surprised"],
                                            "description": "Voice emotion style"
                                        },
                                        "speed": {
                                            "type": "number",
                                            "default": 1.0,
                                            "minimum": 0.5,
                                            "maximum": 2.0,
                                            "description": "Speech speed multiplier"
                                        },
                                        "quality": {
                                            "type": "integer",
                                            "default": 30,
                                            "minimum": 16,
                                            "maximum": 64,
                                            "description": "Inference quality (NFE steps)"
                                        },
                                        "stability": {
                                            "type": "number",
                                            "default": 0.75,
                                            "minimum": 0.1,
                                            "maximum": 1.0,
                                            "description": "Voice stability factor"
                                        },
                                        "auto_translate": {
                                            "type": "boolean",
                                            "default": True,
                                            "description": "Automatically translate English words into Tamil"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Speech successfully generated",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "audio_b64": "UklGRuQEAABXQVZFZm10IBAAAAABAAEA...",
                                        "voice": "female_1",
                                        "voice_label": "Female 1 (Narrator)",
                                        "speed": 1.0,
                                        "quality": 30,
                                        "stability": 0.75,
                                        "emotion": "neutral",
                                        "transliterated_text": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"
                                    }
                                }
                            }
                        },
                        "400": {"description": "Invalid input text or parameters"},
                        "500": {"description": "Server synthesis error"}
                    }
                }
            },
            "/clone_voice": {
                "post": {
                    "tags": ["Voice Management"],
                    "summary": "Clone a new voice profile (Zero-Shot)",
                    "description": "Upload reference audio clips of a speaker to register an instant zero-shot cloned voice profile.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name", "audio_files"],
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "example": "MyCustomVoice",
                                            "description": "Unique display name for the cloned voice"
                                        },
                                        "gender": {
                                            "type": "string",
                                            "enum": ["male", "female", "neutral"],
                                            "default": "neutral",
                                            "description": "Speaker gender"
                                        },
                                        "icon": {
                                            "type": "string",
                                            "default": "🎙️",
                                            "description": "Emoji icon for the profile"
                                        },
                                        "transcript": {
                                            "type": "string",
                                            "description": "Tamil transcript of reference audio (optional, auto-transcribed via Whisper if empty)"
                                        },
                                        "audio_files": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "binary"},
                                            "description": "One or more audio files (.wav or .mp3, 3-10 sec)"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Voice cloned successfully",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "success": True,
                                        "voice_key": "custom_mycustomvoice",
                                        "label": "MyCustomVoice",
                                        "gender": "male",
                                        "icon": "🎙️",
                                        "samples_count": 1,
                                        "primary_transcript": "வணக்கம்"
                                    }
                                }
                            }
                        },
                        "400": {"description": "Missing name or audio files"}
                    }
                }
            },
            "/translate": {
                "post": {
                    "tags": ["Translation"],
                    "summary": "Translate English or Tanglish to Tamil",
                    "description": "Converts English text or Tanglish phonetic spelling into clean Tamil script.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "example": "vanakkam nanba eppadi irukinga",
                                            "description": "English sentence or Tanglish text"
                                        },
                                        "mode": {
                                            "type": "string",
                                            "enum": ["translate", "transliterate"],
                                            "default": "translate",
                                            "description": "Neural translation ('translate') or phonetic transliteration ('transliterate')"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Translation successful",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "original": "vanakkam nanba eppadi irukinga",
                                        "translated": "வணக்கம் நண்பா எப்படி இருக்கீங்க",
                                        "mode": "transliterate",
                                        "source_detected": "en"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
@app.route("/swagger", methods=["GET"])
def swagger_ui():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Tamil TTS API — Swagger UI</title>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5/favicon-32x32.png" />
  <style>
    html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
    *, *:before, *:after { box-sizing: inherit; }
    body { margin: 0; background: #fafafa; font-family: sans-serif; }
    .topbar { display: none !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" charset="UTF-8"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js" charset="UTF-8"></script>
  <script>
    window.onload = function() {
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


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


def sync_voice_to_training_dataset(safe_key, sample_refs, gender):
    """
    Syncs new voice reference WAVs and transcripts into training_dataset/wavs,
    updates metadata.csv, train.txt, val.txt, and dataset_summary.json.
    """
    try:
        train_dir = "training_dataset"
        wavs_dir = os.path.join(train_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)

        meta_csv_path = os.path.join(train_dir, "metadata.csv")
        existing_rows = []
        existing_files = set()

        if os.path.exists(meta_csv_path):
            with open(meta_csv_path, "r", encoding="utf-8") as f:
                _ = f.readline()  # skip header
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 5:
                        try:
                            dur = float(parts[4])
                        except Exception:
                            dur = 0.0
                        existing_rows.append({
                            "audio_file": parts[0],
                            "transcript": parts[1],
                            "speaker_id": parts[2],
                            "gender": parts[3],
                            "duration": dur
                        })
                        existing_files.add(parts[0])

        for idx, ref in enumerate(sample_refs, 1):
            train_wav_name = f"{safe_key}_{idx:04d}.wav"
            train_wav_path = os.path.join(wavs_dir, train_wav_name)
            shutil.copyfile(ref["wav"], train_wav_path)

            row = {
                "audio_file": train_wav_name,
                "transcript": ref["txt"],
                "speaker_id": safe_key,
                "gender": gender,
                "duration": ref["duration"]
            }
            if train_wav_name in existing_files:
                for r in existing_rows:
                    if r["audio_file"] == train_wav_name:
                        r.update(row)
            else:
                existing_rows.append(row)
                existing_files.add(train_wav_name)

        # Write updated metadata.csv
        with open(meta_csv_path, "w", encoding="utf-8") as f:
            f.write("audio_file|transcript|speaker_id|gender|duration\n")
            for r in existing_rows:
                f.write(f"{r['audio_file']}|{r['transcript']}|{r['speaker_id']}|{r['gender']}|{r['duration']}\n")

        # Write train.txt & val.txt
        import random
        random.seed(42)
        shuffled = list(existing_rows)
        random.shuffle(shuffled)
        split_idx = max(int(len(shuffled) * 0.85), 1)
        train_set = shuffled[:split_idx]
        val_set = shuffled[split_idx:]

        with open(os.path.join(train_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train_set:
                f.write(f"wavs/{m['audio_file']}|{m['transcript']}|{m['speaker_id']}\n")

        with open(os.path.join(train_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val_set:
                f.write(f"wavs/{m['audio_file']}|{m['transcript']}|{m['speaker_id']}\n")

        # Update dataset_summary.json
        summary = {
            "total_samples": len(existing_rows),
            "total_duration_seconds": round(sum(m["duration"] for m in existing_rows), 2),
            "speakers": {}
        }
        for m in existing_rows:
            spk = m["speaker_id"]
            if spk not in summary["speakers"]:
                summary["speakers"][spk] = {"count": 0, "total_duration": 0.0, "gender": m["gender"]}
            summary["speakers"][spk]["count"] += 1
            summary["speakers"][spk]["total_duration"] = round(summary["speakers"][spk]["total_duration"] + m["duration"], 2)

        with open(os.path.join(train_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Training Dataset] ✅ Synced '{safe_key}' ({len(sample_refs)} clips) to training_dataset/ (Total: {len(existing_rows)} samples)", flush=True)
    except Exception as e:
        print(f"[Training Dataset] Warning syncing to training dataset: {e}", flush=True)


@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    """
    Endpoint to add a new custom cloned voice profile:
    Accepts 1 or more sample audio files or audio blobs, merges & normalizes them,
    respects manual user transcript if provided (or auto-transcribes via Whisper if empty),
    registers the new speaker profile, and syncs to training_dataset/.
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

        sample_refs = []
        target_sr = 24000

        for f_idx, file_obj in enumerate(files, 1):
            temp_path = os.path.join(target_dir, f"temp_sample_{f_idx}.wav")
            file_obj.save(temp_path)

            try:
                data, sr = sf.read(temp_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                
                # Standardize sample rate to 24kHz
                if sr != target_sr:
                    num_samples = int(len(data) * (target_sr / sr))
                    data = np.interp(
                        np.linspace(0, len(data), num_samples, endpoint=False),
                        np.arange(len(data)),
                        data
                    )

                # Normalize individual clip
                clip_peak = np.max(np.abs(data))
                if clip_peak > 0:
                    data = (data / clip_peak) * 0.95

                ref_wav_name = f"reference_{f_idx:03d}.wav"
                ref_txt_name = f"reference_{f_idx:03d}.txt"
                ref_wav_path = os.path.join(target_dir, ref_wav_name)
                ref_txt_path = os.path.join(target_dir, ref_txt_name)
                sf.write(ref_wav_path, data, samplerate=target_sr)

                # Priority: 1. Manual user transcript, 2. Whisper auto-transcription, 3. Default fallback
                clip_transcript = ""
                if custom_transcript:
                    # User manually provided transcript - DO NOT overwrite with Whisper!
                    clip_transcript = custom_transcript
                    print(f"[Clone] Using manual user transcript for clip {f_idx:03d}: '{clip_transcript}'", flush=True)
                else:
                    load_whisper()
                    if whisper_model:
                        try:
                            asr_res = whisper_model.transcribe(ref_wav_path, language="ta", fp16=False)
                            clip_transcript = asr_res.get("text", "").strip()
                            print(f"[Whisper] Auto-transcribed clip {f_idx:03d}: '{clip_transcript}'", flush=True)
                        except Exception as e:
                            print(f"[Whisper] Sample {f_idx} transcription error: {e}", flush=True)

                if not clip_transcript:
                    clip_transcript = "வணக்கம்"

                with open(ref_txt_path, "w", encoding="utf-8") as f:
                    f.write(clip_transcript)

                sample_refs.append({
                    "wav": ref_wav_path,
                    "txt": clip_transcript,
                    "duration": round(len(data) / target_sr, 2)
                })
                print(f"[Clone] Clip {f_idx:03d} saved: '{ref_wav_name}' ({len(data)/target_sr:.2f}s) -> '{clip_transcript}'", flush=True)

            except Exception as e:
                print(f"[Clone] Error processing sample {f_idx}: {e}", flush=True)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if not sample_refs:
            return jsonify({"error": "Failed to decode any valid audio files"}), 400

        # Set default primary reference to the first clean sample
        shutil.copyfile(sample_refs[0]["wav"], os.path.join(target_dir, "reference.wav"))
        with open(os.path.join(target_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(sample_refs[0]["txt"])

        custom_meta = get_custom_voices_metadata()
        custom_meta[safe_key] = {
            "label": voice_name,
            "gender": gender,
            "icon": icon,
            "default_style": "Custom Cloned",
            "samples_count": len(sample_refs),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_custom_voices_metadata(custom_meta)

        # Sync cloned voice clips directly into training_dataset/
        sync_voice_to_training_dataset(safe_key, sample_refs, gender)

        print(f"[Clone] ✅ Successfully cloned '{voice_name}' with {len(sample_refs)} distinct reference samples and transcripts!", flush=True)
        return jsonify({
            "success": True,
            "voice_key": safe_key,
            "label": voice_name,
            "gender": gender,
            "icon": icon,
            "samples_count": len(sample_refs),
            "primary_transcript": sample_refs[0]["txt"]
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


@app.route("/fine_tune", methods=["POST"])
def fine_tune():
    """Runs Fish Speech S2 multi-speaker fine-tuning & adaptation on dataset/training_dataset."""
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
    load_fish_speech_s2_pipeline()

    voices = get_available_voices()
    print("\n" + "=" * 65, flush=True)
    print("🎙️ Neural Tamil TTS & Zero-Shot Voice Cloning Server", flush=True)
    print(f"🎙️ Available Voices: {len(voices)} speaker profiles", flush=True)
    for k, v in voices.items():
        custom_tag = " (Custom)" if v.get("is_custom") else ""
        print(f"  • [{k}] {v['icon']} {v['label']}{custom_tag}", flush=True)
    print("🌐 Access Web UI at:       http://localhost:5050", flush=True)
    print("📖 Swagger API Docs at:    http://localhost:5050/docs", flush=True)
    print("📋 OpenAPI Spec JSON at:   http://localhost:5050/openapi.json", flush=True)
    print("=" * 65 + "\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)