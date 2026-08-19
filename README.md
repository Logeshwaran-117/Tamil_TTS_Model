# 🎙️ Tamil TTS Model & Studio (Fish Speech S2 / IndicF5)

A comprehensive Tamil Text-to-Speech (TTS) fine-tuning, voice cloning, and interactive studio framework supporting multiple voice profiles, Google Colab GPU training, local CPU fine-tuning, automated Whisper transcription, and real-time generation.

---

## 🌟 Features

- **Multi-Speaker Dataset Pipeline**: Prepares audio, auto-trims silence, standardizes sampling rates, and auto-transcribes Tamil speech using OpenAI Whisper.
- **Fish Speech S2 & IndicF5 Support**: Architecture optimized for natural Tamil phoneme representation and high-fidelity speech synthesis.
- **Interactive Web UI**: Modern glassmorphic studio interface (`tts_ui.html`) with live playback, multi-speaker selection, speed/pitch controls, and audio export.
- **FastAPI / Python Backend**: High-performance streaming backend (`tts_backend.py`) for inference and model checkpoints.
- **Cloud & Local Training**:
  - `fish_speech_colab_training.ipynb`: One-click Google Colab pipeline with Free T4 GPU acceleration.
  - `train_local_cpu.py`: Lightweight CPU training script with gradient accumulation & loss tracking.

---

## 📁 Repository Structure

```
├── batch_process_all_voices.py       # Batch voice processing and dataset aggregator
├── build_training_dataset.py          # Builds standardized train/val sets & metadata
├── process_and_transcribe_voice.py    # Whisper-based Tamil speech transcription & cleanup
├── sync_custom_voices.py              # Custom voice profile synchronizer
├── train_local_cpu.py                 # Local training/fine-tuning pipeline for CPU
├── tts_backend.py                     # FastAPI TTS inference server
├── tts_ui.html                        # Web UI for speech synthesis & preview
├── fish_speech_colab_training.ipynb   # Colab notebook for GPU fine-tuning
├── custom_voice_recording_prompts.txt # Standardized phoneme prompts for recording
├── start_tts.bat                      # One-click launcher for backend server
├── run_cpu_training.bat               # One-click launcher for training
└── .gitignore                         # Configured to exclude heavy models & raw datasets
```

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ or 3.11 is installed.
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# Install requirements:
pip install torch torchaudio fastapi uvicorn openai-whisper gradio soundfile
```

### 2. Launch Studio Server
```bash
python tts_backend.py
```
Open `tts_ui.html` in your browser to interact with the TTS Studio.

### 3. Voice Dataset Preparation & Fine-Tuning
1. Place recorded voice clips in the respective speaker folders.
2. Run the preprocessing script:
   ```bash
   python batch_process_all_voices.py
   ```
3. To train on Google Colab GPU, open and execute `fish_speech_colab_training.ipynb`.
4. To train locally on CPU:
   ```bash
   python train_local_cpu.py
   ```

---

## 📄 License
MIT License
