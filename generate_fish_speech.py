"""
🐟 Fish Speech S2 — Local Audio Generation Script
Run Fish Speech S2 Dual-AR + DAC VQ-GAN model locally on CPU or CUDA.
"""

import os
import sys
import argparse
import subprocess

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_audio_fish_speech(
    text: str,
    prompt_text: str = "வணக்கம்",
    prompt_audio: str = "dataset/female_1/reference.wav",
    checkpoint_path: str = "checkpoints/s2-pro",
    output_path: str = "fish_speech_output.wav",
    device: str = "cpu",
    temperature: float = 0.7,
    top_p: float = 0.85,
    top_k: int = 30
):
    """
    Runs the official Fish Speech S2 inference pipeline locally.
    """
    python_exe = sys.executable

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    if prompt_audio and not os.path.exists(prompt_audio):
        print(f"⚠️ Warning: Reference audio '{prompt_audio}' not found. Generating without reference.")
        prompt_audio = None

    cmd = [
        python_exe,
        "-m", "fish_speech.models.text2semantic.inference",
        "--text", text,
        "--checkpoint-path", checkpoint_path,
        "--device", device,
        "--output", output_path,
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--no-compile"
    ]

    if prompt_text and prompt_audio:
        cmd.extend(["--prompt-text", prompt_text, "--prompt-audio", prompt_audio])

    print("\n" + "=" * 60)
    print("🐟 Fish Speech S2 Local Generation")
    print(f"📝 Text:             {text}")
    print(f"🎙️ Reference Audio:  {prompt_audio if prompt_audio else 'None (Default zero-shot)'}")
    print(f"💻 Device:           {device.upper()}")
    print(f"💾 Output File:      {os.path.abspath(output_path)}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd)
    if result.returncode == 0 and os.path.exists(output_path):
        print(f"\n✅ Success! Generated audio saved to: {os.path.abspath(output_path)}")
        return output_path
    else:
        print(f"\n❌ Inference failed with exit code: {result.returncode}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech locally with Fish Speech S2")
    parser.add_argument("--text", type=str, default="வணக்கம்! இது பிஷ் ஸ்பீச் எஸ்2 மாடலில் உருவாக்கப்பட்ட குரல்.", help="Tamil text to speak")
    parser.add_argument("--prompt-text", type=str, default="வணக்கம்", help="Prompt transcript of reference audio")
    parser.add_argument("--prompt-audio", type=str, default="dataset/female_1/reference.wav", help="Path to reference WAV audio")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/s2-pro", help="Path to s2-pro checkpoint folder")
    parser.add_argument("--output", type=str, default="fish_speech_output.wav", help="Output WAV path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    generate_audio_fish_speech(
        text=args.text,
        prompt_text=args.prompt_text,
        prompt_audio=args.prompt_audio,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        device=args.device,
        temperature=args.temperature
    )
