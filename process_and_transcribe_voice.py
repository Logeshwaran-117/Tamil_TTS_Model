import os
import sys
import zipfile
import glob
import shutil
import soundfile as sf
import whisper

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def process_voice_zip(zip_path, speaker_key):
    """
    Extracts audio files from zip_path, cleans and standardizes them into dataset/<speaker_key>/,
    transcribes them using Whisper ASR into Tamil text, and prepares reference.wav + transcript.txt.
    """
    if not os.path.exists(zip_path):
        print(f"Error: Zip file {zip_path} not found!")
        return False
        
    target_dir = os.path.join("dataset", speaker_key)
    cache_dir = os.path.join("cache", "whisper")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n=======================================================", flush=True)
    print(f"  Processing & Transcribing: {speaker_key} from {zip_path}", flush=True)
    print(f"=======================================================", flush=True)
    
    temp_extract = os.path.join("dataset", f"{speaker_key}_temp")
    os.makedirs(temp_extract, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
        
    # Find all wav files in extracted directory
    wav_paths = sorted(glob.glob(os.path.join(temp_extract, "**", "*.wav"), recursive=True))
    if not wav_paths:
        # Also check for mp3/flac/m4a if any
        wav_paths = sorted(glob.glob(os.path.join(temp_extract, "**", "*.[mwf][pa3][vcl]*"), recursive=True))
        
    print(f"Found {len(wav_paths)} audio files for {speaker_key}.", flush=True)
    
    # Clean existing files in dataset target directory
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
            
    # Load Whisper model
    print("Loading Whisper ASR model for Tamil speech recognition...", flush=True)
    model = whisper.load_model("small", download_root=cache_dir)
    
    transcripts = []
    for idx, src_wav in enumerate(wav_paths, start=1):
        dst_wav_name = f"reference_{idx:03d}.wav"
        dst_txt_name = f"reference_{idx:03d}.txt"
        dst_wav_path = os.path.join(target_dir, dst_wav_name)
        dst_txt_path = os.path.join(target_dir, dst_txt_name)
        
        # Read and write standardized WAV
        data, sr = sf.read(src_wav)
        sf.write(dst_wav_path, data, sr)
        
        # Run Whisper transcription
        result = model.transcribe(dst_wav_path, language="ta", fp16=False)
        tamil_text = result["text"].strip()
        
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(tamil_text)
            
        transcripts.append((dst_wav_name, tamil_text))
        print(f"[{idx:03d}/{len(wav_paths)}] {dst_wav_name} -> {tamil_text}", flush=True)
        
    if transcripts:
        shutil.copyfile(os.path.join(target_dir, "reference_001.wav"), os.path.join(target_dir, "reference.wav"))
        with open(os.path.join(target_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcripts[0][1])
            
    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"\n✅ {speaker_key}: All {len(wav_paths)} clips processed and transcribed!\n", flush=True)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_and_transcribe_voice.py <zip_path> <speaker_key>")
        print("Example: python process_and_transcribe_voice.py female_2.zip female_2")
        sys.exit(1)
        
    zip_p = sys.argv[1]
    spk_k = sys.argv[2]
    success = process_voice_zip(zip_p, spk_k)
    if success:
        # Automatically rebuild training dataset manifest & zip
        import subprocess
        subprocess.run([sys.executable, "build_training_dataset.py"])
