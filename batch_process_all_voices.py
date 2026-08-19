import os
import sys
import zipfile
import glob
import shutil
import time
import soundfile as sf
import whisper

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

JOBS = [
    {"zip": "female_2.zip", "key": "female_2", "title": "Female 2 (Conversational)"},
    {"zip": "female_3.zip", "key": "female_3_own", "title": "Female 3 (Expressive)"},
    {"zip": "male_1.zip", "key": "male_1", "title": "Male 1 (Narrator)"},
    {"zip": "male_2.zip", "key": "male_2", "title": "Male 2 (Conversational)"},
]

CACHE_DIR = os.path.join("cache", "whisper")
os.makedirs(CACHE_DIR, exist_ok=True)

print("Loading Whisper ASR model for Tamil speech recognition...", flush=True)
model = whisper.load_model("small", download_root=CACHE_DIR)

total_processed = 0
start_time = time.time()

for job in JOBS:
    zip_path = job["zip"]
    speaker_key = job["key"]
    speaker_title = job["title"]
    
    if not os.path.exists(zip_path):
        print(f"Skipping {zip_path} (file not found)", flush=True)
        continue
        
    target_dir = os.path.join("dataset", speaker_key)
    os.makedirs(target_dir, exist_ok=True)
    temp_extract = os.path.join("dataset", f"{speaker_key}_temp")
    os.makedirs(temp_extract, exist_ok=True)
    
    print(f"\n=======================================================", flush=True)
    print(f"  Extracting & Transcribing: {speaker_title} [{speaker_key}]", flush=True)
    print(f"=======================================================", flush=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
        
    wav_paths = sorted(glob.glob(os.path.join(temp_extract, "**", "*.wav"), recursive=True))
    print(f"Found {len(wav_paths)} audio files in {zip_path}.", flush=True)
    
    # Clean previous files in target directory
    for f in os.listdir(target_dir):
        fp = os.path.join(target_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
            
    transcripts = []
    for idx, src_wav in enumerate(wav_paths, start=1):
        dst_wav_name = f"reference_{idx:03d}.wav"
        dst_txt_name = f"reference_{idx:03d}.txt"
        dst_wav_path = os.path.join(target_dir, dst_wav_name)
        dst_txt_path = os.path.join(target_dir, dst_txt_name)
        
        # Read & save clean WAV
        data, sr = sf.read(src_wav)
        sf.write(dst_wav_path, data, sr)
        
        # Transcribe to Tamil Unicode text
        result = model.transcribe(dst_wav_path, language="ta", fp16=False)
        tamil_text = result["text"].strip()
        
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(tamil_text)
            
        transcripts.append((dst_wav_name, tamil_text))
        total_processed += 1
        print(f"[{speaker_key} {idx:03d}/{len(wav_paths)}] {dst_wav_name} -> {tamil_text}", flush=True)
        
    if transcripts:
        shutil.copyfile(os.path.join(target_dir, "reference_001.wav"), os.path.join(target_dir, "reference.wav"))
        with open(os.path.join(target_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcripts[0][1])
            
    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"✅ Completed {speaker_key}: {len(wav_paths)} files transcribed.", flush=True)

elapsed = round(time.time() - start_time, 1)
print(f"\n=======================================================", flush=True)
print(f"  All Voice Batches Completed! Total clips transcribed: {total_processed} in {elapsed}s", flush=True)
print(f"=======================================================", flush=True)

# Now rebuild full dataset manifest & zip
import subprocess
print("\nRebuilding complete training dataset package...", flush=True)
subprocess.run([sys.executable, "build_training_dataset.py"])
