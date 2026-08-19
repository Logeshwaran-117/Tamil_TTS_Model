import os
import json
import zipfile
import random
import soundfile as sf
import shutil

male_3_prompts = {
    'male_3_own_0001.wav': 'வணக்கம், நீங்கள் அனைவரும் எப்படி இருக்கிறீர்கள்?',
    'male_3_own_0002.wav': 'இன்று வானிலை மிகவும் இனிமையாகவும் அமைதியாகவும் உள்ளது.',
    'male_3_own_0003.wav': 'தமிழ் மொழி உலகிலேயே மிக தொன்மையான செம்மொழிகளில் ஒன்றாகும்.',
    'male_3_own_0004.wav': 'நாளை காலை எட்டு மணிக்கு முக்கியமான கலந்துரையாடல் தொடங்குகிறது.'
}

# 1. Update dataset/male_3_own/reference.wav and transcript.txt
os.makedirs('dataset/male_3_own', exist_ok=True)
ref_src = os.path.join('training_dataset', 'wavs', 'male_3_own_0001.wav')
ref_dst = os.path.join('dataset', 'male_3_own', 'reference.wav')
if os.path.exists(ref_src):
    shutil.copyfile(ref_src, ref_dst)
with open('dataset/male_3_own/transcript.txt', 'w', encoding='utf-8') as f:
    f.write(male_3_prompts['male_3_own_0001.wav'] + '\n')

# 2. Rescan training_dataset/wavs and rebuild metadata.csv
meta_rows = []
wavs_dir = os.path.join('training_dataset', 'wavs')

prompt_map = {
    '0001': {
        'male_1': 'நாளுக்கு நாள் உலகம் மாறிக்கிட்டு தான் இருக்குது என்னத்தை சொல்கிறது',
        'male_2': 'வீட்டுக்கு கொண்டு போயிடுவாங்க அப்பறோம்',
        'female_1': 'நான் வேலை செய்கிறப்போ எனக்கு ஒரு கஷ்டம் அதை வந்து சமாளித்து நானே அந்த வேலையை',
        'female_2': 'இந்த மாதிரி மியூசிக் போடுறாங்களா ஆ சூப்பராக இருக்கும் இந்த மாதிரி போட்டாங்கனால் இந்த மாதிரி நல்லாயிருக்கும் நாமளே போய் போட சொல்லி நாமளே டான்ஸ் பண்ணலாம்',
        'female_3_own': 'நான் வேலை செய்கிறப்போ எனக்கு ஒரு கஷ்டம் அதை வந்து சமாளித்து நானே அந்த வேலையை'
    },
    '0002': 'இன்று வானிலை மிகவும் இனிமையாகவும் அமைதியாகவும் உள்ளது.',
    '0003': 'தமிழ் மொழி உலகிலேயே மிக தொன்மையான செம்மொழிகளில் ஒன்றாகும்.',
    '0004': 'நாளை காலை எட்டு மணிக்கு முக்கியமான கலந்துரையாடல் தொடங்குகிறது.'
}

for fn in sorted(os.listdir(wavs_dir)):
    if not fn.endswith('.wav'):
        continue
    fp = os.path.join(wavs_dir, fn)
    data, sr = sf.read(fp)
    dur = round(len(data) / sr, 2)
    
    parts = fn.replace('.wav', '').split('_')
    spk = '_'.join(parts[:-1])
    idx = parts[-1]
    
    gender = 'female' if 'female' in spk else 'male'
    
    if spk == 'male_3_own':
        transcript = male_3_prompts.get(fn, '')
    elif idx == '0001':
        transcript = prompt_map['0001'].get(spk, '')
    else:
        transcript = prompt_map.get(idx, '')
        
    meta_rows.append({
        'audio_file': fn,
        'transcript': transcript,
        'speaker_id': spk,
        'gender': gender,
        'duration': dur
    })

# Write metadata.csv
with open('training_dataset/metadata.csv', 'w', encoding='utf-8') as f:
    f.write('audio_file|transcript|speaker_id|gender|duration\n')
    for r in meta_rows:
        f.write(f"{r['audio_file']}|{r['transcript']}|{r['speaker_id']}|{r['gender']}|{r['duration']}\n")

# Write splits
random.seed(42)
shuffled = list(meta_rows)
random.shuffle(shuffled)
split_idx = max(int(len(shuffled) * 0.85), 1)
train_set = shuffled[:split_idx]
val_set = shuffled[split_idx:]

with open('training_dataset/train.txt', 'w', encoding='utf-8') as f:
    for m in train_set:
        f.write(f"wavs/{m['audio_file']}|{m['transcript']}|{m['speaker_id']}\n")

with open('training_dataset/val.txt', 'w', encoding='utf-8') as f:
    for m in val_set:
        f.write(f"wavs/{m['audio_file']}|{m['transcript']}|{m['speaker_id']}\n")

# Summary JSON
summary = {
    'total_samples': len(meta_rows),
    'total_duration_seconds': round(sum(m['duration'] for m in meta_rows), 2),
    'speakers': {}
}
for m in meta_rows:
    spk = m['speaker_id']
    if spk not in summary['speakers']:
        summary['speakers'][spk] = {'count': 0, 'total_duration': 0.0, 'gender': m['gender']}
    summary['speakers'][spk]['count'] += 1
    summary['speakers'][spk]['total_duration'] = round(summary['speakers'][spk]['total_duration'] + m['duration'], 2)

with open('training_dataset/dataset_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

# Update zip archive
zip_path = 'training_dataset.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk('training_dataset'):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, '.')
            zf.write(full_path, arcname)

print('=== DATASET SYNC SUCCESSFUL ===')
print(f"Total Samples: {summary['total_samples']}")
print(f"Total Duration: {summary['total_duration_seconds']}s (~{summary['total_duration_seconds']/60:.1f} mins)")
for spk, info in summary['speakers'].items():
    print(f" - {spk}: {info['count']} clips ({info['total_duration']}s)")
