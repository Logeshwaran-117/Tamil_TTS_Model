import requests
import base64
import os
import sys

# 1. API Endpoint
URL = "http://localhost:5050/generate"

# 2. Payload - identical to what you sent in Postman
payload = {
    "text": "கல்வியும் ஒழுக்கமும் மனித வாழ்வின் மிகச் சிறந்த இரு கண்கள்",
    "voice": "female_1",
    "emotion": "neutral",
    "speed": 1.0,
    "auto_translate": True
}

print("📡 Calling Tamil TTS API at http://localhost:5050/generate ...")
response = requests.post(URL, json=payload)

if response.status_code == 200:
    data = response.json()
    b64_audio = data.get("audio_b64")
    
    # 3. Decode base64 to binary WAV file
    audio_bytes = base64.b64decode(b64_audio)
    output_filename = "output.wav"
    
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)
        
    print(f"✅ Success! Generated speech saved to: {os.path.abspath(output_filename)}")
    print("🔊 Opening audio file to play in Windows...")
    
    # Open and play in default Windows player
    os.system(f'start {output_filename}')
else:
    print(f"❌ Error {response.status_code}: {response.text}")
