import runpod
import torchaudio as ta
import base64
import io
import os
import json

print("Loading Chatterbox TTS model...")
from chatterbox.tts import ChatterboxTTS

# Initialize model (first run downloads it)
model = ChatterboxTTS.from_pretrained(device="cuda")

# Persistent voice storage directory (NETWORK STORAGE - shared across all serverless workers)
# RunPod serverless auto-mounts network volumes at /runpod-volume
VOICE_STORAGE_DIR = "/runpod-volume/voice_clones"
os.makedirs(VOICE_STORAGE_DIR, exist_ok=True)

print(f"[Handler] Voice storage directory: {VOICE_STORAGE_DIR}")
print(f"[Handler] Directory exists: {os.path.exists(VOICE_STORAGE_DIR)}")

def save_voice_clone(clone_id, voice_data):
    """Save voice clone to persistent storage"""
    clone_path = os.path.join(VOICE_STORAGE_DIR, f"{clone_id}.json")
    with open(clone_path, "w") as f:
        json.dump(voice_data, f)
    print(f"[Handler] Saved voice clone: {clone_id} to {clone_path}")
    print(f"[Handler] File exists after save: {os.path.exists(clone_path)}")

def load_voice_clone(clone_id):
    """Load voice clone from persistent storage"""
    clone_path = os.path.join(VOICE_STORAGE_DIR, f"{clone_id}.json")
    print(f"[Handler] Looking for voice clone at: {clone_path}")
    print(f"[Handler] File exists: {os.path.exists(clone_path)}")
    
    if not os.path.exists(clone_path):
        # List all files for debugging
        try:
            all_files = os.listdir(VOICE_STORAGE_DIR)
            print(f"[Handler] Available voice clones: {all_files}")
        except Exception as e:
            print(f"[Handler] Error listing directory: {e}")
        return None
    
    with open(clone_path, "r") as f:
        data = json.load(f)
    print(f"[Handler] Successfully loaded voice clone: {clone_id}")
    return data

def handler(event):
    try:
        input_data = event["input"]
        operation = input_data.get("operation")

        print(f"[Handler] Operation: {operation}")

        if operation == "clone_voice":
            return clone_voice(input_data)
        elif operation == "generate_audio":
            return generate_audio(input_data)
        else:
            error_msg = f"Unknown operation: {operation}"
            print(f"[Error] {error_msg}")
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"[Error] {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

def clone_voice(input_data):
    try:
        voice_name = input_data["voice_name"]
        audio_base64 = input_data["reference_audio_base64"]

        clone_id = f"voice_{voice_name}_{int(os.urandom(4).hex(), 16)}"
        
        voice_data = {
            "name": voice_name,
            "audio_base64": audio_base64
        }
        
        # Save to persistent storage
        save_voice_clone(clone_id, voice_data)

        print(f"[Handler] Voice cloned successfully: {clone_id}")
        return {
            "clone_id": clone_id,
            "voice_name": voice_name,
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Clone voice error: {str(e)}"
        print(f"[Error] {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

def generate_audio(input_data):
    try:
        text = input_data["text"]
        voice_clone_id = input_data["voice_clone_id"]

        print(f"[Generate] Loading voice clone: {voice_clone_id}")
        
        # Load from persistent storage
        voice_data = load_voice_clone(voice_clone_id)
        if voice_data is None:
            error_msg = f"Voice clone not found: {voice_clone_id}"
            print(f"[Error] {error_msg}")
            return {"error": error_msg}

        print(f"[Generate] Generating audio for: {text[:50]}...")
        
        # Decode reference audio
        reference_audio_base64 = voice_data["audio_base64"]
        reference_audio_bytes = base64.b64decode(reference_audio_base64)
        
        # Save reference audio to temp file
        ref_path = f"/tmp/ref_{voice_clone_id}.wav"
        with open(ref_path, "wb") as f:
            f.write(reference_audio_bytes)
        
        print(f"[Generate] Running Chatterbox TTS...")
        # Generate speech with Chatterbox
        wav = model.generate(text, audio_prompt_path=ref_path)
        
        # Save generated audio
        output_path = f"/tmp/output_{voice_clone_id}.wav"
        ta.save(output_path, wav, model.sr)
        
        # Read generated audio and encode to base64
        with open(output_path, "rb") as f:
            generated_audio_bytes = f.read()
        
        generated_audio_base64 = base64.b64encode(generated_audio_bytes).decode('utf-8')
        
        # Calculate duration
        duration = wav.shape[-1] / model.sr
        
        print(f"[Generate] Success! Duration: {duration:.2f}s, Audio size: {len(generated_audio_base64)} chars")
        
        return {
            "audio_base64": generated_audio_base64,
            "duration": round(duration, 2),
            "text": text,
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Generate audio error: {str(e)}"
        print(f"[Error] {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

print("[Handler] Starting RunPod serverless...")
runpod.serverless.start({"handler": handler})
