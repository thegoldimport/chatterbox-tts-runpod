import runpod
import torch
import torchaudio
import base64
import io
from chatterbox import ChatterBox

print("Loading Chatterbox TTS model...")
model = ChatterBox()

# In-memory voice storage
voice_clones = {}

def handler(event):
    try:
        input_data = event["input"]
        operation = input_data.get("operation")

        if operation == "clone_voice":
            return clone_voice(input_data)
        elif operation == "generate_audio":
            return generate_audio(input_data)
        else:
            return {"error": f"Unknown operation: {operation}"}
    except Exception as e:
        return {"error": str(e)}

def clone_voice(input_data):
    voice_name = input_data["voice_name"]
    audio_base64 = input_data["reference_audio_base64"]

    clone_id = f"voice_{voice_name}_{len(voice_clones)}"
    voice_clones[clone_id] = {
        "name": voice_name,
        "audio_base64": audio_base64
    }

    print(f"Voice cloned: {clone_id}")
    return {
        "clone_id": clone_id,
        "voice_name": voice_name,
        "status": "success"
    }

def generate_audio(input_data):
    text = input_data["text"]
    voice_clone_id = input_data["voice_clone_id"]

    if voice_clone_id not in voice_clones:
        return {"error": f"Voice clone not found: {voice_clone_id}"}

    print(f"Generating audio for: {text[:50]}...")
    
    # Get the reference audio
    reference_audio_base64 = voice_clones[voice_clone_id]["audio_base64"]
    reference_audio_bytes = base64.b64decode(reference_audio_base64)
    
    # Save reference audio to temp file
    ref_path = f"/tmp/ref_{voice_clone_id}.wav"
    with open(ref_path, "wb") as f:
        f.write(reference_audio_bytes)
    
    # Generate NEW audio with Chatterbox using the cloned voice
    output_path = f"/tmp/output_{voice_clone_id}.wav"
    
    # Use Chatterbox to generate speech
    model.generate(
        text=text,
        reference_audio=ref_path,
        output_file=output_path
    )
    
    # Read the GENERATED audio (not the reference!)
    with open(output_path, "rb") as f:
        generated_audio_bytes = f.read()
    
    generated_audio_base64 = base64.b64encode(generated_audio_bytes).decode('utf-8')
    
    # Get audio duration
    waveform, sample_rate = torchaudio.load(output_path)
    duration = waveform.shape[1] / sample_rate
    
    return {
        "audio_base64": generated_audio_base64,  # Return GENERATED audio
        "duration": round(duration, 2),
        "text": text,  # Return text for validation
        "status": "success"
    }

runpod.serverless.start({"handler": handler})
