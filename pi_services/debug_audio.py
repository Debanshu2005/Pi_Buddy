#!/usr/bin/env python3
"""
Debug INMP441 with exact buddy_pi configuration
"""
import sounddevice as sd
import numpy as np
from scipy import signal
import vosk
import json
import os

def test_buddy_pi_audio():
    """Test exact same config as buddy_pi"""
    
    # Exact same config as buddy_pi
    DEVICE = "hw:3,0"
    SAMPLE_RATE = 48000
    CHANNELS = 2
    DURATION = 3
    
    print("Testing buddy_pi INMP441 configuration...")
    print(f"Device: {DEVICE}")
    print(f"Sample rate: {SAMPLE_RATE}Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Duration: {DURATION}s")
    
    try:
        # Record exactly like buddy_pi
        print("Recording...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int32",  # Same as buddy_pi
            device=DEVICE
        )
        sd.wait()
        print("Recording finished")
        
        # Process exactly like buddy_pi
        mono = audio[:, 1].astype(np.float32)  # Right channel
        
        # Check raw level
        max_level = np.max(np.abs(mono))
        print(f"Raw audio level: {max_level:.0f}")
        
        if max_level < 100:
            print("❌ Audio too quiet (< 100)")
            return
        
        # Apply buddy_pi processing
        mono -= np.mean(mono)  # Remove DC
        mono *= 10.0  # Boost gain
        
        # Check after boost
        boosted_level = np.max(np.abs(mono))
        print(f"Boosted audio level: {boosted_level:.0f}")
        
        # Normalize
        peak = np.max(np.abs(mono))
        if peak > 0:
            mono /= peak
        
        # Resample to 16kHz for Vosk
        mono_16k = signal.resample(mono, int(len(mono) * 16000 / 48000))
        mono_int16 = (mono_16k * 32767).astype(np.int16)
        
        print(f"Resampled to 16kHz: {len(mono_16k)} samples")
        
        # Test with Vosk
        model_path = "models/vosk-model-small-en-us-0.15"
        if os.path.exists(model_path):
            print("Testing with Vosk...")
            model = vosk.Model(model_path)
            rec = vosk.KaldiRecognizer(model, 16000)
            
            rec.AcceptWaveform(mono_int16.tobytes())
            result = json.loads(rec.FinalResult())
            text = result.get('text', '').strip()
            
            if text:
                print(f"✅ Vosk recognized: '{text}'")
            else:
                print("❌ Vosk could not recognize speech")
        else:
            print("⚠️ Vosk model not found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Speak clearly for 3 seconds...")
    test_buddy_pi_audio()