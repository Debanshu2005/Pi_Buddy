#!/usr/bin/env python3
"""
Simple microphone test for INMP441 or any available audio device
"""
import sounddevice as sd
import numpy as np

def test_microphone():
    """Test available microphones"""
    print("Available audio devices:")
    try:
        devices = sd.query_devices()
        
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
                input_devices.append(i)
        
        if not input_devices:
            print("❌ No input devices found!")
            return
        
        # Test each input device
        for device_id in input_devices:
            device_info = devices[device_id]
            print(f"\n🎤 Testing device {device_id}: {device_info['name']}")
            
            # Test different configurations
            configs = [
                (44100, 2), (44100, 1),
                (48000, 2), (48000, 1),
                (22050, 2), (22050, 1),
                (16000, 2), (16000, 1)
            ]
            
            for rate, channels in configs:
                if channels > device_info['max_input_channels']:
                    continue
                    
                try:
                    print(f"  Testing {rate}Hz, {channels} channels...")
                    audio = sd.rec(
                        int(0.2 * rate),  # 0.2 seconds
                        samplerate=rate,
                        channels=channels,
                        dtype='int16',
                        device=device_id
                    )
                    sd.wait()
                    
                    max_amplitude = np.max(np.abs(audio))
                    print(f"    ✅ Works - max amplitude: {max_amplitude}")
                    
                    if max_amplitude > 50:
                        print(f"    🎤 Audio detected!")
                    
                    return device_id, rate, channels  # Return first working config
                    
                except Exception as e:
                    print(f"    ❌ Failed: {str(e)[:50]}...")
        
        print("\n❌ No working audio configuration found")
        return None, None, None
        
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return None, None, None

if __name__ == "__main__":
    device, rate, channels = test_microphone()
    if device is not None:
        print(f"\n✅ Recommended config: device={device}, rate={rate}, channels={channels}")
    else:
        print("\n❌ No audio input available - speech will be disabled")