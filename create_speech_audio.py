#!/usr/bin/env python3
"""Create a speech audio file using text-to-speech."""

import numpy as np
import wave

# Create a simple audio file with a beep pattern that might be recognized
def create_test_audio():
    sample_rate = 16000
    duration = 3  # seconds
    
    # Create silence
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.int16)
    
    # Add some beeps to simulate speech patterns
    # Create 3 beeps at different frequencies
    for i, freq in enumerate([440, 880, 440]):  # A4, A5, A4
        start = int(i * sample_rate)
        end = int((i + 0.5) * sample_rate)
        t = np.arange(start, end) / sample_rate
        beep = np.sin(2 * np.pi * freq * (t - i))
        # Add envelope
        envelope = np.hanning(len(beep))
        beep = beep * envelope * 10000  # Scale to reasonable amplitude
        audio[start:end] = beep.astype(np.int16)
    
    # Save as WAV
    with wave.open('test_beeps.wav', 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    
    print(f"Created test_beeps.wav: {duration}s, {sample_rate}Hz")
    return 'test_beeps.wav'

if __name__ == "__main__":
    create_test_audio()