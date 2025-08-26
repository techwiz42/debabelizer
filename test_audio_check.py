#!/usr/bin/env python3
"""Check the audio file format and content."""

import wave
import struct

with wave.open('english_sample.wav', 'rb') as wav:
    print(f"Channels: {wav.getnchannels()}")
    print(f"Sample width: {wav.getsampwidth()} bytes")
    print(f"Frame rate: {wav.getframerate()} Hz")
    print(f"Number of frames: {wav.getnframes()}")
    print(f"Duration: {wav.getnframes() / wav.getframerate():.2f} seconds")
    print(f"Compression: {wav.getcomptype()}")
    
    # Read first few samples
    wav.rewind()
    frames = wav.readframes(100)
    samples = struct.unpack('<' + 'h' * 100, frames)
    
    print(f"\nFirst 10 samples: {samples[:10]}")
    print(f"Sample range: [{min(samples)}, {max(samples)}]")
    
    # Check if it's mostly silence
    avg_amplitude = sum(abs(s) for s in samples) / len(samples)
    print(f"Average amplitude (first 100 samples): {avg_amplitude:.1f}")