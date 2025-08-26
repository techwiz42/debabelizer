#!/usr/bin/env python3
"""Test script to verify Soniox transcription with extensive logging."""

import asyncio
import os
import sys
import wave
import numpy as np
from pathlib import Path

# Add the project to sys.path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import debabelizer

# Find the test audio file
test_audio_path = project_root / "test_pypi_package.py"
audio_file = None

# Search for audio file in various locations
for path in [
    project_root / "test_audio.wav",
    project_root / "english_sample.wav",
    project_root / "audio_sample.wav",
    project_root / "test.wav",
    project_root / "sample.wav"
]:
    if path.exists():
        audio_file = str(path)
        break

if not audio_file:
    # Create a simple test audio file with tone
    print("Creating test audio file...")
    audio_file = str(project_root / "test_audio.wav")
    
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more speech-like
    noise = np.random.normal(0, 0.1, audio_data.shape)
    audio_data = audio_data + noise
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV file
    with wave.open(audio_file, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_data.tobytes())

print(f"Using audio file: {audio_file}")

async def test_transcription():
    """Test Soniox transcription with logging."""
    try:
        # Initialize processor
        print("\n=== Initializing VoiceProcessor ===")
        processor = debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor created")
        
        # Load audio file
        print(f"\n=== Loading audio file: {audio_file} ===")
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        print(f"✅ Loaded {len(audio_data)} bytes of audio")
        
        # Get file info
        with wave.open(audio_file, 'rb') as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            duration = n_frames / sample_rate
            print(f"📊 Audio info: {channels} channel(s), {sample_rate}Hz, {sample_width*8}-bit, {duration:.2f}s duration")
        
        # Start streaming transcription
        print("\n=== Starting streaming transcription ===")
        session_id = await processor.start_streaming_transcription(
            audio_format='wav',
            sample_rate=sample_rate,
            language='en-US',
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        
        # Send audio in chunks
        chunk_size = 8000  # bytes
        chunks_sent = 0
        
        print("\n=== Sending audio chunks ===")
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            print(f"📤 Sending chunk {chunks_sent + 1}: {len(chunk)} bytes")
            await processor.stream_audio(session_id, chunk)
            chunks_sent += 1
            # Small delay between chunks
            await asyncio.sleep(0.1)
        
        print(f"✅ Sent {chunks_sent} chunks total")
        
        # Signal end of audio by sending empty chunk
        print("\n=== Signaling end of audio ===")
        await processor.stream_audio(session_id, b'')
        print("✅ End of audio signaled")
        
        # Get results
        print("\n=== Collecting results ===")
        results_collected = 0
        all_text = []
        
        async for result in processor.get_streaming_results(session_id):
            results_collected += 1
            print(f"\n📝 Result {results_collected}:")
            print(f"   Text: '{result.text}'")
            print(f"   Is Final: {result.is_final}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            if result.text:
                all_text.append(result.text)
            
            # Stop after getting some results or timeout
            if results_collected >= 10:
                print("\n⚠️  Collected 10 results, stopping...")
                break
        
        # Stop the session
        print("\n=== Stopping session ===")
        await processor.stop_streaming_transcription(session_id)
        print("✅ Session stopped")
        
        # Print final results
        print("\n=== FINAL RESULTS ===")
        print(f"Total results collected: {results_collected}")
        print(f"Transcribed text: '{' '.join(all_text)}'")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Soniox transcription test with logging...")
    print(f"Python: {sys.version}")
    print(f"Debabelizer module: {debabelizer}")
    
    # Run the async test
    asyncio.run(test_transcription())