#!/usr/bin/env python3
"""Test Rust implementation with real voice data."""

import asyncio
import os
import sys
import numpy as np
import pyaudio
from debabelizer import VoiceProcessor, AudioData, AudioFormat

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
RECORD_SECONDS = 5

def record_audio(duration=RECORD_SECONDS):
    """Record audio from microphone."""
    p = pyaudio.PyAudio()
    
    print(f"🎤 Recording {duration} seconds of audio...")
    
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("✅ Recording complete")
    return b''.join(frames)

def generate_test_audio():
    """Generate a simple test audio saying 'Hello, testing Rust implementation'."""
    # Generate a simple sine wave pattern that simulates speech
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    # Mix of frequencies to simulate speech-like audio
    frequencies = [200, 300, 400, 500]  # Human speech frequencies
    audio = np.zeros_like(t)
    
    for freq in frequencies:
        audio += 0.25 * np.sin(2 * np.pi * freq * t)
    
    # Add envelope to make it more speech-like
    envelope = np.exp(-t * 2) * (1 - np.exp(-t * 10))
    audio *= envelope
    
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    
    return audio.tobytes()

async def test_streaming_with_voice():
    """Test streaming transcription with voice data."""
    print("🚀 Testing Rust Debabelizer with Voice Data\n")
    
    # Check if we should use microphone or generated audio
    use_mic = input("Record from microphone? (y/n): ").lower() == 'y'
    
    if use_mic:
        try:
            audio_data = record_audio()
        except Exception as e:
            print(f"⚠️  Microphone error: {e}")
            print("Falling back to generated audio...")
            audio_data = generate_test_audio()
    else:
        print("Using generated test audio...")
        audio_data = generate_test_audio()
    
    # Create processor
    config = {
        "preferences": {
            "stt_provider": "soniox",
            "auto_select": False
        },
        "soniox": {
            "api_key": os.getenv("SONIOX_API_KEY")
        }
    }
    
    processor = VoiceProcessor(config=config)
    print(f"✅ Created processor with provider: {processor.current_stt_provider}")
    
    # Test batch transcription first
    print("\n📝 Testing Batch Transcription:")
    try:
        audio_format = AudioFormat("pcm", SAMPLE_RATE, CHANNELS, 16)
        audio = AudioData(audio_data, audio_format)
        
        result = processor.transcribe(audio)
        print(f"✅ Transcription: '{result.text}'")
        print(f"   Confidence: {result.confidence:.2f}")
        if result.language_detected:
            print(f"   Language: {result.language_detected}")
    except Exception as e:
        print(f"❌ Batch transcription error: {e}")
    
    # Test streaming transcription
    print("\n🎙️  Testing Streaming Transcription:")
    try:
        # Start streaming session
        session_id = await processor.start_streaming_transcription()
        print(f"✅ Started streaming session: {session_id}")
        
        # Send audio in chunks
        chunk_size = 4096  # Larger chunks for better recognition
        total_chunks = len(audio_data) // chunk_size
        
        print(f"📤 Sending {total_chunks} audio chunks...")
        
        # Start result collection task
        async def collect_results():
            results = []
            try:
                async for result in processor.get_streaming_results(session_id, timeout=2.0):
                    print(f"   📝 Result: text='{result.text}', is_final={result.is_final}, confidence={result.confidence:.2f}")
                    results.append(result)
            except Exception as e:
                print(f"   ⚠️  Result collection error: {e}")
            return results
        
        # Start collecting results
        results_task = asyncio.create_task(collect_results())
        
        # Send audio chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if chunk:
                await processor.stream_audio(session_id, chunk)
                await asyncio.sleep(0.1)  # Simulate real-time streaming
        
        print("✅ All audio chunks sent")
        
        # Stop streaming
        await processor.stop_streaming_transcription(session_id)
        print("✅ Streaming stopped")
        
        # Get final results
        all_results = await results_task
        
        # Combine final results
        final_text = " ".join(r.text for r in all_results if r.is_final and r.text)
        print(f"\n📝 Final transcription: '{final_text}'")
        print(f"   Total results received: {len(all_results)}")
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()

async def test_tts_synthesis():
    """Test TTS synthesis."""
    print("\n🔊 Testing Text-to-Speech:")
    
    config = {
        "preferences": {
            "tts_provider": "elevenlabs",
            "auto_select": False
        },
        "elevenlabs": {
            "api_key": os.getenv("ELEVENLABS_API_KEY")
        }
    }
    
    processor = VoiceProcessor(config=config)
    
    try:
        text = "Hello! I'm testing the Rust implementation of Debabelizer. This is a test of text to speech synthesis."
        
        result = processor.synthesize(text)
        print(f"✅ Synthesized {len(result.audio_data)} bytes of audio")
        print(f"   Format: {result.format}")
        print(f"   Duration: {result.duration:.2f}s")
        
        # Save to file
        output_file = "/home/peter/debabelizer/test_output.mp3"
        with open(output_file, "wb") as f:
            f.write(result.audio_data)
        print(f"✅ Saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ TTS error: {e}")

async def main():
    """Run all tests."""
    # Test STT
    await test_streaming_with_voice()
    
    # Test TTS if API key is available
    if os.getenv("ELEVENLABS_API_KEY"):
        await test_tts_synthesis()
    else:
        print("\n⚠️  Skipping TTS test (no ELEVENLABS_API_KEY)")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import pyaudio
        import numpy
    except ImportError:
        print("⚠️  Please install required dependencies:")
        print("   pip install pyaudio numpy")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("SONIOX_API_KEY"):
        print("⚠️  Please set SONIOX_API_KEY environment variable")
        sys.exit(1)
    
    asyncio.run(main())