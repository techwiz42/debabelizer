#!/usr/bin/env python3
"""Simple test to see if Soniox is receiving and processing audio."""

import asyncio
import os
import sys
from pathlib import Path

# Add the project to sys.path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import debabelizer

async def test_simple():
    """Simple test without async iteration."""
    try:
        # Initialize processor
        print("\n=== Initializing VoiceProcessor ===")
        processor = debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor created")
        
        # Load audio file
        audio_file = str(project_root / "test_speech.wav")
        print(f"\n=== Loading audio file: {audio_file} ===")
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        print(f"✅ Loaded {len(audio_data)} bytes of audio")
        
        # Start streaming transcription
        print("\n=== Starting streaming transcription ===")
        session_id = await processor.start_streaming_transcription(
            audio_format='wav',
            sample_rate=16000,
            language='en-US',
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        
        # Send all audio at once
        print("\n=== Sending all audio at once ===")
        await processor.stream_audio(session_id, audio_data)
        print(f"✅ Sent {len(audio_data)} bytes")
        
        # Send end signal
        print("\n=== Sending end-of-audio signal ===")
        await processor.stream_audio(session_id, b'')
        print("✅ End signal sent")
        
        # Wait a moment for processing
        print("\n=== Waiting for processing ===")
        await asyncio.sleep(3)
        print("✅ Wait complete")
        
        # Try to get one result manually
        print("\n=== Attempting to get results ===")
        try:
            # Create an async iterator
            results_iter = processor.get_streaming_results(session_id)
            
            # Try to get one result with timeout
            result = await asyncio.wait_for(results_iter.__anext__(), timeout=2.0)
            print(f"✅ Got result: text='{result.text}', is_final={result.is_final}")
        except asyncio.TimeoutError:
            print("❌ Timeout waiting for results")
        except StopAsyncIteration:
            print("❌ No results available (iterator ended)")
        
        # Stop the session
        print("\n=== Stopping session ===")
        await processor.stop_streaming_transcription(session_id)
        print("✅ Session stopped")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting simple Soniox test...")
    asyncio.run(test_simple())