#!/usr/bin/env python3
"""Test waiting longer for Soniox results."""

import asyncio
import os
import sys
from pathlib import Path

# Add the project to sys.path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import debabelizer

async def test_wait_longer():
    """Test waiting longer for results."""
    try:
        # Initialize processor
        print("\n=== Initializing VoiceProcessor ===")
        processor = debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor created")
        
        # Load audio file
        audio_file = str(project_root / "english_sample.wav")
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
        
        # Send audio in smaller chunks to simulate real-time
        print("\n=== Sending audio in chunks ===")
        chunk_size = 4000  # smaller chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await processor.stream_audio(session_id, chunk)
            print(f"✅ Sent chunk: {len(chunk)} bytes")
            await asyncio.sleep(0.1)  # simulate real-time
        
        # Send end signal
        print("\n=== Sending end-of-audio signal ===")
        await processor.stream_audio(session_id, b'')
        print("✅ End signal sent")
        
        # Collect results with longer timeout
        print("\n=== Collecting results (waiting up to 30 seconds) ===")
        results_collected = 0
        all_text = []
        
        start_time = asyncio.get_event_loop().time()
        timeout = 30.0  # 30 seconds timeout
        
        async for result in processor.get_streaming_results(session_id):
            results_collected += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            
            print(f"\n📝 Result {results_collected} (after {elapsed:.1f}s):")
            print(f"   Text: '{result.text}'")
            print(f"   Is Final: {result.is_final}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            if result.text:
                all_text.append(result.text)
            
            # Stop if we get a final result with text
            if result.is_final and result.text:
                print("\n✅ Got final result with text!")
                break
            
            # Stop after timeout
            if elapsed >= timeout:
                print(f"\n⏰ Timeout after {elapsed:.1f}s")
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
    print("🚀 Starting longer wait test...")
    asyncio.run(test_wait_longer())