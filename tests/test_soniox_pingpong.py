#!/usr/bin/env python3
"""Test Soniox streaming with ping/pong handling."""

import asyncio
import os
import time
from pathlib import Path

# Import the Rust-based debabelizer
import debabelizer

async def test_soniox_streaming():
    """Test Soniox streaming with actual audio."""
    print("Testing Soniox streaming with ping/pong handling...")
    
    # Initialize processor
    config_dict = {
        "preferences": {
            "stt_provider": "soniox",
            "tts_provider": "openai"
        },
        "soniox": {
            "api_key": os.getenv("SONIOX_API_KEY")
        }
    }
    
    config = debabelizer.DebabelizerConfig(config_dict)
    processor = debabelizer.VoiceProcessor(config=config)
    print("✅ Processor initialized with Soniox")
    
    # Load test audio - using the existing test file
    audio_path = Path("/home/peter/debabelizer/english_sample.wav")
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
        
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    print(f"✅ Loaded audio file: {len(audio_data)} bytes")
    
    # Create audio object
    audio_format = debabelizer.AudioFormat("wav", 16000, 1, 16)
    audio = debabelizer.AudioData(audio_data, audio_format)
    
    # Start streaming session
    print("\n🎤 Starting streaming session...")
    session_id = await processor.start_streaming_transcription()
    print(f"✅ Session started: {session_id}")
    
    # Simulate chunked audio streaming
    chunk_size = 3200  # 200ms at 16kHz
    audio_bytes = audio.data
    
    # Skip WAV header (44 bytes)
    audio_bytes = audio_bytes[44:]
    
    print(f"\n📊 Streaming {len(audio_bytes)} bytes in {chunk_size}-byte chunks...")
    
    # Keep track of results
    results = []
    result_count = 0
    
    # Start result collection task
    async def collect_results():
        nonlocal result_count
        try:
            async for result in processor.get_streaming_results(session_id):
                result_count += 1
                if result.text:
                    print(f"\n🎯 Result {result_count}: '{result.text}' (final={result.is_final})")
                    results.append(result.text)
                else:
                    print(f"💓 Keep-alive {result_count}")
                    
                # Test for at least 10 seconds to see if ping/pong keeps connection alive
                if time.time() - start_time > 10:
                    print("\n✅ Connection stayed alive for 10+ seconds!")
                    break
        except Exception as e:
            print(f"\n❌ Result collection error: {e}")
    
    # Start result collection
    result_task = asyncio.create_task(collect_results())
    
    # Send audio chunks
    start_time = time.time()
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if chunk:
            print(f"📤 Sending chunk {i//chunk_size + 1}: {len(chunk)} bytes")
            await processor.stream_audio(session_id, chunk)
            await asyncio.sleep(0.2)  # Simulate real-time streaming
    
    print("\n⏳ Waiting for final results...")
    
    # Wait a bit more for final results
    await asyncio.sleep(5)
    
    # Stop streaming
    print("\n🛑 Stopping stream...")
    await processor.stop_streaming_transcription(session_id)
    
    # Cancel result collection
    result_task.cancel()
    try:
        await result_task
    except asyncio.CancelledError:
        pass
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"- Connection duration: {time.time() - start_time:.1f} seconds")
    print(f"- Results received: {result_count}")
    print(f"- Transcriptions: {len(results)}")
    if results:
        print(f"- Full text: {' '.join(results)}")
    else:
        print("- No transcription text received")

if __name__ == "__main__":
    # Set Soniox API key if not already set
    if not os.getenv("SONIOX_API_KEY"):
        # Try to load from .env file
        env_path = Path("/home/peter/debabelizer/.env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("SONIOX_API_KEY="):
                        os.environ["SONIOX_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                        break
    
    asyncio.run(test_soniox_streaming())