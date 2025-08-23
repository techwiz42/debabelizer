#!/usr/bin/env python3
"""Test Python Soniox implementation with same audio file."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the python-legacy path
sys.path.insert(0, '/home/peter/debabelizer/python-legacy')

from debabelizer.providers.stt.soniox import SonioxSTTProvider

async def test_python_soniox():
    """Test Python Soniox with the same audio file."""
    print("🐍 Testing Python Soniox implementation...")
    
    # Initialize provider
    api_key = os.getenv("SONIOX_API_KEY")
    if not api_key:
        print("❌ SONIOX_API_KEY not set")
        return
        
    provider = SonioxSTTProvider(api_key=api_key)
    print("✅ Provider initialized")
    
    # Load test audio file
    audio_path = Path("/home/peter/debabelizer/english_sample.wav")
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
        
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    print(f"✅ Loaded audio file: {len(audio_data)} bytes")
    
    # Start streaming session
    print("\n🎤 Starting streaming session...")
    session_id = await provider.start_streaming(
        audio_format="wav",
        sample_rate=16000,
        language="en"
    )
    print(f"✅ Session started: {session_id}")
    
    # Send audio in chunks
    chunk_size = 3200
    pcm_data = audio_data[44:]  # Skip WAV header
    chunks = [pcm_data[i:i+chunk_size] for i in range(0, len(pcm_data), chunk_size)]
    
    print(f"\n📊 Streaming {len(chunks)} chunks...")
    
    # Collect results
    results = []
    
    # Start result collection task
    async def collect_results():
        count = 0
        async for result in provider.get_streaming_results(session_id):
            count += 1
            if result.text.strip():
                print(f"\n🎯 Result {count}: '{result.text}' (final={result.is_final})")
                results.append(result.text)
            else:
                print(f"💓 Keep-alive {count}")
            
            # Stop after reasonable time
            if count > 50:
                break
    
    # Start result collection
    result_task = asyncio.create_task(collect_results())
    
    # Send chunks
    for i, chunk in enumerate(chunks):
        print(f"📤 Sending chunk {i+1}/{len(chunks)}: {len(chunk)} bytes")
        await provider.stream_audio(session_id, chunk)
        await asyncio.sleep(0.2)
    
    print("\n⏳ Waiting for results...")
    await asyncio.sleep(5)
    
    # Stop streaming
    print("\n🛑 Stopping stream...")
    await provider.stop_streaming(session_id)
    
    # Cancel result collection
    result_task.cancel()
    try:
        await result_task
    except asyncio.CancelledError:
        pass
    
    # Print summary
    print(f"\n📊 Summary:")
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
    
    asyncio.run(test_python_soniox())