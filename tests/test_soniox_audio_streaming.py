#!/usr/bin/env python3
"""
Test Soniox streaming with actual audio data to verify the fix works in real scenarios.
"""

import asyncio
import os
import sys
import time
import threading

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_soniox_streaming_with_audio():
    """Test Soniox streaming with actual audio streaming like in the WebSocket handler."""
    
    print("🎵 Soniox Streaming + Audio Test")
    print("Testing the complete streaming workflow with audio chunks")
    print("=" * 60)
    
    try:
        import debabelizer
        print(f"✅ Debabelizer loaded: version {getattr(debabelizer, '__version__', 'unknown')}")
        
        # Create processor
        processor = debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor created")
        
        # Start streaming session
        print("\n🚀 Starting streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        print(f"✅ Session: {session_id}")
        
        # Create a task to handle streaming results
        result_count = 0
        results_active = True
        
        async def handle_streaming_results():
            nonlocal result_count, results_active
            try:
                print("📥 Starting streaming results handler...")
                results_iter = processor.get_streaming_results(session_id)
                
                async for result in results_iter:
                    result_count += 1
                    print(f"   📥 Result {result_count}: '{result.text}' (final: {result.is_final}, conf: {result.confidence:.2f})")
                    
                    # Break after a reasonable number of results
                    if result_count >= 20:
                        print("   🛑 Stopping after 20 results for testing")
                        break
                        
                print(f"📤 Streaming results handler ended after {result_count} results")
                results_active = False
                
            except Exception as e:
                print(f"❌ Error in results handler: {e}")
                results_active = False
        
        # Start the results handler
        results_task = asyncio.create_task(handle_streaming_results())
        
        # Wait a moment for initialization
        await asyncio.sleep(0.5)
        
        # Stream some audio chunks
        print("\n🎵 Streaming audio chunks...")
        
        # Create dummy PCM audio data (varying patterns to simulate real audio)
        chunk_size = 1600  # 0.05 seconds at 16kHz 16-bit
        num_chunks = 10
        
        for i in range(num_chunks):
            # Create audio chunk with some variation (not pure silence)
            chunk = bytearray(chunk_size)
            
            # Add some random variation to simulate real audio
            for j in range(0, chunk_size, 2):
                # Small amplitude sine wave pattern
                import math
                amplitude = int(1000 * math.sin(2 * math.pi * i * j / chunk_size))
                chunk[j] = amplitude & 0xFF
                chunk[j + 1] = (amplitude >> 8) & 0xFF
            
            try:
                print(f"   📤 Sending chunk {i+1}/{num_chunks} ({len(chunk)} bytes)...")
                await processor.stream_audio(session_id, bytes(chunk))
                print(f"   ✅ Chunk {i+1} sent successfully")
                
                # Small delay between chunks
                await asyncio.sleep(0.1)
                
            except Exception as audio_error:
                print(f"   ❌ Error streaming chunk {i+1}: {audio_error}")
                break
        
        # Wait a bit more for final results
        print("\n⏳ Waiting for final results...")
        await asyncio.sleep(2.0)
        
        # Cancel the results task if it's still running
        if not results_task.done():
            print("🛑 Canceling results task...")
            results_task.cancel()
            try:
                await results_task
            except asyncio.CancelledError:
                pass
        
        # Final summary
        print(f"\n📊 Test Results:")
        print(f"   Audio chunks sent: {min(i+1, num_chunks)}")
        print(f"   Streaming results received: {result_count}")
        print(f"   Session ID: {session_id}")
        
        # Stop the session
        try:
            print("\n🛑 Stopping streaming session...")
            await processor.stop_streaming_transcription(session_id)
            print("✅ Session stopped")
        except Exception as stop_error:
            print(f"❌ Error stopping session: {stop_error}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test runner."""
    await test_soniox_streaming_with_audio()

if __name__ == "__main__":
    asyncio.run(main())