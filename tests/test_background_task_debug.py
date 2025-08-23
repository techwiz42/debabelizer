#!/usr/bin/env python3

"""
Test script to verify if the background WebSocket task main loop is working.
Look for specific debug messages about the main event loop.
"""

import os
import asyncio

# Set real Soniox API key
os.environ["SONIOX_API_KEY"] = "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"

async def test_background_task():
    try:
        import debabelizer
        print("🔍 Starting background task debug test...")
        
        processor = debabelizer.VoiceProcessor(stt_provider="soniox")
        session_id = await processor.start_streaming_transcription()
        print(f"✅ Session created: {session_id}")
        
        # Send a small audio chunk to trigger processing
        audio_chunk = b'\x00' * 100  # 100 bytes of silence
        print("🔊 Sending audio chunk to background task...")
        await processor.stream_audio(session_id, audio_chunk)
        print("✅ Audio sent")
        
        # Give background task time to process
        print("⏱️  Waiting 2 seconds for background processing...")
        await asyncio.sleep(2)
        
        # Try to get results
        results_iterator = processor.get_streaming_results(session_id)
        print("🔍 Checking for results...")
        
        try:
            async def check_results():
                count = 0
                async for result in results_iterator:
                    count += 1
                    print(f"📈 Got result {count}: {result.text}")
                    if count >= 2:  # Just get a couple results
                        break
                return count
            
            result_count = await asyncio.wait_for(check_results(), timeout=3.0)
            print(f"✅ Got {result_count} results")
        except asyncio.TimeoutError:
            print("⚠️  No results received within timeout")
        
        print("🛑 Stopping session...")
        await processor.stop_streaming_transcription(session_id)
        
        print("\n" + "="*60)
        print("LOOK FOR THESE DEBUG MESSAGES IN THE OUTPUT ABOVE:")
        print("🔄 Entering main WebSocket event loop")
        print("🔄 Loop iteration - waiting for WebSocket messages or commands")
        print("Sending X bytes of audio to Soniox")
        print("")
        print("If you DON'T see these, the background task main loop isn't running!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_background_task())