#!/usr/bin/env python3
"""
Debug script to see exactly what messages Soniox WebSocket is sending/receiving
"""

import asyncio
import sys
import os
import json

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def debug_soniox_streaming():
    """Debug the exact WebSocket flow to see what messages are sent/received"""
    
    print("🔍 Debugging Soniox WebSocket Communication")
    print("=" * 60)
    
    import debabelizer
    voice_service = debabelizer.VoiceProcessor()
    
    print(f"✅ VoiceProcessor created: {type(voice_service)}")
    
    try:
        # Start streaming session
        print("\n📡 Starting Soniox streaming session...")
        session_id = await voice_service.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        
        # Send one audio chunk immediately to trigger server response
        print("\n🎵 Sending first audio chunk...")
        audio_chunk = bytearray(1600)  # 100ms of 16kHz PCM
        for i in range(0, len(audio_chunk), 2):
            import struct
            # Small sine wave to simulate audio
            sample = int(1000 * ((i // 2) % 256 - 128) / 128)
            audio_chunk[i:i+2] = struct.pack('<h', sample)
        
        await voice_service.stream_audio(session_id, bytes(audio_chunk))
        print("✅ First audio chunk sent")
        
        # Now try to read results with detailed debugging
        print("\n👂 Attempting to read streaming results...")
        result_count = 0
        max_results = 5
        
        try:
            async for result in voice_service.get_streaming_results(session_id):
                result_count += 1
                print(f"📥 Result #{result_count}:")
                print(f"   Text: '{result.text}'")
                print(f"   Is Final: {result.is_final}")
                print(f"   Confidence: {result.confidence:.2f}")
                
                # Check for metadata
                if hasattr(result, 'metadata') and result.metadata:
                    print(f"   Metadata: {result.metadata}")
                
                # Check if this is a keep-alive
                is_keep_alive = (
                    not result.text.strip() and 
                    hasattr(result, 'metadata') and 
                    result.metadata and 
                    isinstance(result.metadata, dict) and 
                    result.metadata.get("type") == "keep-alive"
                )
                
                if is_keep_alive:
                    reason = result.metadata.get("reason", "unknown")
                    print(f"   🔄 Keep-alive type: {reason}")
                
                if result_count >= max_results:
                    print(f"🛑 Stopping after {max_results} results")
                    break
            
            print(f"\n📊 Iterator ended after {result_count} results")
            
            if result_count == 0:
                print("❌ PROBLEM: Iterator ended immediately with 0 results")
                print("   This suggests the WebSocket connection closed or had an error")
            elif result_count < max_results:
                print(f"⚠️  Iterator ended early after {result_count} results")
            else:
                print("✅ Iterator worked correctly")
                
        except Exception as iter_error:
            print(f"❌ Iterator error: {iter_error}")
            print(f"   Error type: {type(iter_error)}")
            import traceback
            traceback.print_exc()
        
        # Wait a bit more for potential messages
        print("\n⏳ Waiting 2 seconds for any delayed messages...")
        await asyncio.sleep(2.0)
        
        # Send more audio chunks to see if connection is still alive
        print("\n🎵 Sending additional audio chunks to test connection...")
        for i in range(3):
            try:
                await voice_service.stream_audio(session_id, bytes(audio_chunk))
                print(f"   ✅ Audio chunk {i+2} sent successfully")
                await asyncio.sleep(0.1)
            except Exception as audio_error:
                print(f"   ❌ Audio chunk {i+2} failed: {audio_error}")
                break
        
    except Exception as e:
        print(f"❌ Session error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            print(f"\n🧹 Stopping session: {session_id}")
            await voice_service.stop_streaming_transcription(session_id)
            print("✅ Session stopped successfully")
        except Exception as cleanup_error:
            print(f"❌ Cleanup error: {cleanup_error}")
    
    print("\n" + "=" * 60)
    print("Debug session complete")

if __name__ == "__main__":
    asyncio.run(debug_soniox_streaming())