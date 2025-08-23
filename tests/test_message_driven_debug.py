#!/usr/bin/env python3
"""
Test script with detailed debug information for the message-driven architecture
"""

import asyncio
import sys
import time
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_message_driven_approach():
    """Test the message-driven architecture with comprehensive debugging"""
    
    print("🧪 Message-Driven Architecture Test")
    print("==" * 30)
    
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
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        
        # Immediately get iterator to test background task communication
        print("\n🔍 Getting streaming results iterator...")
        iterator = voice_service.get_streaming_results(session_id)
        print(f"✅ Iterator created: {type(iterator)}")
        
        # Send audio chunk to trigger WebSocket activity
        print("\n🎵 Sending audio chunk...")
        audio_chunk = bytearray(1600)  # 100ms of 16kHz PCM
        for i in range(0, len(audio_chunk), 2):
            import struct
            # Generate a clear sine wave pattern
            sample = int(8000 * ((i // 2) % 160 - 80) / 80)
            audio_chunk[i:i+2] = struct.pack('<h', sample)
        
        await voice_service.stream_audio(session_id, bytes(audio_chunk))
        print("✅ Audio chunk sent")
        
        # Give background task time to process
        print("\n⏳ Waiting for background WebSocket processing...")
        await asyncio.sleep(1.0)
        
        # Test the iterator with aggressive debugging
        print("\n🔍 Testing async iterator (10 second limit)...")
        result_count = 0
        timeout_count = 0
        
        try:
            async def debug_iterator():
                nonlocal result_count, timeout_count
                
                # Test individual iterator calls with detailed timing
                for attempt in range(10):  # Try up to 10 times
                    print(f"   Attempt #{attempt + 1}: Calling __anext__()...")
                    start_time = time.time()
                    
                    try:
                        # This should call receive_transcript internally
                        result = await iterator.__anext__()
                        elapsed = time.time() - start_time
                        result_count += 1
                        
                        print(f"   ✅ Result #{result_count} (took {elapsed:.3f}s):")
                        print(f"      Text: '{result.text}'")
                        print(f"      Final: {result.is_final}")
                        print(f"      Confidence: {result.confidence:.2f}")
                        
                        # Check metadata for keep-alive messages
                        if hasattr(result, 'metadata') and result.metadata:
                            print(f"      Metadata: {result.metadata}")
                            if isinstance(result.metadata, dict):
                                msg_type = result.metadata.get('type', 'unknown')
                                reason = result.metadata.get('reason', 'unknown')
                                if msg_type == 'keep-alive':
                                    print(f"      🔄 Keep-alive: {reason}")
                        
                        # If we get real content, great!
                        if result.text.strip():
                            print(f"      🎯 Got real transcription: '{result.text.strip()}'")
                            
                        # Give some time between attempts
                        await asyncio.sleep(0.2)
                        
                    except StopAsyncIteration:
                        elapsed = time.time() - start_time
                        print(f"   🛑 Iterator ended after {elapsed:.3f}s")
                        break
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"   ❌ Error after {elapsed:.3f}s: {e}")
                        print(f"      Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        break
                        
                print(f"\n📊 Debug iterator completed with {result_count} results")
            
            # Run with overall timeout
            await asyncio.wait_for(debug_iterator(), timeout=10.0)
            
        except asyncio.TimeoutError:
            print(f"⏰ Iterator test timed out after 10 seconds with {result_count} results")
        except Exception as e:
            print(f"❌ Iterator test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Send more audio to see if background task is still working
        print(f"\n🎵 Sending more audio chunks to test background task...")
        for i in range(3):
            try:
                await voice_service.stream_audio(session_id, bytes(audio_chunk))
                print(f"   ✅ Additional audio chunk {i+1} sent")
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"   ❌ Audio chunk {i+1} failed: {e}")
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
    
    print("\n" + "==" * 30)
    print("Message-driven architecture test complete")

if __name__ == "__main__":
    asyncio.run(test_message_driven_approach())