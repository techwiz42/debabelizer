#!/usr/bin/env python3
"""
Minimal test to identify exactly when and why Soniox WebSocket closes.
"""

import asyncio
import os
import sys
import time
import threading

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_connection_timing():
    """Test the exact timing of WebSocket closure."""
    
    print("🔍 Connection Timing Analysis")
    print("Identifying exactly when Soniox WebSocket closes")
    print("=" * 60)
    
    try:
        import debabelizer
        print(f"✅ Debabelizer v{getattr(debabelizer, '__version__', 'unknown')}")
        
        # Create processor
        processor = debabelizer.VoiceProcessor()
        
        # Start streaming session
        print("\n1️⃣ Starting streaming session...")
        start_time = time.time()
        
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        handshake_time = time.time() - start_time
        print(f"✅ Session started in {handshake_time:.3f}s: {session_id}")
        
        # Get iterator
        print("\n2️⃣ Getting streaming results iterator...")
        results_iter = processor.get_streaming_results(session_id)
        
        # Test how many results we can get before connection closes
        print("\n3️⃣ Testing iterator behavior...")
        result_count = 0
        
        try:
            # Try to get up to 5 results to see when it stops
            for i in range(5):
                print(f"   📞 Calling __anext__() #{i+1}...")
                iter_start = time.time()
                
                try:
                    result = await results_iter.__anext__()
                    iter_duration = time.time() - iter_start
                    result_count += 1
                    
                    print(f"   ✅ Result {result_count} in {iter_duration:.3f}s:")
                    print(f"      Text: '{result.text}'")
                    print(f"      Final: {result.is_final}")
                    print(f"      Confidence: {result.confidence}")
                    
                    if hasattr(result, 'metadata') and result.metadata:
                        metadata_type = result.metadata.get('type', 'none')
                        metadata_reason = result.metadata.get('reason', 'none')
                        print(f"      Metadata: type={metadata_type}, reason={metadata_reason}")
                    else:
                        print("      Metadata: none")
                        
                except StopAsyncIteration:
                    iter_duration = time.time() - iter_start
                    print(f"   🛑 Iterator ended with StopAsyncIteration after {iter_duration:.3f}s")
                    break
                except Exception as e:
                    iter_duration = time.time() - iter_start
                    print(f"   ❌ Iterator error after {iter_duration:.3f}s: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Iterator setup failed: {e}")
        
        total_time = time.time() - start_time
        print(f"\n📊 Timing Analysis:")
        print(f"   Handshake duration: {handshake_time:.3f}s")
        print(f"   Total test duration: {total_time:.3f}s")
        print(f"   Results received: {result_count}")
        
        # Test if we can send audio immediately after getting first result
        if result_count > 0:
            print(f"\n4️⃣ Testing audio sending after {result_count} results...")
            dummy_audio = b'\x00' * 1600
            
            try:
                await processor.stream_audio(session_id, dummy_audio)
                print("✅ Audio sent successfully")
            except Exception as audio_error:
                print(f"❌ Audio send failed: {audio_error}")
        
        # Try to stop session
        try:
            print("\n5️⃣ Stopping session...")
            await processor.stop_streaming_transcription(session_id)
            print("✅ Session stopped")
        except Exception as stop_error:
            print(f"❌ Session stop failed: {stop_error}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_connection_timing()

if __name__ == "__main__":
    asyncio.run(main())