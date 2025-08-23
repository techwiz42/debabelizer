#!/usr/bin/env python3
"""
Simplified test to isolate the exact issue
"""

import asyncio
import sys
import time

sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def simple_soniox_test():
    """Very simple test to isolate the issue"""
    
    print("🔧 Simple Soniox Test")
    print("=" * 40)
    
    import debabelizer
    processor = debabelizer.VoiceProcessor()
    
    try:
        # Start session
        print("Starting session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            interim_results=True
        )
        print(f"✅ Session: {session_id}")
        
        # Try to get the iterator
        print("Getting iterator...")
        iterator = processor.get_streaming_results(session_id)
        print(f"✅ Iterator: {type(iterator)}")
        
        # Send some audio first to trigger server response
        print("Sending audio chunk to trigger response...")
        audio_chunk = bytearray(1600)
        for i in range(0, len(audio_chunk), 2):
            import struct
            sample = int(100 * ((i // 2) % 256 - 128) / 128)
            audio_chunk[i:i+2] = struct.pack('<h', sample)
        await processor.stream_audio(session_id, bytes(audio_chunk))
        print("✅ Audio chunk sent")
        
        # Small delay to let server process
        await asyncio.sleep(0.1)
        
        # Check if iterator works with timeout
        print("Testing async iterator with 5 second timeout...")
        result_count = 0
        try:
            async def iterate_with_timeout():
                async for result in iterator:
                    nonlocal result_count
                    result_count += 1
                    print(f"📥 Result #{result_count}: '{result.text}' (final: {result.is_final}, conf: {result.confidence:.2f})")
                    
                    # Check for keep-alive
                    if hasattr(result, 'metadata') and result.metadata:
                        print(f"   Metadata: {result.metadata}")
                    
                    if result_count >= 3:  # Get a few results
                        print("🛑 Got 3 results, stopping")
                        break
                        
            # Run with timeout
            await asyncio.wait_for(iterate_with_timeout(), timeout=5.0)
            print(f"✅ Iterator completed with {result_count} results")
            
        except asyncio.TimeoutError:
            print(f"⏰ Iterator timed out after 5 seconds with {result_count} results")
            print("   This suggests the first receive_transcript() call is hanging")
        except StopAsyncIteration:
            print(f"✅ Iterator ended naturally with {result_count} results")
        except Exception as e:
            print(f"❌ Iterator failed: {e}")
            print(f"   Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test complete")

if __name__ == "__main__":
    asyncio.run(simple_soniox_test())