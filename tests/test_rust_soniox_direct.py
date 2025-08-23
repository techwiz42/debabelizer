#!/usr/bin/env python3
"""
Direct test of the Rust Soniox implementation to identify the exact failure point.
"""

import asyncio
import os
import sys
import time

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def main():
    """Test the core issue: why does the iterator end immediately?"""
    
    print("🚀 Direct Rust Soniox Test")
    print("Identifying why the streaming iterator ends immediately")
    print("=" * 60)
    
    try:
        import debabelizer
        print(f"✅ Debabelizer loaded: version {getattr(debabelizer, '__version__', '0.1.25')}")
        
        # Create processor
        processor = debabelizer.VoiceProcessor()
        print("✅ VoiceProcessor created")
        
        # Start streaming 
        print("\n🔄 Starting streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        
        # Get iterator
        print("\n📡 Getting streaming results...")
        results_iterator = processor.get_streaming_results(session_id)
        print(f"✅ Iterator: {type(results_iterator)}")
        
        # Test the iterator with detailed timing
        print("\n🔍 Testing iterator behavior...")
        start_time = time.time()
        result_count = 0
        
        print("   Starting async iteration...")
        
        # Use anext() directly to see what happens
        try:
            while True:
                try:
                    print(f"   Calling __anext__() iteration {result_count + 1}...")
                    start_next = time.time()
                    
                    result = await results_iterator.__anext__()
                    
                    elapsed_next = time.time() - start_next
                    result_count += 1
                    
                    print(f"   ✅ Got result {result_count} in {elapsed_next:.3f}s:")
                    print(f"      Text: '{result.text}'")
                    print(f"      Final: {result.is_final}")
                    print(f"      Confidence: {result.confidence}")
                    print(f"      Metadata: {getattr(result, 'metadata', None)}")
                    
                    # Stop after a few results for testing
                    if result_count >= 10:
                        print("   🛑 Stopping after 10 results")
                        break
                        
                except StopAsyncIteration:
                    elapsed = time.time() - start_time
                    print(f"   🏁 Iterator ended with StopAsyncIteration after {elapsed:.3f}s")
                    print(f"   Total results received: {result_count}")
                    break
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"   ❌ Exception in __anext__() after {elapsed:.3f}s: {e}")
                    print(f"   Exception type: {type(e)}")
                    break
                    
        except Exception as iter_error:
            elapsed = time.time() - start_time
            print(f"❌ Iterator setup failed after {elapsed:.3f}s: {iter_error}")
            
        total_elapsed = time.time() - start_time
        
        print(f"\n📊 Test Summary:")
        print(f"   Total time: {total_elapsed:.3f}s") 
        print(f"   Results received: {result_count}")
        print(f"   Session ID: {session_id}")
        
        # Try to stop the session
        try:
            print(f"\n🛑 Stopping session {session_id}...")
            await processor.stop_streaming_transcription(session_id)
            print("✅ Session stopped")
        except Exception as stop_error:
            print(f"❌ Failed to stop session: {stop_error}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())