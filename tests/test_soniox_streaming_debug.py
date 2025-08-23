#!/usr/bin/env python3
"""
Comprehensive test script to debug the Soniox streaming issue.
This script will reproduce the exact behavior seen in the WebSocket handler.
"""

import asyncio
import os
import sys
import traceback
import time
from typing import AsyncGenerator

# Add the Python package path
sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_soniox_streaming():
    """Test Soniox streaming functionality exactly as used in the WebSocket handler."""
    
    print("🔍 Starting comprehensive Soniox streaming test...")
    print("=" * 60)
    
    # Check environment
    soniox_key = os.getenv('SONIOX_API_KEY')
    if not soniox_key:
        print("❌ SONIOX_API_KEY not found in environment")
        return False
        
    print(f"✅ SONIOX_API_KEY present: {soniox_key[:8]}...")
    
    try:
        # Import debabelizer
        print("\n1️⃣ Importing debabelizer...")
        import debabelizer
        print(f"✅ Debabelizer version: {debabelizer.__version__ if hasattr(debabelizer, '__version__') else 'unknown'}")
        
        # Create processor with Soniox configuration
        print("\n2️⃣ Creating VoiceProcessor with Soniox...")
        
        # Try different approaches to create the processor
        processor = None
        
        # Approach 1: Environment-based (should pick up SONIOX_API_KEY)
        try:
            print("   Trying environment-based configuration...")
            processor = debabelizer.VoiceProcessor()
            print("✅ Environment-based VoiceProcessor created")
        except Exception as env_error:
            print(f"   ❌ Environment approach failed: {env_error}")
            
            # Approach 2: Try with explicit string provider
            try:
                print("   Trying explicit provider string...")
                processor = debabelizer.VoiceProcessor(stt_provider="soniox")
                print("✅ Explicit provider VoiceProcessor created")
            except Exception as provider_error:
                print(f"   ❌ Explicit provider approach failed: {provider_error}")
                return False
        print("✅ VoiceProcessor created successfully")
        
        # Start streaming session
        print("\n3️⃣ Starting streaming transcription session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            has_pending_audio=True,
            interim_results=True
        )
        print(f"✅ Session started: {session_id}")
        print(f"   Session ID type: {type(session_id)}")
        
        # Small delay to ensure session is initialized
        await asyncio.sleep(0.2)
        print("✅ Session initialization delay completed")
        
        # Test the streaming results iterator
        print("\n4️⃣ Testing streaming results iterator...")
        print("   Creating async iterator...")
        
        result_count = 0
        iterator_active = True
        start_time = time.time()
        
        try:
            # Get the async iterator
            results_iter = processor.get_streaming_results(session_id)
            print(f"✅ Iterator created: {type(results_iter)}")
            
            # Test the iterator with a timeout
            async def iterator_test():
                nonlocal result_count, iterator_active
                
                print("   Starting iterator loop...")
                async for result in results_iter:
                    result_count += 1
                    elapsed = time.time() - start_time
                    
                    print(f"   📥 Result {result_count}: text='{result.text}', is_final={result.is_final}, confidence={result.confidence:.2f} (t={elapsed:.1f}s)")
                    
                    # Break after a few results or if we get actual text
                    if result_count >= 10 or (result.text and result.text.strip()):
                        break
                        
                print(f"   🔄 Iterator loop ended after {result_count} results")
                iterator_active = False
            
            # Run iterator test with timeout
            print("   Running iterator with 5-second timeout...")
            try:
                await asyncio.wait_for(iterator_test(), timeout=5.0)
                print("✅ Iterator test completed normally")
            except asyncio.TimeoutError:
                print("⏱️ Iterator test timed out (this may be expected for keep-alive results)")
                
        except Exception as iter_error:
            print(f"❌ Iterator error: {iter_error}")
            print(f"   Error type: {type(iter_error)}")
            traceback.print_exc()
            
        # Test sending some dummy audio data
        print("\n5️⃣ Testing audio streaming...")
        try:
            # Create dummy PCM audio data (silence)
            dummy_audio = b'\x00' * 3200  # 0.1 seconds of silence at 16kHz 16-bit
            print(f"   Sending {len(dummy_audio)} bytes of dummy audio...")
            
            await processor.stream_audio(session_id, dummy_audio)
            print("✅ Audio sent successfully")
            
            # Wait briefly and check for results
            print("   Waiting 2 seconds for potential results...")
            await asyncio.sleep(2.0)
            
        except Exception as audio_error:
            print(f"❌ Audio streaming error: {audio_error}")
            print(f"   Error type: {type(audio_error)}")
            traceback.print_exc()
        
        # Final status check
        elapsed_total = time.time() - start_time
        print(f"\n📊 Test Results Summary:")
        print(f"   - Total runtime: {elapsed_total:.1f} seconds")
        print(f"   - Results received: {result_count}")
        print(f"   - Iterator still active: {iterator_active}")
        print(f"   - Session ID: {session_id}")
        
        # Stop the streaming session
        print("\n6️⃣ Stopping streaming session...")
        try:
            await processor.stop_streaming_transcription(session_id)
            print("✅ Session stopped successfully")
        except Exception as stop_error:
            print(f"❌ Error stopping session: {stop_error}")
            traceback.print_exc()
            
        return result_count > 0
        
    except Exception as e:
        print(f"❌ Critical error in test: {e}")
        print(f"   Error type: {type(e)}")
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    print("🚀 Soniox Streaming Debug Test")
    print("Testing v0.1.32 behavior to identify root cause")
    print("=" * 60)
    
    success = await test_soniox_streaming()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed - iterator received results")
    else:
        print("❌ Test failed - iterator ended immediately or crashed")
        
    print("🏁 Debug test finished")

if __name__ == "__main__":
    asyncio.run(main())