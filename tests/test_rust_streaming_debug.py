#!/usr/bin/env python3

"""
Test script specifically for streaming functionality to see if the 
background WebSocket handler is being called.
"""

import os
import sys
import asyncio

# Set up environment for Soniox
os.environ["SONIOX_API_KEY"] = "fake_key_for_testing"

async def test_streaming():
    try:
        print("🔍 Importing debabelizer...")
        import debabelizer
        print("✅ Successfully imported debabelizer")
        
        print("\n🔍 Creating VoiceProcessor with soniox...")
        processor = debabelizer.VoiceProcessor(stt_provider="soniox")
        print("✅ Successfully created VoiceProcessor")
        
        print("\n🔍 Starting streaming transcription session...")
        session_id = await processor.start_streaming_transcription()
        print(f"✅ Started streaming session: {session_id}")
        
        print("\n🔍 Getting streaming results iterator...")
        results_iterator = processor.get_streaming_results(session_id)
        print("✅ Got streaming results iterator")
        
        print("\n🔍 Sending audio chunk...")
        # Send a small audio chunk
        dummy_audio = b'\x00' * 100  # 100 bytes of silence
        await processor.stream_audio(session_id, dummy_audio)
        print("✅ Audio chunk sent")
        
        print("\n🔍 Attempting to get first result (with 2 second timeout)...")
        try:
            async def get_result_with_timeout():
                async for result in results_iterator:
                    return result
                return None
            
            result = await asyncio.wait_for(get_result_with_timeout(), timeout=2.0)
            if result:
                print(f"✅ Got streaming result: {result}")
            else:
                print("⚠️  Iterator ended without results")
                
        except asyncio.TimeoutError:
            print("⚠️  Timeout waiting for streaming result (this suggests background task isn't running)")
        except Exception as e:
            print(f"❌ Error getting streaming result: {e}")
        
        print("\n🔍 Stopping streaming session...")
        await processor.stop_streaming_transcription(session_id)
        print("✅ Streaming session stopped")
        
        print("\n" + "="*70)
        print("EXPECTED RUST STREAMING DEBUG OUTPUT:")
        print("🚀 RUST: SonioxStream::new called for session {session_id}")
        print("🔧 RUST: Starting Soniox WebSocket background handler for session {session_id}")
        print("")
        print("If you DON'T see these messages above, the streaming background task is not running!")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming())