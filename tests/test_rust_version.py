#!/usr/bin/env python3

import asyncio
import debabelizer

async def test_rust_version():
    print("🚀 Testing Rust version of debabelizer...")
    print(f"📦 Debabelizer version: {debabelizer.__version__}")
    print(f"📦 Debabelizer location: {debabelizer.__file__}")
    print(f"📦 Available attributes: {[attr for attr in dir(debabelizer) if not attr.startswith('_')]}")
    
    print("\n1. Creating VoiceProcessor with Soniox...")
    print("   (This should trigger our debug messages!)")
    
    try:
        processor = debabelizer.VoiceProcessor(stt_provider="soniox")
        print(f"✅ VoiceProcessor created: {type(processor)}")
        
        print("\n2. Starting streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="wav",
            sample_rate=16000
        )
        print(f"✅ Session started: {session_id}")
        
        print("\n3. Getting streaming results...")
        results_iter = await processor.get_streaming_results(session_id)
        print(f"✅ Iterator created: {type(results_iter)}")
        
        print("\n4. Sending audio chunk...")
        fake_audio = b'\x00' * 1024  # Fake audio data
        await processor.stream_audio(session_id, fake_audio)
        print("✅ Audio sent")
        
        print("\n5. Testing iterator (brief test)...")
        try:
            result = await asyncio.wait_for(results_iter.__anext__(), timeout=2.0)
            print(f"✅ Got result: {result}")
        except asyncio.TimeoutError:
            print("⏱️ Iterator timeout (expected)")
        except StopAsyncIteration:
            print("🛑 Iterator ended (this should not happen immediately!)")
        
        print("\n6. Cleaning up...")
        await processor.stop_streaming_transcription(session_id)
        print("✅ Session stopped")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rust_version())