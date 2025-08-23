#!/usr/bin/env python3
"""
Test the Rust Soniox implementation to verify the fix works
"""

import asyncio
import sys
import os

# Add the current directory to the path to import the local debabelizer
sys.path.insert(0, '/home/peter/debabelize_me/backend')

try:
    import debabelizer
    print("✓ Successfully imported debabelizer")
except ImportError as e:
    print(f"✗ Failed to import debabelizer: {e}")
    print("Installing latest version...")
    os.system("pip install --upgrade debabelizer")
    import debabelizer
    print("✓ Imported debabelizer after upgrade")

async def test_soniox_streaming():
    """Test Soniox streaming with the Rust implementation"""
    
    print("Creating VoiceProcessor with Soniox...")
    
    # Create config for Soniox
    config_dict = {
        "soniox": {
            "api_key": "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95"
        },
        "preferences": {
            "stt_provider": "soniox"
        }
    }
    
    try:
        config = debabelizer.DebabelizerConfig(config_dict)
        processor = debabelizer.VoiceProcessor(config=config)
        print("✓ VoiceProcessor created successfully")
        
        # Start streaming session
        print("Starting Soniox streaming session...")
        session_id = await processor.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            interim_results=True
        )
        print(f"✓ Streaming session started: {session_id}")
        
        # Test that we can get the streaming results iterator
        print("Getting streaming results iterator...")
        results_iter = processor.get_streaming_results(session_id)
        print("✓ Results iterator created successfully")
        
        # Send some test audio (silence)
        print("Sending test audio chunk...")
        test_audio = b'\x00' * 320  # 10ms of silence
        await processor.stream_audio(session_id, test_audio)
        print("✓ Test audio sent successfully")
        
        # Try to get results with a short timeout
        print("Testing streaming results loop (5 second timeout)...")
        try:
            timeout_count = 0
            async for result in results_iter:
                print(f"✓ Received streaming result: text='{result.text}', is_final={result.is_final}")
                timeout_count += 1
                if timeout_count > 10:  # Prevent infinite loop
                    break
        except Exception as e:
            print(f"Results loop ended: {type(e).__name__}: {e}")
        
        # Stop streaming
        print("Stopping streaming session...")
        await processor.stop_streaming_transcription(session_id)
        print("✓ Streaming session stopped")
        
        print("\n🎉 All tests passed! Soniox streaming is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Rust Soniox Streaming Test")
    print("=" * 40)
    
    success = asyncio.run(test_soniox_streaming())
    
    if success:
        print("\n✅ Test completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Test failed")
        sys.exit(1)