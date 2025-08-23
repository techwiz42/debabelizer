#!/usr/bin/env python3
"""
Direct test to verify the Rust WebSocket background task is working
"""

import asyncio
import sys

sys.path.insert(0, '/home/peter/debabelizer/debabelizer-python/python')

async def test_rust_logging():
    """Test with explicit logging to see Rust tracing output"""
    
    print("🦀 Direct Rust Implementation Test")
    print("=" * 40)
    
    # Enable Rust logging by setting environment variable
    import os
    os.environ['RUST_LOG'] = 'debug'
    
    import debabelizer
    voice_service = debabelizer.VoiceProcessor()
    
    try:
        session_id = await voice_service.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            interim_results=True
        )
        print(f"✅ Session: {session_id}")
        
        # Try to get one result with detailed error reporting
        iterator = voice_service.get_streaming_results(session_id)
        
        print("Attempting first __anext__() call...")
        try:
            result = await iterator.__anext__()
            print(f"✅ Got result: {result}")
        except StopAsyncIteration:
            print("❌ StopAsyncIteration raised immediately")
        except Exception as e:
            print(f"❌ Exception: {e}")
            print(f"   Type: {type(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Session error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test complete")

if __name__ == "__main__":
    asyncio.run(test_rust_logging())