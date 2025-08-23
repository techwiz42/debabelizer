#!/usr/bin/env python3
"""
Test with explicit println! statements to bypass tracing issues
"""

import asyncio
import sys
sys.path.insert(0, 'debabelizer-python/python')

async def test_println():
    print("🔧 Testing with explicit println debugging...")
    
    import debabelizer
    voice_service = debabelizer.VoiceProcessor()
    
    print("1. Creating session...")
    session_id = await voice_service.start_streaming_transcription(
        audio_format="pcm",
        sample_rate=16000,
        enable_language_identification=True,
        interim_results=True
    )
    print(f"2. Session created: {session_id}")
    
    print("3. Waiting for background task to initialize...")
    await asyncio.sleep(0.5)
    
    print("4. Sending audio chunk...")
    audio_chunk = bytearray(1600)  # 100ms of 16kHz PCM
    for i in range(0, len(audio_chunk), 2):
        import struct
        # Generate sine wave
        sample = int(8000 * ((i // 2) % 160 - 80) / 80)
        audio_chunk[i:i+2] = struct.pack('<h', sample)
    
    await voice_service.stream_audio(session_id, bytes(audio_chunk))
    print("5. Audio chunk sent")
    
    print("6. Getting iterator...")
    iterator = voice_service.get_streaming_results(session_id)
    
    print("7. Testing single __anext__ call with 3 second timeout...")
    try:
        result = await asyncio.wait_for(iterator.__anext__(), timeout=3.0)
        print(f"8. ✅ SUCCESS - Got result: text='{result.text}', final={result.is_final}")
        if hasattr(result, 'metadata') and result.metadata:
            print(f"   Metadata: {result.metadata}")
    except asyncio.TimeoutError:
        print("8. ⏰ TIMEOUT - No response in 3 seconds")
    except StopAsyncIteration:
        print("8. 🛑 STOP - Iterator ended immediately")
    except Exception as e:
        print(f"8. ❌ ERROR - {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("9. Cleanup...")
    await voice_service.stop_streaming_transcription(session_id)
    print("10. Test complete")

if __name__ == "__main__":
    asyncio.run(test_println())