#!/usr/bin/env python3
"""
Test by explicitly configuring Soniox as the STT provider
"""

import asyncio
import sys
sys.path.insert(0, 'debabelizer-python/python')

async def test_explicit_soniox():
    print("🎯 Testing with explicitly configured Soniox provider...")
    
    import debabelizer
    
    # Create config that explicitly selects Soniox
    config_dict = {
        "preferences": {
            "stt_provider": "soniox",
        },
        "soniox": {
            "api_key": "cfb7e338d94102d7b59deea599b238e4ae2fa8085830097c7e3ed89696c4ec95",
        }
    }
    
    print("1. Creating VoiceProcessor with explicit Soniox config...")
    config = debabelizer.DebabelizerConfig(config_dict)
    voice_service = debabelizer.VoiceProcessor(config=config)
    print("2. VoiceProcessor created")
    
    try:
        print("3. Starting streaming session...")
        session_id = await voice_service.start_streaming_transcription(
            audio_format="pcm",
            sample_rate=16000,
            enable_language_identification=True,
            interim_results=True
        )
        print(f"4. Session created: {session_id}")
        
        print("5. Waiting for background task...")
        await asyncio.sleep(1.0)
        
        print("6. Test complete - checking if Soniox provider was used")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_explicit_soniox())