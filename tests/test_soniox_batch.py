#!/usr/bin/env python3
"""Test Soniox batch transcription."""

import asyncio
import os
from pathlib import Path

# Import the Rust-based debabelizer
import debabelizer

async def test_soniox_batch():
    """Test Soniox batch transcription."""
    print("Testing Soniox batch transcription...")
    
    # Initialize processor
    config_dict = {
        "preferences": {
            "stt_provider": "soniox",
            "tts_provider": "openai"
        },
        "soniox": {
            "api_key": os.getenv("SONIOX_API_KEY")
        }
    }
    
    config = debabelizer.DebabelizerConfig(config_dict)
    processor = debabelizer.VoiceProcessor(config=config)
    print("✅ Processor initialized with Soniox")
    
    # Load test audio
    audio_path = Path("/home/peter/debabelizer/english_sample.wav")
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
        
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    print(f"✅ Loaded audio file: {len(audio_data)} bytes")
    
    # Create audio object
    audio_format = debabelizer.AudioFormat("wav", 16000, 1, 16)
    audio = debabelizer.AudioData(audio_data, audio_format)
    
    # Try batch transcription
    print("\n🎤 Starting batch transcription...")
    try:
        result = await processor.transcribe(audio)
        print(f"\n✅ Transcription successful!")
        print(f"Text: '{result.text}'")
        print(f"Confidence: {result.confidence}")
        if result.language_detected:
            print(f"Language: {result.language_detected}")
    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set Soniox API key if not already set
    if not os.getenv("SONIOX_API_KEY"):
        # Try to load from .env file
        env_path = Path("/home/peter/debabelizer/.env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("SONIOX_API_KEY="):
                        os.environ["SONIOX_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                        break
    
    asyncio.run(test_soniox_batch())