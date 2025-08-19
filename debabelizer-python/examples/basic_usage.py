#!/usr/bin/env python3
"""
Basic usage example for Debabelizer Python bindings.

This example demonstrates how to:
1. Set up configuration
2. Create a VoiceProcessor
3. Transcribe audio to text
4. Synthesize text to speech
"""

import debabelizer
from debabelizer.utils import load_audio_file, get_audio_format_from_extension, create_synthesis_options


def main():
    """Main example function."""
    
    # Example 1: Basic setup with environment variables
    print("=== Example 1: Basic Setup ===")
    
    # Create processor using environment variables
    # Make sure you have DEEPGRAM_API_KEY and OPENAI_API_KEY set
    try:
        processor = debabelizer.VoiceProcessor()
        print(f"STT Provider: {processor.get_stt_provider_name()}")
        print(f"TTS Provider: {processor.get_tts_provider_name()}")
    except debabelizer.DebabelizerException as e:
        print(f"Failed to create processor: {e}")
        print("Make sure you have API keys configured in environment variables")
        return
    
    # Example 2: Programmatic configuration
    print("\n=== Example 2: Programmatic Configuration ===")
    
    config = {
        "preferences": {
            "stt_provider": "deepgram",
            "tts_provider": "openai",
            "optimize_for": "quality"
        },
        "deepgram": {
            "api_key": "your_deepgram_key_here",  # Replace with actual key
            "model": "nova-2",
            "language": "en-US"
        },
        "openai": {
            "api_key": "your_openai_key_here",  # Replace with actual key
            "tts_model": "tts-1-hd",
            "tts_voice": "alloy"
        }
    }
    
    try:
        processor = debabelizer.VoiceProcessor(config=config)
        print(f"Configured STT Provider: {processor.get_stt_provider_name()}")
        print(f"Configured TTS Provider: {processor.get_tts_provider_name()}")
    except debabelizer.DebabelizerException as e:
        print(f"Configuration failed: {e}")
    
    # Example 3: Audio transcription (requires actual audio file)
    print("\n=== Example 3: Audio Transcription ===")
    
    # Create sample audio data (in real usage, load from file)
    sample_audio_data = b"fake_audio_data_for_demo"  # Replace with actual audio
    
    audio_format = debabelizer.AudioFormat(
        format="wav",
        sample_rate=16000,
        channels=1,
        bit_depth=16
    )
    
    audio = debabelizer.AudioData(sample_audio_data, audio_format)
    
    try:
        result = processor.transcribe(audio)
        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence}")
        print(f"Language: {result.language_detected}")
        print(f"Duration: {result.duration}s")
        
        if result.words:
            print("Word timings:")
            for word in result.words[:5]:  # Show first 5 words
                print(f"  '{word.word}': {word.start_time:.2f}s - {word.end_time:.2f}s")
                
    except debabelizer.DebabelizerException as e:
        print(f"Transcription failed: {e}")
    
    # Example 4: Text-to-speech synthesis
    print("\n=== Example 4: Text-to-Speech Synthesis ===")
    
    text = "Hello, this is a test of the Debabelizer text-to-speech functionality!"
    
    # Create synthesis options
    options = create_synthesis_options(
        voice="alloy",
        speed=1.0,
        format="mp3"
    )
    
    try:
        synthesis_result = processor.synthesize(text, options)
        print(f"Generated audio format: {synthesis_result.format}")
        print(f"Audio size: {synthesis_result.size_bytes} bytes")
        print(f"Duration: {synthesis_result.duration}s")
        
        # Save to file
        output_file = "output_speech.mp3"
        with open(output_file, "wb") as f:
            f.write(synthesis_result.audio_data)
        print(f"Audio saved to: {output_file}")
        
    except debabelizer.DebabelizerException as e:
        print(f"Synthesis failed: {e}")
    
    # Example 5: List available voices
    print("\n=== Example 5: Available Voices ===")
    
    try:
        voices = processor.get_available_voices()
        print(f"Found {len(voices)} available voices:")
        
        for voice in voices[:10]:  # Show first 10 voices
            print(f"  {voice.voice_id}: {voice.name} ({voice.language})")
            if voice.description:
                print(f"    Description: {voice.description}")
                
    except debabelizer.DebabelizerException as e:
        print(f"Failed to get voices: {e}")
    
    # Example 6: Working with real audio files
    print("\n=== Example 6: Real Audio File Processing ===")
    
    # This example shows how to process real audio files
    audio_file_path = "sample_audio.wav"  # Replace with actual file path
    
    try:
        # Load audio file
        audio_data = load_audio_file(audio_file_path)
        format_name = get_audio_format_from_extension(audio_file_path)
        
        # Create audio format (you may need to adjust these parameters)
        audio_format = debabelizer.AudioFormat(
            format=format_name,
            sample_rate=16000,  # Adjust based on your file
            channels=1,         # Adjust based on your file
            bit_depth=16        # Adjust based on your file
        )
        
        # Create audio data object
        audio = debabelizer.AudioData(audio_data, audio_format)
        
        # Transcribe
        result = processor.transcribe(audio)
        print(f"File transcription: {result.text}")
        
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
        print("Create a sample audio file to test this functionality")
    except debabelizer.DebabelizerException as e:
        print(f"Audio processing failed: {e}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()