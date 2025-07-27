#!/usr/bin/env python3
"""
Text-to-speech example using Debabelizer

This example shows how to synthesize speech from text using different TTS providers.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import debabelizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig


async def synthesize_text(text: str, output_file: str, provider: str = "elevenlabs", voice_id: str = None):
    """
    Convert text to speech using the specified provider
    
    Args:
        text: Text to synthesize
        output_file: Output audio file path
        provider: TTS provider to use
        voice_id: Specific voice ID (optional)
    """
    
    # Create configuration
    config = DebabelizerConfig()
    
    # Check if provider is configured
    if not config.is_provider_configured(provider):
        print(f"❌ Provider '{provider}' is not configured!")
        print(f"Please set the API key environment variable for {provider}")
        return
    
    # Create voice processor
    processor = VoiceProcessor(
        tts_provider=provider,
        config=config
    )
    
    print(f"🗣️  Synthesizing text using {provider}...")
    print(f"📝 Text: {text}")
    
    try:
        # Set voice if specified
        synthesis_options = {}
        if voice_id:
            synthesis_options["voice_id"] = voice_id
        
        # Synthesize the text
        result = await processor.synthesize_text(
            text=text,
            output_file=output_file,
            **synthesis_options
        )
        
        print(f"✅ Synthesis complete!")
        print(f"🎵 Audio saved to: {output_file}")
        print(f"📊 Audio format: {result.format}")
        print(f"⏱️  Duration: {result.duration:.1f}s")
        print(f"📏 Size: {result.size_bytes} bytes")
        
        if result.voice_used:
            print(f"🎭 Voice: {result.voice_used}")
            
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")


async def list_available_voices(provider: str = "elevenlabs"):
    """
    List available voices for a provider
    
    Args:
        provider: TTS provider name
    """
    
    config = DebabelizerConfig()
    
    if not config.is_provider_configured(provider):
        print(f"❌ Provider '{provider}' is not configured!")
        return
    
    processor = VoiceProcessor(
        tts_provider=provider,
        config=config
    )
    
    print(f"🎭 Available voices for {provider}:")
    
    try:
        voices = await processor.get_available_voices()
        
        if not voices:
            print("No voices found or provider doesn't support voice listing")
            return
        
        print(f"Found {len(voices)} voices:\n")
        
        for voice in voices[:10]:  # Show first 10
            print(f"  ID: {voice.voice_id}")
            print(f"  Name: {voice.name}")
            if voice.language:
                print(f"  Language: {voice.language}")
            if voice.gender:
                print(f"  Gender: {voice.gender}")
            if voice.description:
                print(f"  Description: {voice.description[:100]}...")
            print()
            
        if len(voices) > 10:
            print(f"... and {len(voices) - 10} more voices")
            
    except Exception as e:
        print(f"❌ Failed to list voices: {e}")


async def compare_providers_tts(text: str, output_dir: str = "output"):
    """
    Compare TTS results from multiple providers
    
    Args:
        text: Text to synthesize
        output_dir: Directory to save audio files
    """
    
    config = DebabelizerConfig()
    configured_providers = config.get_configured_providers()["tts"]
    
    if not configured_providers:
        print("❌ No TTS providers are configured!")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🔄 Comparing {len(configured_providers)} TTS providers...")
    print(f"📝 Text: {text}")
    
    results = {}
    
    for provider in configured_providers:
        print(f"\n--- Testing {provider} ---")
        
        try:
            processor = VoiceProcessor(
                tts_provider=provider,
                config=config
            )
            
            output_file = output_path / f"{provider}_output.wav"
            
            result = await processor.synthesize_text(
                text=text,
                output_file=str(output_file)
            )
            
            results[provider] = result
            
            print(f"✅ {provider}: {result.duration:.1f}s, {result.size_bytes} bytes")
            print(f"   Saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ {provider} failed: {e}")
            results[provider] = None
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    for provider, result in results.items():
        if result:
            print(f"{provider:12} | Duration: {result.duration:.1f}s | Size: {result.size_bytes:,} bytes")
        else:
            print(f"{provider:12} | ❌ Failed")


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} \"<text>\" [output_file] [provider] [voice_id]")
        print(f"  {sys.argv[0]} voices [provider]")
        print(f"  {sys.argv[0]} compare \"<text>\"")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} \"Hello world!\" output.wav elevenlabs")
        print(f"  {sys.argv[0]} \"Hello world!\" output.wav elevenlabs 21m00Tcm4TlvDq8ikWAM") 
        print(f"  {sys.argv[0]} voices elevenlabs")
        print(f"  {sys.argv[0]} compare \"Hello world!\"")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "voices":
        provider = sys.argv[2] if len(sys.argv) > 2 else "elevenlabs"
        asyncio.run(list_available_voices(provider))
        
    elif command == "compare":
        if len(sys.argv) < 3:
            print("❌ Please provide text for comparison")
            sys.exit(1)
        text = sys.argv[2]
        asyncio.run(compare_providers_tts(text))
        
    else:
        # Text synthesis
        text = command
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
        provider = sys.argv[3] if len(sys.argv) > 3 else "elevenlabs"
        voice_id = sys.argv[4] if len(sys.argv) > 4 else None
        
        asyncio.run(synthesize_text(text, output_file, provider, voice_id))


if __name__ == "__main__":
    main()