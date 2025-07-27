#!/usr/bin/env python3
"""
Basic audio transcription example using Debabelizer

This example shows how to transcribe an audio file using different STT providers.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import debabelizer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig


async def transcribe_file(audio_file_path: str, provider: str = "soniox"):
    """
    Transcribe an audio file using the specified provider
    
    Args:
        audio_file_path: Path to the audio file
        provider: STT provider to use ("soniox", "deepgram", etc.)
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
        stt_provider=provider,
        config=config
    )
    
    print(f"🎤 Transcribing {audio_file_path} using {provider}...")
    
    try:
        # Transcribe the file
        result = await processor.transcribe_file(audio_file_path)
        
        print(f"✅ Transcription complete!")
        print(f"📝 Text: {result.text}")
        print(f"🔍 Confidence: {result.confidence:.2f}")
        print(f"⏱️  Duration: {result.duration:.1f}s")
        
        if result.language_detected:
            print(f"🌍 Language: {result.language_detected}")
        
        if result.words:
            print(f"📊 Word count: {len(result.words)}")
            
    except Exception as e:
        print(f"❌ Transcription failed: {e}")


async def compare_providers(audio_file_path: str):
    """
    Compare transcription results from multiple providers
    
    Args:
        audio_file_path: Path to the audio file
    """
    
    config = DebabelizerConfig()
    configured_providers = config.get_configured_providers()["stt"]
    
    if not configured_providers:
        print("❌ No STT providers are configured!")
        return
    
    print(f"🔄 Comparing {len(configured_providers)} providers...")
    
    results = {}
    
    for provider in configured_providers:
        print(f"\n--- Testing {provider} ---")
        
        try:
            processor = VoiceProcessor(
                stt_provider=provider,
                config=config
            )
            
            result = await processor.transcribe_file(audio_file_path)
            results[provider] = result
            
            print(f"✅ {provider}: {result.text[:100]}...")
            print(f"   Confidence: {result.confidence:.2f}, Duration: {result.duration:.1f}s")
            
        except Exception as e:
            print(f"❌ {provider} failed: {e}")
            results[provider] = None
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    for provider, result in results.items():
        if result:
            print(f"{provider:12} | Conf: {result.confidence:.2f} | Text: {result.text[:50]}...")
        else:
            print(f"{provider:12} | ❌ Failed")


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <audio_file> [provider]")
        print(f"  {sys.argv[0]} <audio_file> compare")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} sample.wav soniox")
        print(f"  {sys.argv[0]} sample.wav deepgram") 
        print(f"  {sys.argv[0]} sample.wav compare")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        provider_or_action = sys.argv[2]
        
        if provider_or_action == "compare":
            asyncio.run(compare_providers(audio_file))
        else:
            asyncio.run(transcribe_file(audio_file, provider_or_action))
    else:
        # Default to soniox
        asyncio.run(transcribe_file(audio_file, "soniox"))


if __name__ == "__main__":
    main()