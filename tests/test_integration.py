"""
Integration tests for Debabelizer

These tests verify end-to-end functionality with real providers (when configured).
They are marked as integration tests and can be skipped if providers are not configured.
"""

import pytest
import asyncio
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig, create_processor
from debabelizer.providers.base import TranscriptionResult, SynthesisResult, ProviderError
from tests.conftest import TestLanguages


@pytest.mark.integration
class TestRealProviderIntegration:
    """Integration tests with real providers (requires API keys)"""

    def _get_real_config(self):
        """Get configuration from environment variables"""
        return DebabelizerConfig()  # Will load from environment

    def _get_available_providers(self):
        """Get actually configured providers"""
        config = self._get_real_config()
        return config.get_configured_providers()

    @pytest.mark.asyncio
    async def test_real_stt_transcription(self, temp_audio_file):
        """Test transcription with real STT providers"""
        config = self._get_real_config()
        available_stt = config.get_configured_providers()["stt"]
        
        if not available_stt:
            pytest.skip("No STT providers configured with API keys")
        
        for provider_name in available_stt[:1]:  # Test first available provider
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            try:
                result = await processor.transcribe_file(str(temp_audio_file))
                
                # Verify result structure
                assert isinstance(result, TranscriptionResult)
                assert hasattr(result, 'text')
                assert hasattr(result, 'confidence')
                assert hasattr(result, 'duration')
                
                # Text might be empty for silence, but should be a string
                assert isinstance(result.text, str)
                assert isinstance(result.confidence, (int, float))
                assert isinstance(result.duration, (int, float))
                
                print(f"✅ {provider_name} transcription: '{result.text}' (confidence: {result.confidence:.2f})")
                
            except ProviderError as e:
                pytest.fail(f"Provider {provider_name} failed: {e}")

    @pytest.mark.asyncio
    async def test_real_tts_synthesis(self, temp_directory):
        """Test synthesis with real TTS providers"""
        config = self._get_real_config()
        available_tts = config.get_configured_providers()["tts"]
        
        if not available_tts:
            pytest.skip("No TTS providers configured with API keys")
        
        test_text = "Hello, this is a test of the text-to-speech system."
        
        for provider_name in available_tts[:1]:  # Test first available provider
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            output_file = temp_directory / f"test_output_{provider_name}.wav"
            
            try:
                result = await processor.synthesize_text(
                    text=test_text,
                    output_file=str(output_file)
                )
                
                # Verify result structure
                assert isinstance(result, SynthesisResult)
                assert hasattr(result, 'audio_data')
                assert hasattr(result, 'format')
                assert hasattr(result, 'duration')
                assert hasattr(result, 'size_bytes')
                
                # Verify audio data
                assert isinstance(result.audio_data, bytes)
                assert len(result.audio_data) > 0
                assert result.duration > 0
                assert result.size_bytes > 0
                
                # Verify file was created
                assert output_file.exists()
                assert output_file.stat().st_size > 0
                
                print(f"✅ {provider_name} synthesis: {result.duration:.1f}s, {result.size_bytes} bytes")
                
            except ProviderError as e:
                pytest.fail(f"Provider {provider_name} failed: {e}")

    @pytest.mark.asyncio
    async def test_language_detection_with_real_providers(self):
        """Test automatic language detection with real providers"""
        config = self._get_real_config()
        available_stt = config.get_configured_providers()["stt"]
        
        if not available_stt:
            pytest.skip("No STT providers configured with API keys")
        
        # Use a provider that supports language detection
        for provider_name in available_stt:
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Test with silence (should still return language info if supported)
            try:
                # Create a longer audio file for better detection
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    # Create 2-second WAV file
                    wav_header = (
                        b'RIFF' + (32044).to_bytes(4, byteorder='little') + b'WAVE'
                        + b'fmt ' + (16).to_bytes(4, byteorder='little')
                        + (1).to_bytes(2, byteorder='little')   # PCM
                        + (1).to_bytes(2, byteorder='little')   # Mono
                        + (16000).to_bytes(4, byteorder='little')  # 16kHz
                        + (32000).to_bytes(4, byteorder='little')
                        + (2).to_bytes(2, byteorder='little')
                        + (16).to_bytes(2, byteorder='little')
                        + b'data' + (32000).to_bytes(4, byteorder='little')
                    )
                    
                    tmp_file.write(wav_header + b'\x00' * 32000)  # 2 seconds of silence
                    tmp_file.flush()
                    
                    # Test auto language detection (don't specify language)
                    result = await processor.transcribe_file(tmp_file.name)
                    
                    # Language detection may or may not work with silence
                    # Just verify the interface works
                    assert isinstance(result, TranscriptionResult)
                    
                    if result.language_detected:
                        print(f"✅ {provider_name} detected language: {result.language_detected}")
                    else:
                        print(f"ℹ️  {provider_name} did not detect language (likely due to silence)")
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
                    break  # Test one provider to avoid quota issues
                    
            except ProviderError as e:
                print(f"⚠️  {provider_name} language detection test failed: {e}")
                continue

    @pytest.mark.asyncio
    async def test_voice_listing_with_real_providers(self):
        """Test voice listing with real TTS providers"""
        config = self._get_real_config()
        available_tts = config.get_configured_providers()["tts"]
        
        if not available_tts:
            pytest.skip("No TTS providers configured with API keys")
        
        for provider_name in available_tts[:1]:  # Test first available provider
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            try:
                voices = await processor.get_available_voices()
                
                # Verify voice list structure
                assert isinstance(voices, list)
                
                if voices:  # Some providers may not support voice listing
                    for voice in voices[:3]:  # Check first few voices
                        assert hasattr(voice, 'voice_id')
                        assert hasattr(voice, 'name')
                        assert isinstance(voice.voice_id, str)
                        assert isinstance(voice.name, str)
                        assert len(voice.voice_id) > 0
                        assert len(voice.name) > 0
                    
                    print(f"✅ {provider_name} voices: {len(voices)} available")
                    
                    # Show first few voices
                    for voice in voices[:3]:
                        lang_info = f" ({voice.language})" if hasattr(voice, 'language') and voice.language else ""
                        print(f"   - {voice.name} (ID: {voice.voice_id}){lang_info}")
                else:
                    print(f"ℹ️  {provider_name} returned no voices or doesn't support voice listing")
                
            except ProviderError as e:
                print(f"⚠️  {provider_name} voice listing failed: {e}")

    @pytest.mark.asyncio
    async def test_end_to_end_voice_processing(self, temp_directory):
        """Test complete voice processing pipeline with real providers"""
        config = self._get_real_config()
        providers = config.get_configured_providers()
        
        if not providers["stt"] or not providers["tts"]:
            pytest.skip("Both STT and TTS providers required for end-to-end test")
        
        # Use first available providers
        stt_provider = providers["stt"][0]
        tts_provider = providers["tts"][0]
        
        processor = VoiceProcessor(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            config=config
        )
        
        # Step 1: Create initial audio (TTS)
        initial_text = "Testing end to end voice processing."
        initial_audio = temp_directory / "initial.wav"
        
        try:
            tts_result = await processor.synthesize_text(
                text=initial_text,
                output_file=str(initial_audio)
            )
            
            assert initial_audio.exists()
            print(f"✅ Generated initial audio: {tts_result.duration:.1f}s")
            
            # Step 2: Transcribe the generated audio (STT)
            stt_result = await processor.transcribe_file(str(initial_audio))
            
            print(f"✅ Transcribed audio: '{stt_result.text}' (confidence: {stt_result.confidence:.2f})")
            
            # Step 3: Generate response (TTS again)
            response_text = f"You said: {stt_result.text}"
            response_audio = temp_directory / "response.wav"
            
            response_result = await processor.synthesize_text(
                text=response_text,
                output_file=str(response_audio)
            )
            
            assert response_audio.exists()
            print(f"✅ Generated response audio: {response_result.duration:.1f}s")
            
            # Verify the pipeline worked
            assert isinstance(stt_result.text, str)
            assert len(stt_result.text) > 0  # Should have transcribed something
            assert tts_result.duration > 0
            assert response_result.duration > 0
            
            print(f"✅ End-to-end test completed successfully with {stt_provider} + {tts_provider}")
            
        except ProviderError as e:
            pytest.fail(f"End-to-end test failed: {e}")


@pytest.mark.integration
class TestConvenienceFunctions:
    """Test convenience functions with real configuration"""

    def test_create_processor_function(self):
        """Test the create_processor convenience function"""
        config = DebabelizerConfig()
        providers = config.get_configured_providers()
        
        if not providers["stt"] and not providers["tts"]:
            pytest.skip("No providers configured for testing")
        
        # Test with defaults
        try:
            processor = create_processor()
            assert isinstance(processor, VoiceProcessor)
            print("✅ create_processor() with defaults succeeded")
        except ProviderError:
            print("ℹ️  create_processor() requires provider configuration")
        
        # Test with specific providers
        if providers["stt"]:
            stt_provider = providers["stt"][0]
            try:
                processor = create_processor(stt_provider=stt_provider)
                assert processor.stt_provider_name == stt_provider
                print(f"✅ create_processor() with {stt_provider} succeeded")
            except ProviderError as e:
                print(f"⚠️  create_processor() with {stt_provider} failed: {e}")

    def test_config_validation(self):
        """Test configuration validation"""
        config = DebabelizerConfig()
        
        # Test provider configuration check
        providers = config.get_configured_providers()
        
        print("📊 Provider Configuration Status:")
        print(f"   STT Providers: {providers['stt'] if providers['stt'] else 'None configured'}")
        print(f"   TTS Providers: {providers['tts'] if providers['tts'] else 'None configured'}")
        
        # Test individual provider checks
        test_providers = ["soniox", "deepgram", "elevenlabs", "openai", "azure"]
        
        for provider in test_providers:
            is_configured = config.is_provider_configured(provider)
            status = "✅ Configured" if is_configured else "❌ Not configured"
            print(f"   {provider}: {status}")


@pytest.mark.integration
@pytest.mark.slow
class TestMultiLanguageIntegration:
    """Integration tests with multiple languages"""

    @pytest.mark.asyncio
    async def test_multi_language_transcription(self, temp_directory):
        """Test transcription with multiple languages (if provider supports it)"""
        config = DebabelizerConfig()
        available_stt = config.get_configured_providers()["stt"]
        
        if not available_stt:
            pytest.skip("No STT providers configured")
        
        # Test with first provider that supports multiple languages
        provider_name = available_stt[0]
        processor = VoiceProcessor(stt_provider=provider_name, config=config)
        
        # Test a subset of our languages to avoid quota issues
        test_languages = ["en", "es", "de"]  # English, Spanish, German
        
        for lang_code in test_languages:
            # Create a test audio file
            test_file = temp_directory / f"test_{lang_code}.wav"
            
            # Create minimal WAV file
            wav_header = (
                b'RIFF' + (1044).to_bytes(4, byteorder='little') + b'WAVE'
                + b'fmt ' + (16).to_bytes(4, byteorder='little')
                + (1).to_bytes(2, byteorder='little')   # PCM
                + (1).to_bytes(2, byteorder='little')   # Mono
                + (16000).to_bytes(4, byteorder='little')  # 16kHz
                + (32000).to_bytes(4, byteorder='little')
                + (2).to_bytes(2, byteorder='little')
                + (16).to_bytes(2, byteorder='little')
                + b'data' + (1000).to_bytes(4, byteorder='little')
            )
            
            with open(test_file, 'wb') as f:
                f.write(wav_header + b'\x00' * 1000)
            
            try:
                # Test with specific language
                result = await processor.transcribe_file(str(test_file), language=lang_code)
                
                assert isinstance(result, TranscriptionResult)
                print(f"✅ {provider_name} transcription ({lang_code}): confidence {result.confidence:.2f}")
                
                # Test auto-detection
                auto_result = await processor.transcribe_file(str(test_file))
                assert isinstance(auto_result, TranscriptionResult)
                print(f"✅ {provider_name} auto-detection ({lang_code}): detected {auto_result.language_detected}")
                
            except ProviderError as e:
                print(f"⚠️  {provider_name} failed for language {lang_code}: {e}")

    @pytest.mark.asyncio
    async def test_provider_performance_comparison(self, temp_audio_file):
        """Compare performance across multiple providers"""
        config = DebabelizerConfig()
        providers = config.get_configured_providers()
        
        if len(providers["stt"]) < 2:
            pytest.skip("Need at least 2 STT providers for comparison")
        
        results = {}
        
        for provider_name in providers["stt"][:3]:  # Test up to 3 providers
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            try:
                import time
                start_time = time.time()
                
                result = await processor.transcribe_file(str(temp_audio_file))
                
                end_time = time.time()
                duration = end_time - start_time
                
                results[provider_name] = {
                    "success": True,
                    "duration": duration,
                    "confidence": result.confidence,
                    "text": result.text,
                    "language": result.language_detected
                }
                
                print(f"✅ {provider_name}: {duration:.2f}s, confidence: {result.confidence:.2f}")
                
            except ProviderError as e:
                results[provider_name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ {provider_name}: {e}")
        
        # Print comparison summary
        print("\n📊 Provider Performance Comparison:")
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if len(successful_results) >= 2:
            fastest = min(successful_results.items(), key=lambda x: x[1]["duration"])
            highest_confidence = max(successful_results.items(), key=lambda x: x[1]["confidence"])
            
            print(f"   Fastest: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
            print(f"   Highest Confidence: {highest_confidence[0]} ({highest_confidence[1]['confidence']:.2f})")
        
        # Verify at least one provider worked
        assert any(r["success"] for r in results.values()), "No providers succeeded"


if __name__ == "__main__":
    # Allow running integration tests directly
    print("🧪 Running Debabelizer Integration Tests")
    print("=" * 50)
    
    config = DebabelizerConfig()
    providers = config.get_configured_providers()
    
    if not providers["stt"] and not providers["tts"]:
        print("❌ No providers configured!")
        print("Please set API keys as environment variables:")
        print("   export SONIOX_API_KEY=your_key")
        print("   export DEEPGRAM_API_KEY=your_key") 
        print("   export ELEVENLABS_API_KEY=your_key")
        exit(1)
    
    print(f"✅ Found STT providers: {providers['stt']}")
    print(f"✅ Found TTS providers: {providers['tts']}")
    print()
    
    # Run a simple test
    import asyncio
    
    async def simple_test():
        if providers["tts"]:
            processor = create_processor(tts_provider=providers["tts"][0])
            result = await processor.synthesize_text("Integration test successful!", "test_output.wav")
            print(f"✅ Simple TTS test passed: {result.duration:.1f}s audio generated")
    
    asyncio.run(simple_test())