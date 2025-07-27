"""
Provider-agnostic unit tests for TTS providers

These tests work with any TTS provider that implements the TTSProvider interface.
Tests focus on the contract and behavior rather than implementation details.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig
from debabelizer.providers.base import SynthesisResult, Voice, ProviderError


class TestTTSProviderInterface:
    """Provider-agnostic tests for TTS provider interface"""

    # Eight test languages from Thanotopolis tests
    TEST_LANGUAGES = {
        "en": "english",    # English
        "es": "spanish",    # Spanish  
        "de": "german",     # German
        "ru": "russian",    # Russian
        "zh": "chinese",    # Chinese
        "tl": "tagalog",    # Tagalog (Filipino)
        "ko": "korean",     # Korean
        "vi": "vietnamese"  # Vietnamese
    }

    def _get_available_tts_providers(self):
        """Get list of configured TTS providers for testing"""
        config = DebabelizerConfig()
        return config.get_configured_providers()["tts"]

    @pytest.mark.asyncio
    async def test_basic_text_synthesis(self):
        """Test basic text-to-speech synthesis functionality"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Mock the actual provider to avoid API calls
            mock_provider = AsyncMock()
            mock_provider.synthesize_text.return_value = SynthesisResult(
                audio_data=b"fake_audio_data_hello_world",
                format="wav",
                sample_rate=16000,
                duration=1.5,
                size_bytes=48000
            )
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                result = await processor.synthesize_text(
                    text="Hello world",
                    output_file="test_output.wav"
                )
                
                # Test interface contract
                assert isinstance(result, SynthesisResult)
                assert hasattr(result, 'audio_data')
                assert hasattr(result, 'format')
                assert hasattr(result, 'sample_rate')
                assert hasattr(result, 'duration')
                assert hasattr(result, 'size_bytes')
                
                # Test basic values
                assert isinstance(result.audio_data, bytes)
                assert isinstance(result.format, str)
                assert isinstance(result.sample_rate, int)
                assert isinstance(result.duration, (int, float))
                assert isinstance(result.size_bytes, int)
                assert result.duration > 0
                assert result.size_bytes > 0

    @pytest.mark.asyncio
    async def test_synthesis_with_voice_selection(self):
        """Test text synthesis with specific voice selection"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            mock_provider = AsyncMock()
            mock_provider.synthesize_text.return_value = SynthesisResult(
                audio_data=b"voice_specific_audio_data",
                format="wav",
                sample_rate=16000,
                duration=2.0,
                size_bytes=64000,
                voice_used="test_voice_123"
            )
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                result = await processor.synthesize_text(
                    text="Hello with specific voice",
                    output_file="voice_test.wav",
                    voice_id="test_voice_123"
                )
                
                assert result.voice_used == "test_voice_123"
                
                # Verify provider was called with correct voice
                mock_provider.synthesize_text.assert_called_with(
                    "Hello with specific voice",
                    output_file="voice_test.wav",
                    voice_id="test_voice_123"
                )

    @pytest.mark.asyncio
    async def test_synthesis_with_language_specific_content(self):
        """Test synthesis with content in different languages"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Test with each of our test languages
            for lang_code, lang_name in self.TEST_LANGUAGES.items():
                mock_provider = AsyncMock()
                mock_provider.synthesize_text.return_value = SynthesisResult(
                    audio_data=f"audio_for_{lang_name}".encode(),
                    format="wav",
                    sample_rate=16000,
                    duration=1.8,
                    size_bytes=57600,
                    voice_used=f"{lang_name}_voice",
                    language=lang_code
                )
                
                with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                    result = await processor.synthesize_text(
                        text=f"Hello in {lang_name}",
                        output_file=f"{lang_name}_output.wav",
                        language=lang_code
                    )
                    
                    assert lang_name.encode() in result.audio_data
                    if hasattr(result, 'language'):
                        assert result.language == lang_code

    @pytest.mark.asyncio
    async def test_get_available_voices(self):
        """Test retrieving available voices from providers"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Mock voice list
            mock_voices = [
                Voice(
                    voice_id="voice_1",
                    name="Voice One",
                    language="en-US",
                    gender="female",
                    description="Professional female voice"
                ),
                Voice(
                    voice_id="voice_2",
                    name="Voice Two", 
                    language="es-ES",
                    gender="male",
                    description="Spanish male voice"
                ),
                Voice(
                    voice_id="voice_3",
                    name="Voice Three",
                    language="fr-FR",
                    gender="neutral",
                    description="French neutral voice"
                )
            ]
            
            mock_provider = AsyncMock()
            mock_provider.get_available_voices.return_value = mock_voices
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                voices = await processor.get_available_voices()
                
                assert len(voices) == 3
                
                for voice in voices:
                    assert isinstance(voice, Voice)
                    assert hasattr(voice, 'voice_id')
                    assert hasattr(voice, 'name')
                    assert hasattr(voice, 'language')
                    assert hasattr(voice, 'gender')
                    assert hasattr(voice, 'description')
                    
                    # Test required fields
                    assert isinstance(voice.voice_id, str)
                    assert isinstance(voice.name, str)
                    assert len(voice.voice_id) > 0
                    assert len(voice.name) > 0

    @pytest.mark.asyncio
    async def test_streaming_synthesis_lifecycle(self):
        """Test streaming TTS synthesis lifecycle"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            mock_provider = AsyncMock()
            mock_provider.start_streaming_synthesis.return_value = "tts_session_456"
            mock_provider.get_streaming_audio.return_value = b"streaming_audio_chunk"
            mock_provider.end_streaming_synthesis.return_value = True
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                # Start streaming synthesis
                session_id = await processor.start_streaming_tts(
                    text="Streaming synthesis test",
                    voice_id="test_voice"
                )
                
                assert isinstance(session_id, str)
                assert len(session_id) > 0
                
                # Get streaming audio
                audio_chunk = await processor.get_streaming_audio(session_id)
                
                assert isinstance(audio_chunk, bytes)
                assert len(audio_chunk) > 0
                
                # End streaming
                ended = await processor.end_streaming_synthesis(session_id)
                assert ended is True

    @pytest.mark.asyncio
    async def test_audio_format_options(self):
        """Test synthesis with different audio format options"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        audio_formats = ["wav", "mp3", "ogg", "flac"]
        sample_rates = [8000, 16000, 22050, 44100]
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            for audio_format in audio_formats:
                for sample_rate in sample_rates:
                    mock_provider = AsyncMock()
                    mock_provider.synthesize_text.return_value = SynthesisResult(
                        audio_data=f"audio_data_{audio_format}_{sample_rate}".encode(),
                        format=audio_format,
                        sample_rate=sample_rate,
                        duration=1.0,
                        size_bytes=sample_rate  # Mock size based on sample rate
                    )
                    
                    with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                        result = await processor.synthesize_text(
                            text="Format test",
                            output_file=f"test.{audio_format}",
                            audio_format=audio_format,
                            sample_rate=sample_rate
                        )
                        
                        assert result.format == audio_format
                        assert result.sample_rate == sample_rate
                        assert audio_format.encode() in result.audio_data

    @pytest.mark.asyncio
    async def test_long_text_handling(self):
        """Test synthesis of long text content"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        # Create long text (simulate article or document)
        long_text = " ".join([
            "This is a very long text that needs to be synthesized.",
            "It contains multiple sentences and should test the provider's",
            "ability to handle longer content gracefully.",
            "Some providers may need to chunk this content.",
            "The synthesis should still work correctly and produce",
            "a complete audio output that covers all the text."
        ] * 10)  # Repeat to make it really long
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            mock_provider = AsyncMock()
            mock_provider.synthesize_text.return_value = SynthesisResult(
                audio_data=b"long_audio_data" * 100,  # Simulate longer audio
                format="wav",
                sample_rate=16000,
                duration=len(long_text) * 0.05,  # Approximate duration
                size_bytes=len(long_text) * 100
            )
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                result = await processor.synthesize_text(
                    text=long_text,
                    output_file="long_text_output.wav"
                )
                
                # Should handle long text without errors
                assert len(result.audio_data) > 1000  # Should be substantial
                assert result.duration > 10  # Should be reasonably long
                
                # Verify the full text was passed to provider
                call_args = mock_provider.synthesize_text.call_args
                assert call_args[0][0] == long_text

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in TTS operations"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Test synthesis error
            mock_provider = AsyncMock()
            mock_provider.synthesize_text.side_effect = ProviderError("TTS API error")
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                with pytest.raises(ProviderError, match="TTS API error"):
                    await processor.synthesize_text("Test text", "output.wav")
            
            # Test voice listing error
            mock_provider.get_available_voices.side_effect = ProviderError("Voice list failed")
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                with pytest.raises(ProviderError, match="Voice list failed"):
                    await processor.get_available_voices()

    @pytest.mark.asyncio
    async def test_concurrent_synthesis_operations(self):
        """Test concurrent TTS synthesis operations"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            mock_provider = AsyncMock()
            
            # Create multiple mock responses
            mock_responses = [
                SynthesisResult(
                    audio_data=f"audio_data_{i}".encode(),
                    format="wav",
                    sample_rate=16000,
                    duration=1.0 + i * 0.5,
                    size_bytes=16000 + i * 8000
                )
                for i in range(3)
            ]
            
            mock_provider.synthesize_text.side_effect = mock_responses
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                # Start multiple concurrent synthesis operations
                tasks = [
                    processor.synthesize_text(f"Text {i}", f"output_{i}.wav")
                    for i in range(3)
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all completed successfully
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert isinstance(result, SynthesisResult)
                    assert f"audio_data_{i}".encode() == result.audio_data

    @pytest.mark.asyncio
    async def test_voice_filtering_by_language(self):
        """Test filtering available voices by language"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Create voices for different languages
            all_voices = []
            for lang_code, lang_name in self.TEST_LANGUAGES.items():
                all_voices.append(Voice(
                    voice_id=f"voice_{lang_code}",
                    name=f"{lang_name.title()} Voice",
                    language=lang_code,
                    gender="neutral"
                ))
            
            mock_provider = AsyncMock()
            mock_provider.get_available_voices.return_value = all_voices
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                all_voices_result = await processor.get_available_voices()
                
                # Test that we get all voices
                assert len(all_voices_result) == len(self.TEST_LANGUAGES)
                
                # Test language filtering (if supported by processor)
                for lang_code in self.TEST_LANGUAGES.keys():
                    lang_voices = [v for v in all_voices_result if v.language == lang_code]
                    assert len(lang_voices) == 1
                    assert lang_voices[0].language == lang_code

    @pytest.mark.asyncio
    async def test_synthesis_result_structure(self):
        """Test that synthesis results have the expected structure"""
        providers = self._get_available_tts_providers()
        
        if not providers:
            pytest.skip("No TTS providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            mock_provider = AsyncMock()
            mock_provider.synthesize_text.return_value = SynthesisResult(
                audio_data=b"complete_audio_data",
                format="wav",
                sample_rate=22050,
                duration=3.14,
                size_bytes=138557,
                voice_used="premium_voice_id",
                language="en-US"
            )
            
            with patch.object(processor, '_get_tts_provider', return_value=mock_provider):
                result = await processor.synthesize_text("Structure test", "test.wav")
                
                # Test required fields
                assert isinstance(result.audio_data, bytes)
                assert isinstance(result.format, str)
                assert isinstance(result.sample_rate, int)
                assert isinstance(result.duration, (int, float))
                assert isinstance(result.size_bytes, int)
                
                # Test constraints
                assert len(result.audio_data) > 0
                assert result.sample_rate > 0
                assert result.duration > 0
                assert result.size_bytes > 0
                
                # Test optional fields
                if hasattr(result, 'voice_used') and result.voice_used:
                    assert isinstance(result.voice_used, str)
                
                if hasattr(result, 'language') and result.language:
                    assert isinstance(result.language, str)


class TestTTSProviderConfiguration:
    """Test TTS provider configuration and initialization"""

    def test_provider_initialization_with_config(self):
        """Test that providers can be initialized with configuration"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "test_elevenlabs_key", "model": "eleven_turbo_v2_5"},
            "openai": {"api_key": "test_openai_key"}
        })
        
        available_providers = config.get_configured_providers()["tts"]
        
        for provider_name in available_providers:
            processor = VoiceProcessor(tts_provider=provider_name, config=config)
            
            # Should initialize without error
            assert processor.tts_provider_name == provider_name
            assert processor.config is not None

    def test_provider_initialization_without_api_key(self):
        """Test that providers fail gracefully without API keys"""
        empty_config = DebabelizerConfig({})
        
        # Should raise ProviderError for unconfigured providers
        with pytest.raises(ProviderError):
            VoiceProcessor(tts_provider="elevenlabs", config=empty_config)

    def test_provider_auto_selection(self):
        """Test automatic provider selection"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "elevenlabs_key"},
            "openai": {"api_key": "openai_key"}
        })
        
        # Should auto-select first available provider
        processor = VoiceProcessor(config=config)
        
        # Should have configuration but not instantiated providers yet
        assert processor.config is not None
        available_providers = config.get_configured_providers()["tts"]
        assert len(available_providers) >= 1

    def test_invalid_provider_name(self):
        """Test handling of invalid provider names"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "test_key"}
        })
        
        with pytest.raises(ProviderError, match="not configured"):
            VoiceProcessor(tts_provider="nonexistent_provider", config=config)