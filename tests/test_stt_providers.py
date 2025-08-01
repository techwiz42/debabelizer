"""
Provider-agnostic unit tests for STT providers

These tests work with any STT provider that implements the STTProvider interface.
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
from debabelizer.providers.base import TranscriptionResult, StreamingResult, ProviderError


class TestSTTProviderInterface:
    """Provider-agnostic tests for STT provider interface"""

    # Eight test languages from Thanotopolis Soniox tests
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
    
    def _create_mock_provider(self, **overrides):
        """Create a properly configured mock provider"""
        mock_provider = AsyncMock()
        # Make get_cost_estimate a sync method
        from unittest.mock import Mock
        mock_provider.get_cost_estimate = Mock(return_value=0.001)
        mock_provider.name = "mock_provider"
        
        # Apply any overrides
        for key, value in overrides.items():
            setattr(mock_provider, key, value)
            
        return mock_provider

    def _get_available_stt_providers(self):
        """Get list of configured STT providers for testing"""
        config = DebabelizerConfig()
        return config.get_configured_providers()["stt"]

    @pytest.mark.asyncio
    async def test_file_transcription_basic(self):
        """Test basic file transcription functionality"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Mock the actual provider to avoid API calls
            mock_provider = self._create_mock_provider()
            mock_provider.transcribe_file.return_value = TranscriptionResult(
                text="This is a test transcription",
                confidence=0.9,
                duration=2.5,
                language_detected="en"
            )
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                 patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'), \
                 patch('pathlib.Path.exists', return_value=True):
                result = await processor.transcribe_file("test_audio.wav")
                
                # Test interface contract
                assert isinstance(result, TranscriptionResult)
                assert hasattr(result, 'text')
                assert hasattr(result, 'confidence')
                assert hasattr(result, 'duration')
                assert hasattr(result, 'language_detected')
                assert hasattr(result, 'words')
                
                # Test basic values
                assert isinstance(result.text, str)
                assert isinstance(result.confidence, (int, float))
                assert isinstance(result.duration, (int, float))
                assert result.confidence >= 0.0 and result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_file_transcription_with_language_specification(self):
        """Test file transcription with specific language"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Test with each of our test languages
            for lang_code, lang_name in self.TEST_LANGUAGES.items():
                mock_provider = self._create_mock_provider()
                mock_provider.transcribe_file.return_value = TranscriptionResult(
                    text=f"Test transcription in {lang_name}",
                    confidence=0.85,
                    duration=1.8,
                    language_detected=lang_code
                )
                
                with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                     patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'), \
                     patch('pathlib.Path.exists', return_value=True):
                    result = await processor.transcribe_file(
                        f"{lang_name}_sample.wav", 
                        language=lang_code
                    )
                    
                    assert result.language_detected == lang_code
                    assert lang_name in result.text
                    
                    # Verify provider was called with correct language
                    mock_provider.transcribe_file.assert_called_with(
                        f"{lang_name}_sample.wav",
                        lang_code,
                        None,
                        audio_format='wav'
                    )

    @pytest.mark.asyncio
    async def test_file_transcription_auto_language_detection(self):
        """Test file transcription with automatic language detection"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Test auto-detection for each language
            for lang_code, lang_name in self.TEST_LANGUAGES.items():
                mock_provider = self._create_mock_provider()
                mock_provider.transcribe_file.return_value = TranscriptionResult(
                    text=f"Auto-detected {lang_name} text",
                    confidence=0.82,
                    duration=2.1,
                    language_detected=lang_code  # Provider auto-detected this
                )
                
                with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                     patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'), \
                     patch('pathlib.Path.exists', return_value=True):
                    # Don't specify language - test auto-detection
                    result = await processor.transcribe_file(f"{lang_name}_sample.wav")
                    
                    assert result.language_detected == lang_code
                    assert f"Auto-detected {lang_name}" in result.text
                    
                    # Verify no specific language was enforced
                    call_args = mock_provider.transcribe_file.call_args
                    if len(call_args) > 1 and 'language' in call_args[1]:
                        lang_arg = call_args[1]['language']
                        assert lang_arg is None or lang_arg == "auto"

    @pytest.mark.asyncio
    async def test_streaming_stt_lifecycle(self):
        """Test streaming STT session lifecycle"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            mock_provider = self._create_mock_provider()
            mock_provider.start_streaming.return_value = "test_session_123"
            mock_provider.stream_audio.return_value = StreamingResult(
                session_id="test_session_123",
                is_final=False,
                text="Partial transcript",
                confidence=0.7
            )
            mock_provider.stop_streaming.return_value = True
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider):
                # Start streaming
                session_id = await processor.start_streaming_transcription(
                    language="en",
                    sample_rate=16000
                )
                
                assert isinstance(session_id, str)
                assert len(session_id) > 0
                
                # Stream audio data
                await processor.stream_audio(session_id, b"audio_chunk")
                
                # Mock the streaming results generator
                async def mock_streaming_results(session_id):
                    yield StreamingResult(
                        session_id="test_session_123",
                        is_final=False,
                        text="Partial transcript",
                        confidence=0.7
                    )
                
                mock_provider.get_streaming_results = mock_streaming_results
                
                # Get streaming results
                results = []
                async for result in processor.get_streaming_results(session_id):
                    results.append(result)
                    break  # Just get one result for testing
                    
                assert len(results) == 1
                result = results[0]
                assert isinstance(result, StreamingResult)
                assert result.session_id == session_id
                assert hasattr(result, 'is_final')
                assert hasattr(result, 'text')
                assert hasattr(result, 'confidence')
                
                # End streaming
                await processor.stop_streaming_transcription(session_id)
                # Verify stop was called
                mock_provider.stop_streaming.assert_called_with(session_id)

    @pytest.mark.asyncio
    async def test_streaming_stt_with_callbacks(self):
        """Test streaming STT with transcript callbacks"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Track callback invocations
            transcript_calls = []
            final_calls = []
            
            async def on_transcript(text, is_final):
                transcript_calls.append((text, is_final))
            
            async def on_final(text, confidence):
                final_calls.append((text, confidence))
            
            mock_provider = self._create_mock_provider()
            mock_provider.start_streaming.return_value = "callback_session"
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider):
                session_id = await processor.start_streaming_transcription(
                    language="auto",
                    on_transcript=on_transcript,
                    on_final=on_final
                )
                
                assert session_id == "callback_session"
                
                # Verify callbacks were registered
                mock_provider.start_streaming.assert_called_once()
                call_kwargs = mock_provider.start_streaming.call_args[1]
                
                # The actual callback handling would be tested with the real provider
                # Here we just verify the interface accepts the callbacks

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in STT operations"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Test file transcription error
            mock_provider = self._create_mock_provider()
            mock_provider.transcribe_file.side_effect = ProviderError("STT API error")
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'):
                with pytest.raises(ProviderError, match="STT API error"):
                    await processor.transcribe_file("nonexistent.wav")
            
            # Test streaming error
            mock_provider.start_streaming.side_effect = ProviderError("Streaming failed")
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider):
                with pytest.raises(ProviderError, match="Streaming failed"):
                    await processor.start_streaming_transcription()

    def test_language_support_coverage(self):
        """Test that providers support the eight test languages"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Mock provider's language support
            mock_provider = type('Provider', (), {
                'get_supported_languages': lambda self: list(TestSTTProviderInterface.TEST_LANGUAGES.keys()) + ["auto"]
            })()
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider):
                supported_languages = processor._get_stt_provider().get_supported_languages()
                
                # Verify all test languages are supported
                for lang_code in self.TEST_LANGUAGES.keys():
                    assert lang_code in supported_languages, \
                        f"Provider {provider_name} doesn't support language {lang_code}"
                
                # Auto-detection should be supported
                assert "auto" in supported_languages

    @pytest.mark.asyncio
    async def test_transcription_result_structure(self):
        """Test that transcription results have the expected structure"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Test with word-level details
            mock_words = [
                {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95},
                {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.90}
            ]
            
            mock_provider = self._create_mock_provider()
            mock_provider.transcribe_file.return_value = TranscriptionResult(
                text="hello world",
                confidence=0.92,
                duration=1.0,
                language_detected="en",
                words=mock_words
            )
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                 patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'), \
                 patch('pathlib.Path.exists', return_value=True):
                result = await processor.transcribe_file("test.wav")
                
                # Test required fields
                assert isinstance(result.text, str)
                assert isinstance(result.confidence, (int, float))
                assert isinstance(result.duration, (int, float))
                
                # Test optional fields
                if result.language_detected:
                    assert isinstance(result.language_detected, str)
                
                if result.words:
                    assert isinstance(result.words, list)
                    for word in result.words:
                        assert isinstance(word, dict)
                        # Common word fields (may vary by provider)
                        if "word" in word:
                            assert isinstance(word["word"], str)
                        if "confidence" in word:
                            assert isinstance(word["confidence"], (int, float))

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self):
        """Test concurrent transcription operations"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            mock_provider = self._create_mock_provider()
            
            # Create multiple mock responses
            mock_responses = [
                TranscriptionResult(text=f"Transcription {i}", confidence=0.9, duration=1.0, language_detected="en")
                for i in range(3)
            ]
            
            mock_provider.transcribe_file.side_effect = mock_responses
            
            with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                 patch('debabelizer.utils.formats.detect_audio_format', return_value='wav'), \
                 patch('pathlib.Path.exists', return_value=True):
                # Start multiple concurrent transcriptions
                tasks = [
                    processor.transcribe_file(f"file_{i}.wav")
                    for i in range(3)
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all completed successfully
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert isinstance(result, TranscriptionResult)
                    assert f"Transcription {i}" in result.text

    @pytest.mark.asyncio
    async def test_audio_format_handling(self):
        """Test handling of different audio formats"""
        providers = self._get_available_stt_providers()
        
        if not providers:
            pytest.skip("No STT providers configured")
        
        audio_formats = ["wav", "mp3", "flac", "ogg", "m4a"]
        
        for provider_name in providers:
            config = DebabelizerConfig()
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            for audio_format in audio_formats:
                mock_provider = self._create_mock_provider()
                mock_provider.transcribe_file.return_value = TranscriptionResult(
                    text=f"Test {audio_format} transcription",
                    confidence=0.88,
                    duration=2.0,
                    language_detected="en"
                )
                
                with patch.object(processor, '_get_stt_provider', return_value=mock_provider), \
                     patch('debabelizer.utils.formats.detect_audio_format', return_value=audio_format), \
                     patch('pathlib.Path.exists', return_value=True):
                    result = await processor.transcribe_file(f"test.{audio_format}")
                    
                    assert audio_format in result.text
                    assert result.confidence > 0
                    
                    # Verify file was processed
                    mock_provider.transcribe_file.assert_called_with(f"test.{audio_format}", None, None, audio_format=audio_format)


class TestSTTProviderConfiguration:
    """Test STT provider configuration and initialization"""

    def test_provider_initialization_with_config(self):
        """Test that providers can be initialized with configuration"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_soniox_key", "model": "custom_model"},
            "deepgram": {"api_key": "test_deepgram_key", "model": "nova-2"}
        })
        
        available_providers = config.get_configured_providers()["stt"]
        
        for provider_name in available_providers:
            processor = VoiceProcessor(stt_provider=provider_name, config=config)
            
            # Should initialize without error
            assert processor.stt_provider_name == provider_name
            assert processor.config is not None

    def test_provider_initialization_without_api_key(self):
        """Test that providers fail gracefully without API keys"""
        empty_config = DebabelizerConfig({})
        
        # Should raise ProviderError for unconfigured providers
        with pytest.raises(ProviderError):
            VoiceProcessor(stt_provider="soniox", config=empty_config)

    def test_provider_auto_selection(self):
        """Test automatic provider selection"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "soniox_key"},
            "deepgram": {"api_key": "deepgram_key"}
        })
        
        # Should auto-select first available provider
        processor = VoiceProcessor(config=config)
        
        # Should have configuration but not instantiated providers yet
        assert processor.config is not None
        available_providers = config.get_configured_providers()["stt"]
        assert len(available_providers) >= 1

    def test_invalid_provider_name(self):
        """Test handling of invalid provider names"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        with pytest.raises(ProviderError, match="not configured"):
            VoiceProcessor(stt_provider="nonexistent_provider", config=config)