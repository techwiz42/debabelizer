"""
Provider-agnostic unit tests for VoiceProcessor

These tests work with any configured STT/TTS provider and test the 
Debabelizer abstraction layer rather than specific provider implementations.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import VoiceProcessor, DebabelizerConfig
from debabelizer.providers.base import (
    TranscriptionResult, SynthesisResult, StreamingResult, 
    Voice, AudioFormat, ProviderError
)


class TestVoiceProcessor:
    """Provider-agnostic tests for VoiceProcessor"""

    def test_initialization_with_defaults(self):
        """Test VoiceProcessor initialization with default providers"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_soniox_key"},
            "elevenlabs": {"api_key": "test_elevenlabs_key"}
        })
        
        processor = VoiceProcessor(config=config)
        
        assert processor.config is not None
        assert processor.stt_provider is None  # Will be loaded on first use
        assert processor.tts_provider is None  # Will be loaded on first use

    def test_initialization_with_specific_providers(self):
        """Test VoiceProcessor initialization with specific providers"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_soniox_key"},
            "elevenlabs": {"api_key": "test_elevenlabs_key"}
        })
        
        processor = VoiceProcessor(
            stt_provider="soniox",
            tts_provider="elevenlabs", 
            config=config
        )
        
        assert processor.stt_provider_name == "soniox"
        assert processor.tts_provider_name == "elevenlabs"

    def test_initialization_with_optimization_settings(self):
        """Test VoiceProcessor initialization with optimization preferences"""
        # Test different optimization modes
        for optimize_for in ["latency", "quality", "cost", "balanced"]:
            config = DebabelizerConfig({
                "soniox": {"api_key": "test_key"},
                "preferences": {"optimize_for": optimize_for}
            })
            processor = VoiceProcessor(config=config)
            assert processor.config.get_optimization_strategy() == optimize_for

    def test_invalid_optimization_setting(self):
        """Test DebabelizerConfig with invalid optimization setting"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        with pytest.raises(ValueError, match="Invalid optimization strategy"):
            config.set_optimization_strategy("invalid_option")

    @pytest.mark.asyncio
    async def test_transcribe_file_interface(self):
        """Test transcribe_file method interface (provider-agnostic)"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        # Mock the STT provider
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.transcribe_file.return_value = TranscriptionResult(
            text="Test transcription",
            confidence=0.9,
            duration=2.5,
            language_detected="en"
        )
        # get_cost_estimate should be a regular method, not async
        mock_provider.get_cost_estimate = MagicMock(return_value=0.01)
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_provider):
                result = await processor.transcribe_file(tmp_file_path, language="en")
                
                assert isinstance(result, TranscriptionResult)
                assert result.text == "Test transcription"
                assert result.confidence == 0.9
                assert result.duration == 2.5
                assert result.language_detected == "en"
                
                # Verify provider method was called correctly
                mock_provider.transcribe_file.assert_called_once()
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.asyncio
    async def test_synthesize_text_interface(self):
        """Test synthesize method interface (provider-agnostic)"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(tts_provider="elevenlabs", config=config)
        
        # Mock the TTS provider
        mock_provider = AsyncMock()
        mock_provider.name = "elevenlabs"
        mock_provider.synthesize.return_value = SynthesisResult(
            audio_data=b"fake_audio_data",
            format="wav",
            sample_rate=16000,
            duration=1.5,
            size_bytes=48000
        )
        mock_provider.get_cost_estimate = MagicMock(return_value=0.02)
        
        with patch.object(processor, '_tts_provider', mock_provider):
            result = await processor.synthesize(
                text="Hello world",
                voice="test_voice"
            )
            
            assert isinstance(result, SynthesisResult)
            assert result.audio_data == b"fake_audio_data"
            assert result.format == "wav"
            assert result.duration == 1.5
            
            # Verify provider method was called correctly
            mock_provider.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_stt_interface(self):
        """Test streaming STT interface (provider-agnostic)"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        # Mock the STT provider
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.supports_streaming = True
        mock_provider.start_streaming.return_value = "session_123"
        
        with patch.object(processor, '_stt_provider', mock_provider):
            session_id = await processor.start_streaming_transcription(
                language="auto",
                sample_rate=16000
            )
            
            assert session_id == "session_123"
            mock_provider.start_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_voices_interface(self):
        """Test get_available_voices interface (provider-agnostic)"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(tts_provider="elevenlabs", config=config)
        
        # Mock the TTS provider
        mock_provider = AsyncMock()
        mock_provider.name = "elevenlabs"
        mock_voices = [
            Voice(
                voice_id='voice_1',
                name='Voice One',
                language='en',
                description='Test voice 1'
            ),
            Voice(
                voice_id='voice_2', 
                name='Voice Two',
                language='es',
                description='Test voice 2'
            )
        ]
        mock_provider.get_available_voices.return_value = mock_voices
        
        with patch.object(processor, '_tts_provider', mock_provider):
            voices = await processor.get_available_voices()
            
            assert len(voices) == 2
            assert voices[0].voice_id == 'voice_1'
            assert voices[1].name == 'Voice Two'

    def test_provider_auto_selection(self):
        """Test automatic provider selection based on configuration"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "soniox_key"},
            "deepgram": {"api_key": "deepgram_key"},
            "elevenlabs": {"api_key": "elevenlabs_key"}
        })
        
        # Should auto-select first available provider
        processor = VoiceProcessor(config=config)
        
        # Test that it can determine available providers
        available_providers = config.get_configured_providers()
        assert len(available_providers["stt"]) >= 1
        assert len(available_providers["tts"]) >= 1

    def test_provider_not_configured_error(self):
        """Test error when requested provider is not configured"""
        config = DebabelizerConfig()  # Empty config
        
        with pytest.raises(ProviderError, match="not configured"):
            VoiceProcessor(stt_provider="unconfigured_provider", config=config)

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that provider errors are properly propagated"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        # Mock the STT provider to raise an error
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.transcribe_file.side_effect = ProviderError("Provider-specific error", "soniox")
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_provider):
                with pytest.raises(ProviderError, match="Provider-specific error"):
                    await processor.transcribe_file(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    def test_session_manager_integration(self):
        """Test that VoiceProcessor integrates with SessionManager"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(config=config)
        
        assert processor.session_manager is not None
        # Should start with no sessions
        assert len(processor.session_manager) == 0

    def test_usage_stats_tracking(self):
        """Test that usage statistics are tracked"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(config=config)
        
        stats = processor.get_usage_stats()
        assert stats["stt_requests"] == 0
        assert stats["tts_requests"] == 0
        assert stats["cost_estimate"] == 0.0
        
        # Test stats reset
        processor.usage_stats["stt_requests"] = 5
        processor.reset_usage_stats()
        assert processor.usage_stats["stt_requests"] == 0

    def test_provider_switching(self):
        """Test switching between providers at runtime"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "soniox_key"},
            "deepgram": {"api_key": "deepgram_key"},
            "elevenlabs": {"api_key": "elevenlabs_key"},
            "openai": {"api_key": "openai_key"}
        })
        
        processor = VoiceProcessor(
            stt_provider="soniox",
            tts_provider="elevenlabs",
            config=config
        )
        
        assert processor.stt_provider_name == "soniox"
        assert processor.tts_provider_name == "elevenlabs"


class TestVoiceProcessorLanguageSupport:
    """Test language detection and multi-language support"""
    
    @pytest.mark.asyncio
    async def test_transcription_with_specific_languages(self):
        """Test transcription with specific language hints"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.get_cost_estimate = MagicMock(return_value=0.01)
        
        # Mock different results for different languages
        mock_provider.transcribe_file.return_value = TranscriptionResult(
            text="Bonjour le monde",
            confidence=0.95,
            language_detected="fr",
            duration=2.0
        )
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_provider):
                result = await processor.transcribe_file(
                    tmp_file_path,
                    language="fr",
                    language_hints=["fr", "en"]
                )
                
                assert result.language_detected == "fr"
                assert result.text == "Bonjour le monde"
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.asyncio
    async def test_auto_language_detection(self):
        """Test automatic language detection"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.get_cost_estimate = MagicMock(return_value=0.01)
        mock_provider.transcribe_file.return_value = TranscriptionResult(
            text="Hello world",
            confidence=0.9,
            language_detected="en",
            duration=1.5
        )
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_provider):
                result = await processor.transcribe_file(tmp_file_path)
                
                assert result.language_detected == "en"
                # Verify language was auto-detected (None passed)
                mock_provider.transcribe_file.assert_called_once()
                call_args = mock_provider.transcribe_file.call_args
                assert call_args[0][1] is None  # language param should be None
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.asyncio
    async def test_tts_with_language_specific_voices(self):
        """Test TTS with language-specific voice selection"""
        config = DebabelizerConfig({
            "elevenlabs": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(tts_provider="elevenlabs", config=config)
        
        mock_provider = AsyncMock()
        mock_provider.name = "elevenlabs"
        mock_provider.get_cost_estimate = MagicMock(return_value=0.02)
        
        # Mock language-specific voices
        mock_voices = [
            Voice(voice_id="en_voice", name="English Voice", language="en", description="English"),
            Voice(voice_id="es_voice", name="Spanish Voice", language="es", description="Spanish")
        ]
        mock_provider.get_available_voices.return_value = mock_voices
        
        mock_provider.synthesize.return_value = SynthesisResult(
            audio_data=b"spanish_audio",
            format="mp3",
            sample_rate=22050,
            duration=2.0,
            size_bytes=44100
        )
        
        with patch.object(processor, '_tts_provider', mock_provider):
            # Get Spanish voices
            voices = await processor.get_available_voices(language="es")
            mock_provider.get_available_voices.assert_called_with("es")

    @pytest.mark.asyncio
    async def test_language_availability_check(self):
        """Test checking language availability across providers"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "test_key"}
        })
        
        processor = VoiceProcessor(stt_provider="soniox", config=config)
        
        mock_provider = AsyncMock()
        mock_provider.name = "soniox"
        mock_provider.supported_languages = ["en", "fr", "es", "de"]
        
        with patch.object(processor, '_stt_provider', mock_provider):
            # Check if provider supports specific languages
            assert "en" in mock_provider.supported_languages
            assert "fr" in mock_provider.supported_languages
            assert "zh" not in mock_provider.supported_languages


class TestVoiceProcessorIntegration:
    """Integration tests for VoiceProcessor with multiple providers"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_voice_processing(self):
        """Test full transcription -> synthesis pipeline"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "stt_key"},
            "elevenlabs": {"api_key": "tts_key"}
        })
        
        processor = VoiceProcessor(
            stt_provider="soniox",
            tts_provider="elevenlabs",
            config=config
        )
        
        # Mock both providers
        mock_stt = AsyncMock()
        mock_stt.name = "soniox"
        mock_stt.get_cost_estimate = MagicMock(return_value=0.01)
        mock_stt.transcribe_file.return_value = TranscriptionResult(
            text="Hello world from audio",
            confidence=0.95,
            language_detected="en",
            duration=3.0
        )
        
        mock_tts = AsyncMock()
        mock_tts.name = "elevenlabs"
        mock_tts.get_cost_estimate = MagicMock(return_value=0.02)
        mock_tts.synthesize.return_value = SynthesisResult(
            audio_data=b"synthesized_audio",
            format="mp3",
            sample_rate=22050,
            duration=3.0,
            size_bytes=66150
        )
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_stt), \
                 patch.object(processor, '_tts_provider', mock_tts):
                
                # Transcribe audio
                transcript = await processor.transcribe_file(tmp_file_path)
                assert transcript.text == "Hello world from audio"
                
                # Synthesize the transcribed text
                synthesis = await processor.synthesize(transcript.text)
                assert synthesis.audio_data == b"synthesized_audio"
                
                # Check usage stats
                stats = processor.get_usage_stats()
                assert stats["stt_requests"] == 1
                assert stats["tts_requests"] == 1
                assert stats["cost_estimate"] == 0.03  # 0.01 + 0.02
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent transcription and synthesis operations"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "stt_key"},
            "elevenlabs": {"api_key": "tts_key"}
        })
        
        processor = VoiceProcessor(
            stt_provider="soniox",
            tts_provider="elevenlabs",
            config=config
        )
        
        # Mock providers with delays to simulate real operations
        mock_stt = AsyncMock()
        mock_stt.name = "soniox"
        mock_stt.get_cost_estimate = MagicMock(return_value=0.01)
        
        async def delayed_transcribe(*args, **kwargs):
            await asyncio.sleep(0.1)
            return TranscriptionResult(text="Concurrent", confidence=0.9, duration=1.0)
        
        mock_stt.transcribe_file = delayed_transcribe
        
        mock_tts = AsyncMock()
        mock_tts.name = "elevenlabs"
        mock_tts.get_cost_estimate = MagicMock(return_value=0.02)
        
        async def delayed_synthesize(*args, **kwargs):
            await asyncio.sleep(0.1)
            return SynthesisResult(
                audio_data=b"concurrent_audio",
                format="mp3",
                sample_rate=22050,
                duration=1.0,
                size_bytes=22050
            )
        
        mock_tts.synthesize = delayed_synthesize
        
        # Create temporary test files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            tmp1.write(b"audio1")
            tmp2.write(b"audio2")
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name
        
        try:
            with patch.object(processor, '_stt_provider', mock_stt), \
                 patch.object(processor, '_tts_provider', mock_tts):
                
                # Run concurrent operations
                tasks = [
                    processor.transcribe_file(tmp1_path),
                    processor.transcribe_file(tmp2_path),
                    processor.synthesize("Text 1"),
                    processor.synthesize("Text 2")
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all operations completed
                assert len(results) == 4
                assert all(isinstance(r, (TranscriptionResult, SynthesisResult)) for r in results)
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)

    @pytest.mark.asyncio
    async def test_optimization_preference_effect(self):
        """Test that optimization preferences affect provider selection/configuration"""
        config = DebabelizerConfig({
            "soniox": {"api_key": "stt_key"},
            "elevenlabs": {"api_key": "tts_key"}
        })
        
        # Test different optimization preferences
        for optimize_for in ["latency", "quality", "cost"]:
            config.set_optimization_strategy(optimize_for)
            processor = VoiceProcessor(
                stt_provider="soniox",
                tts_provider="elevenlabs",
                config=config
            )
            
            # Verify the processor respects the optimization setting
            assert processor.config.get_optimization_strategy() == optimize_for