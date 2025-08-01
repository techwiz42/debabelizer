"""
Tests for OpenAI TTS provider
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from io import BytesIO

from debabelizer.providers.tts.openai import OpenAITTSProvider
from debabelizer.providers.base.tts_provider import (
    SynthesisResult,
    Voice,
    AudioFormat
)


class TestOpenAITTSProvider:
    @pytest.fixture
    def provider(self):
        """Create an OpenAI TTS provider instance."""
        return OpenAITTSProvider(api_key="test_key")
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAITTSProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.default_model == "tts-1"
        assert provider.default_voice == "alloy"
    
    def test_initialization_with_config(self):
        """Test provider initialization with config."""
        provider = OpenAITTSProvider(
            api_key="config_key",
            tts_model="tts-1-hd",
            tts_voice="nova"
        )
        assert provider.api_key == "config_key"
        assert provider.default_model == "tts-1-hd"
        assert provider.default_voice == "nova"
    
    @pytest.mark.asyncio
    async def test_synthesize(self, provider):
        """Test text synthesis - connection failure test."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize(
                    text="Hello world",
                    voice_id="echo"
                )
            assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_with_default_voice(self, provider):
        """Test synthesis with default voice - connection failure test."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize(text="Test")
            assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_with_custom_parameters(self, provider):
        """Test synthesis with custom parameters - connection failure test."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize(
                    text="Test with custom params",
                    voice_id="fable",
                    model="tts-1-hd",
                    speed=0.75
                )
            assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_stream(self, provider):
        """Test streaming synthesis - connection failure test."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            try:
                async for chunk in provider.synthesize_streaming(
                    text="Hello streaming world",
                    voice_id="shimmer"
                ):
                    pass
                assert False, "Should have raised exception"
            except Exception as exc_info:
                assert "OpenAI library not installed" in str(exc_info)
    
    @pytest.mark.asyncio
    async def test_get_voices(self, provider):
        """Test getting available voices."""
        voices = await provider.get_available_voices()
        
        # Verify voices
        assert len(voices) == 6
        assert all(isinstance(v, Voice) for v in voices)
        
        # Check voice IDs
        voice_ids = [v.voice_id for v in voices]
        assert "alloy" in voice_ids
        assert "echo" in voice_ids
        assert "fable" in voice_ids
        assert "onyx" in voice_ids
        assert "nova" in voice_ids
        assert "shimmer" in voice_ids
        
        # Check voice properties
        alloy = next(v for v in voices if v.voice_id == "alloy")
        assert alloy.name == "Alloy"
        assert alloy.language == "en"
        assert alloy.gender == "neutral"
        assert "neutral" in alloy.description.lower()
        
        nova = next(v for v in voices if v.voice_id == "nova")
        assert nova.name == "Nova"
        assert nova.gender == "female"
        
        onyx = next(v for v in voices if v.voice_id == "onyx")
        assert onyx.name == "Onyx"
        assert onyx.gender == "male"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, provider):
        """Test error handling."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize("Test", voice_id="alloy")
            assert "OpenAI library not installed" in str(exc_info.value)
    
    def test_supported_languages(self, provider):
        """Test supported languages."""
        languages = provider.supported_languages
        assert isinstance(languages, list)
        # OpenAI TTS supports multiple languages
        assert len(languages) > 10
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "it" in languages
        assert "pt" in languages
        assert "pl" in languages
        assert "tr" in languages
    
    def test_supported_formats(self, provider):
        """Test supported audio formats."""
        formats = provider._supported_formats
        assert isinstance(formats, list)
        assert len(formats) == 6
        
        # Check format names
        assert "mp3" in formats
        assert "opus" in formats
        assert "aac" in formats
        assert "flac" in formats
        assert "wav" in formats
        assert "pcm" in formats
    
    @pytest.mark.asyncio
    async def test_synthesize_with_different_formats(self, provider):
        """Test synthesis with different audio formats - connection failure test."""
        # Test each supported format
        formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        
        for format_name in formats:
            # Mock the import to avoid requiring actual openai library
            with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
                with pytest.raises(Exception) as exc_info:
                    await provider.synthesize(
                        "Test", 
                        voice_id="alloy", 
                        response_format=format_name
                    )
                assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_with_speed_variations(self, provider):
        """Test synthesis with different speed settings - connection failure test."""
        # Test different speeds
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        
        for speed in speeds:
            # Mock the import to avoid requiring actual openai library
            with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
                with pytest.raises(Exception) as exc_info:
                    await provider.synthesize("Test", voice_id="alloy", speed=speed)
                assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_synthesize_long_text(self, provider):
        """Test synthesis with long text - should raise error for text too long."""
        # Text longer than 4096 characters (OpenAI's limit)
        long_text = "This is a test sentence. " * 200  # ~5000 characters
        
        with pytest.raises(Exception) as exc_info:
            await provider.synthesize(
                text=long_text,
                voice_id="nova"
            )
        
        # Should raise TextTooLongError before even trying to connect
        assert "Text too long" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, provider):
        """Test handling of empty text."""
        # Mock the import to avoid requiring actual openai library
        with patch('openai.AsyncOpenAI', side_effect=ImportError("OpenAI not installed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize("", voice_id="alloy")
            assert "OpenAI library not installed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_voice_handling(self, provider):
        """Test handling of invalid voice ID."""
        # Test with invalid voice - should raise VoiceNotFoundError before API call
        with pytest.raises(Exception) as exc_info:
            await provider.synthesize("Test", voice_id="invalid_voice")
        
        assert "Voice 'invalid_voice' not found" in str(exc_info.value)