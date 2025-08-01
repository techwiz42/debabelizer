"""
Tests for ElevenLabs TTS provider
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from io import BytesIO

from debabelizer.providers.tts.elevenlabs import ElevenLabsTTSProvider
from debabelizer.providers.base.tts_provider import (
    SynthesisResult,
    Voice,
    AudioFormat
)


class TestElevenLabsTTSProvider:
    @pytest.fixture
    def provider(self):
        """Create an ElevenLabs TTS provider instance."""
        return ElevenLabsTTSProvider(api_key="test_key")
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = ElevenLabsTTSProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.default_model == "eleven_turbo_v2_5"
        assert provider.base_url == "https://api.elevenlabs.io/v1"
    
    def test_initialization_with_config(self):
        """Test provider initialization with config."""
        provider = ElevenLabsTTSProvider(
            api_key="config_key",
            model="eleven_multilingual_v2",
            default_voice_id="custom_voice_id"
        )
        assert provider.api_key == "config_key"
        assert provider.default_model == "eleven_multilingual_v2"
        assert provider.default_voice_id == "custom_voice_id"
        assert provider.base_url == "https://api.elevenlabs.io/v1"
    
    @pytest.mark.asyncio
    async def test_synthesize_connection_failure(self, provider):
        """Test text synthesis with connection failure."""
        # Mock aiohttp ClientSession to raise connection error
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception) as exc_info:
                await provider.synthesize(
                    text="Hello world",
                    voice_id="21m00Tcm4TlvDq8ikWAM"
                )
            assert "Connection failed" in str(exc_info.value)
    
    @pytest.mark.asyncio 
    async def test_synthesize_empty_text(self, provider):
        """Test synthesis with empty text."""
        # Should handle empty text gracefully
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception):  # Should raise some kind of error
                await provider.synthesize(
                    text="",
                    voice_id="21m00Tcm4TlvDq8ikWAM"
                )
    
    @pytest.mark.asyncio
    async def test_get_available_voices_connection_error(self, provider):
        """Test getting available voices with connection error."""
        # Mock connection failure
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception) as exc_info:
                await provider.get_available_voices()
            assert "Connection failed" in str(exc_info.value)
    
    def test_supported_languages(self, provider):
        """Test supported languages."""
        languages = provider.supported_languages
        assert isinstance(languages, list)
        assert len(languages) > 30  # ElevenLabs supports many languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
    
    def test_supports_streaming(self, provider):
        """Test streaming support."""
        assert provider.supports_streaming is True
        
    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.name == "elevenlabs"
        
    def test_default_voice_id(self, provider):
        """Test default voice ID."""
        assert provider.default_voice_id == "21m00Tcm4TlvDq8ikWAM"  # Rachel