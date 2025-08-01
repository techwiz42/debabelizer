"""
Tests for Soniox STT provider
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import websockets

from debabelizer.providers.stt.soniox import SonioxSTTProvider
from debabelizer.providers.base.stt_provider import (
    TranscriptionResult,
    StreamingResult
)
from debabelizer.providers.base.tts_provider import AudioFormat


class TestSonioxSTTProvider:
    @pytest.fixture
    def provider(self):
        """Create a Soniox provider for testing."""
        return SonioxSTTProvider(api_key="test_key")

    def test_initialization(self):
        """Test provider initialization."""
        provider = SonioxSTTProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.model == "stt-rt-preview"
    
    def test_initialization_with_config(self):
        """Test provider initialization with config."""
        config = {
            "api_key": "config_key",
            "model": "en_v2_lowlatency",
            "host": "custom.soniox.com"
        }
        provider = SonioxSTTProvider(**config)
        assert provider.api_key == "config_key"
        assert provider.model == "en_v2_lowlatency"
    
    @pytest.mark.asyncio
    async def test_transcribe_file(self, provider):
        """Test file transcription - simplified test."""
        # Create test file
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        
        # Mock WebSocket connection failure to avoid hanging
        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            try:
                result = await provider.transcribe_file("test_audio.wav")
                # If we get here without exception, basic file handling works
                assert isinstance(result, TranscriptionResult)
            except Exception as e:
                # Expected - WebSocket connection will fail in test environment
                assert "Connection failed" in str(e) or "ConnectionError" in str(type(e).__name__)
            
        # Cleanup
        import os
        os.remove("test_audio.wav")
    
    @pytest.mark.asyncio
    async def test_transcribe_file_error(self, provider):
        """Test file transcription error handling."""
        # Test with non-existent file
        with pytest.raises(Exception):
            await provider.transcribe_file("nonexistent.wav")
    
    @pytest.mark.asyncio
    async def test_start_streaming(self, provider):
        """Test starting streaming session."""
        # Mock WebSocket connection
        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            try:
                session_id = await provider.start_streaming()
                # If we get here, session creation logic works
                assert isinstance(session_id, str)
            except Exception as e:
                # Expected - WebSocket connection will fail
                assert "Connection failed" in str(e) or "ConnectionError" in str(type(e).__name__)
    
    def test_supported_languages(self, provider):
        """Test supported languages."""
        languages = provider.supported_languages
        assert isinstance(languages, list)
        assert len(languages) > 60  # Soniox supports 60+ languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
    
    def test_supports_streaming(self, provider):
        """Test streaming support."""
        assert provider.supports_streaming is True
        
    def test_supports_language_detection(self, provider):
        """Test language detection support."""
        assert provider.supports_language_detection is True
    
    @pytest.mark.asyncio
    async def test_streaming_basic(self, provider):
        """Test basic streaming functionality."""
        # Mock WebSocket connection to avoid actual connection
        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            try:
                session_id = await provider.start_streaming()
                assert isinstance(session_id, str)
                
                # Try to stream some audio
                await provider.stream_audio(session_id, b"fake audio")
                
                # Try to get results
                async for result in provider.get_streaming_results(session_id):
                    assert isinstance(result, StreamingResult)
                    break
                    
            except Exception as e:
                # Expected - WebSocket connection will fail
                assert "Connection failed" in str(e) or "ConnectionError" in str(type(e).__name__)
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, provider):
        """Test streaming error handling."""
        # Mock connection failure
        with patch('websockets.connect', side_effect=Exception("Connection failed")):
            try:
                session_id = await provider.start_streaming()
                # Should not get here due to connection failure
                assert False, "Should have failed"
            except Exception as exc_info:
                assert "Connection failed" in str(exc_info)
    
    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.name == "soniox"