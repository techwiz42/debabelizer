"""
Tests for Deepgram STT provider
"""
import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import websockets

from debabelizer.providers.stt.deepgram import DeepgramSTTProvider
from debabelizer.providers.base.stt_provider import (
    TranscriptionResult,
    StreamingResult
)
from debabelizer.providers.base.tts_provider import AudioFormat


class TestDeepgramSTTProvider:
    @pytest.fixture
    def provider(self):
        """Create a Deepgram provider for testing."""
        with patch('debabelizer.providers.stt.deepgram.DeepgramClient'):
            return DeepgramSTTProvider(api_key="test_key")

    @pytest.fixture  
    def mock_client(self):
        """Create mock Deepgram client."""
        client = Mock()
        client.listen = Mock()
        client.listen.prerecorded = Mock()
        client.listen.prerecorded.v = Mock()
        client.listen.live = Mock()
        client.listen.live.v = Mock()
        return client
    
    def test_initialization(self):
        """Test provider initialization."""
        with patch('debabelizer.providers.stt.deepgram.DeepgramClient') as mock_client_class:
            provider = DeepgramSTTProvider(api_key="test_key")
            mock_client_class.assert_called_once_with("test_key")
            assert provider.model == "nova-2"
            assert provider.language == "en-US"
    
    def test_initialization_with_config(self):
        """Test provider initialization with config."""
        config = {
            "api_key": "config_key",
            "model": "nova-2-phonecall",
            "language": "es-ES"
        }
        with patch('debabelizer.providers.stt.deepgram.DeepgramClient') as mock_client_class:
            provider = DeepgramSTTProvider(config=config)
            mock_client_class.assert_called_once_with("config_key")
            assert provider.model == "nova-2-phonecall"
            assert provider.language == "es-ES"
    
    @pytest.mark.asyncio
    async def test_transcribe_file(self, provider, mock_client):
        """Test file transcription."""
        provider.client = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.results = Mock()
        mock_response.results.channels = [Mock()]
        mock_response.results.channels[0].alternatives = [Mock()]
        mock_response.results.channels[0].alternatives[0].transcript = "Test transcription"
        mock_response.results.channels[0].alternatives[0].confidence = 0.95
        mock_response.results.channels[0].alternatives[0].words = [
            Mock(word="Test", start=0.0, end=0.5, confidence=0.96),
            Mock(word="transcription", start=0.5, end=1.5, confidence=0.94)
        ]
        mock_response.metadata = Mock()
        mock_response.metadata.duration = 1.5
        mock_response.results.channels[0].detected_language = "en"
        
        # Setup mock - return the response directly without AsyncMock wrapper
        mock_prerecorded = Mock()
        mock_prerecorded.transcribe_file = Mock(return_value=mock_response)
        mock_client.listen.prerecorded.v.return_value = mock_prerecorded
        
        # Test transcription
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        
        result = await provider.transcribe_file("test_audio.wav")
        
        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"
        assert result.confidence == 0.95
        assert result.language_detected == "en"
        assert result.duration == 1.5
        assert len(result.words) == 2
        assert result.words[0].word == "Test"
        assert result.words[0].start_time == 0.0
        assert result.words[0].end_time == 0.5
        assert result.words[0].confidence == 0.96
        
        # Cleanup
        import os
        os.remove("test_audio.wav")
    
    @pytest.mark.asyncio
    async def test_transcribe_file_with_language_detection(self, provider, mock_client):
        """Test file transcription with language detection."""
        provider.client = mock_client
        
        # Mock response with detected language
        mock_response = Mock()
        mock_response.results = Mock()
        mock_response.results.channels = [Mock()]
        mock_response.results.channels[0].alternatives = [Mock()]
        mock_response.results.channels[0].alternatives[0].transcript = "Hola mundo"
        mock_response.results.channels[0].alternatives[0].confidence = 0.92
        mock_response.results.channels[0].detected_language = "es"
        mock_response.results.channels[0].alternatives[0].words = []
        mock_response.metadata = Mock()
        mock_response.metadata.duration = 1.0
        
        # Setup mock - return the response directly without AsyncMock wrapper
        mock_prerecorded = Mock()
        mock_prerecorded.transcribe_file = Mock(return_value=mock_response)
        mock_client.listen.prerecorded.v.return_value = mock_prerecorded
        
        # Test transcription
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        
        result = await provider.transcribe_file("test_audio.wav", language="auto")
        
        # Verify result
        assert result.text == "Hola mundo"
        assert result.language_detected == "es"
        
        # Verify API was called
        mock_client.listen.prerecorded.v.assert_called_once()
        
        # Cleanup
        import os
        os.remove("test_audio.wav")
    
    @pytest.mark.asyncio
    async def test_transcribe_stream(self, provider, mock_client):
        """Test streaming transcription - simplified test."""
        provider.client = mock_client
        
        # Mock the live connection
        mock_connection = Mock()
        mock_connection.start = AsyncMock()
        mock_connection.finish = AsyncMock()
        mock_connection.on = Mock()
        mock_connection.send = Mock()
        
        mock_client.listen.live.v.return_value = mock_connection
        
        # Create a simple audio stream
        async def audio_generator():
            yield b"fake audio chunk"
            # End stream quickly
            
        # Test that streaming can be started and stopped
        try:
            results = []
            async for result in provider.transcribe_stream(audio_generator()):
                results.append(result)
                break  # Exit after first iteration to avoid hanging
        except Exception as e:
            # Expected since we're not fully mocking the event system
            pass
            
        # Verify that the connection methods were called
        mock_connection.start.assert_called_once()
        mock_connection.on.assert_called()
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, provider, mock_client):
        """Test streaming error handling."""
        provider.client = mock_client
        
        # Mock connection failure
        mock_client.listen.live.v.side_effect = Exception("Connection failed")
        
        # Create audio stream
        async def audio_generator():
            yield b"fake audio"
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            async for _ in provider.transcribe_stream(audio_generator()):
                pass
        
        assert "Connection failed" in str(exc_info.value)
    
    def test_supported_languages(self, provider):
        """Test supported languages."""
        languages = provider.supported_languages
        assert isinstance(languages, list)
        assert len(languages) > 40  # Deepgram supports 40+ languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "zh" in languages
    
    def test_supported_formats(self, provider):
        """Test supported audio formats."""
        formats = provider.supported_formats
        assert isinstance(formats, list)
        assert len(formats) > 0
        
        # Check for common formats
        format_names = formats
        assert "wav" in format_names
        assert "mp3" in format_names
        assert "flac" in format_names
        assert "opus" in format_names
    
    @pytest.mark.asyncio
    async def test_transcribe_with_custom_options(self, provider, mock_client):
        """Test transcription with custom options."""
        provider.client = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.results = Mock()
        mock_response.results.channels = [Mock()]
        mock_response.results.channels[0].alternatives = [Mock()]
        mock_response.results.channels[0].alternatives[0].transcript = "Custom test"
        mock_response.results.channels[0].alternatives[0].confidence = 0.88
        mock_response.results.channels[0].alternatives[0].words = []
        mock_response.results.channels[0].detected_language = None
        mock_response.metadata = Mock()
        mock_response.metadata.duration = 1.0
        
        # Setup mock - return the response directly without AsyncMock wrapper
        mock_prerecorded = Mock()
        mock_prerecorded.transcribe_file = Mock(return_value=mock_response)
        mock_client.listen.prerecorded.v.return_value = mock_prerecorded
        
        # Test with custom options
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        
        # Test with custom options
        await provider.transcribe_file(
            "test_audio.wav",
            punctuate=True,
            diarize=True
        )
        
        # Verify API was called
        mock_client.listen.prerecorded.v.assert_called_once()
        
        # Cleanup
        import os
        os.remove("test_audio.wav")
    
    @pytest.mark.asyncio
    async def test_empty_transcription(self, provider, mock_client):
        """Test handling of empty transcription."""
        provider.client = mock_client
        
        # Mock response with empty transcript
        mock_response = Mock()
        mock_response.results = Mock()
        mock_response.results.channels = [Mock()]
        mock_response.results.channels[0].alternatives = [Mock()]
        mock_response.results.channels[0].alternatives[0].transcript = ""
        mock_response.results.channels[0].alternatives[0].confidence = 0.0
        mock_response.results.channels[0].alternatives[0].words = []
        mock_response.results.channels[0].detected_language = None
        mock_response.metadata = Mock()
        mock_response.metadata.duration = 0.5
        
        # Setup mock - return the response directly without AsyncMock wrapper
        mock_prerecorded = Mock()
        mock_prerecorded.transcribe_file = Mock(return_value=mock_response)
        mock_client.listen.prerecorded.v.return_value = mock_prerecorded
        
        # Test transcription
        with open("test_audio.wav", "wb") as f:
            f.write(b"fake audio data")
        
        result = await provider.transcribe_file("test_audio.wav")
        
        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.duration == 0.5
        assert len(result.words) == 0
        
        # Cleanup
        import os
        os.remove("test_audio.wav")