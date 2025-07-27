"""
Unit tests for base provider interfaces
"""

import pytest
from abc import ABC
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer.providers.base import (
    STTProvider, TTSProvider, TranscriptionResult, SynthesisResult,
    StreamingResult, Voice, AudioFormat, ProviderError
)


class TestBaseProviders:
    """Test cases for base provider classes"""

    def test_stt_provider_is_abstract(self):
        """Test that STTProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            STTProvider()

    def test_tts_provider_is_abstract(self):
        """Test that TTSProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TTSProvider()

    def test_transcription_result_creation(self):
        """Test TranscriptionResult dataclass creation"""
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            language_detected="en",
            duration=2.5
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.language_detected == "en"
        assert result.duration == 2.5
        assert result.words == []  # Default empty list

    def test_transcription_result_with_words(self):
        """Test TranscriptionResult with word-level details"""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.98},
            {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.92}
        ]
        
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            words=words,
            duration=1.0
        )
        
        assert len(result.words) == 2
        assert result.words[0]["word"] == "Hello"
        assert result.words[1]["confidence"] == 0.92

    def test_synthesis_result_creation(self):
        """Test SynthesisResult dataclass creation"""
        result = SynthesisResult(
            audio_data=b"fake_audio_data",
            format="wav",
            sample_rate=16000,
            duration=3.2,
            size_bytes=102400
        )
        
        assert result.audio_data == b"fake_audio_data"
        assert result.format == "wav"
        assert result.sample_rate == 16000
        assert result.duration == 3.2
        assert result.size_bytes == 102400

    def test_streaming_result_creation(self):
        """Test StreamingResult dataclass creation"""
        result = StreamingResult(
            session_id="session_123",
            is_final=True,
            text="Streaming text",
            confidence=0.88
        )
        
        assert result.session_id == "session_123"
        assert result.is_final is True
        assert result.text == "Streaming text"
        assert result.confidence == 0.88

    def test_voice_creation(self):
        """Test Voice dataclass creation"""
        voice = Voice(
            voice_id="voice_123",
            name="Rachel",
            language="en-US",
            gender="female",
            description="Professional female voice"
        )
        
        assert voice.voice_id == "voice_123"
        assert voice.name == "Rachel"
        assert voice.language == "en-US"
        assert voice.gender == "female"
        assert voice.description == "Professional female voice"

    def test_audio_format_creation(self):
        """Test AudioFormat dataclass creation"""
        audio_format = AudioFormat(
            format="wav",
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )
        
        assert audio_format.format == "wav"
        assert audio_format.sample_rate == 16000
        assert audio_format.channels == 1
        assert audio_format.bit_depth == 16

    def test_provider_error_exception(self):
        """Test ProviderError exception"""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Test error message")
        
        assert str(exc_info.value) == "Test error message"
        assert isinstance(exc_info.value, Exception)

    def test_provider_error_with_details(self):
        """Test ProviderError with additional details"""
        error = ProviderError("API failed", provider="soniox", status_code=401)
        
        assert str(error) == "API failed"
        # Test if we can add custom attributes
        assert hasattr(error, 'args')


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing abstract methods"""
    
    @property
    def name(self) -> str:
        return "mock_stt"
    
    @property
    def supported_languages(self) -> list:
        return ["en", "es", "fr"]
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_language_detection(self) -> bool:
        return True
    
    async def transcribe_file(self, file_path, language=None, **kwargs):
        return TranscriptionResult(
            text="Mock transcription",
            confidence=0.9,
            duration=1.0
        )
    
    async def transcribe_audio(self, audio_data, audio_format="wav", sample_rate=16000, language=None, **kwargs):
        return TranscriptionResult(
            text="Mock transcription from audio",
            confidence=0.85,
            duration=2.0
        )
    
    async def start_streaming(self, audio_format="wav", sample_rate=16000, language=None, **kwargs):
        return "mock_session_123"
    
    async def stream_audio(self, session_id, audio_chunk):
        pass
    
    async def get_streaming_results(self, session_id):
        yield StreamingResult(
            session_id=session_id,
            is_final=False,
            text="Partial mock",
            confidence=0.8
        )
    
    async def stop_streaming(self, session_id):
        pass


class MockTTSProvider(TTSProvider):
    """Mock TTS provider for testing abstract methods"""
    
    @property
    def name(self) -> str:
        return "mock_tts"
    
    @property
    def supported_languages(self) -> list:
        return ["en", "es", "fr"]
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_voice_cloning(self) -> bool:
        return False
    
    async def synthesize(self, text, voice=None, voice_id=None, audio_format=None, sample_rate=22050, **kwargs):
        return SynthesisResult(
            audio_data=b"mock_audio",
            format="wav",
            sample_rate=sample_rate,
            duration=len(text) * 0.1,  # Mock duration calculation
            size_bytes=len(text) * 100
        )
    
    async def synthesize_streaming(self, text, voice=None, voice_id=None, audio_format=None, sample_rate=22050, **kwargs):
        yield b"mock_streaming_audio_chunk"
    
    async def get_available_voices(self, language=None):
        voices = [
            Voice(
                voice_id="mock_voice_1",
                name="MockVoice",
                language="en",
                gender="neutral"
            )
        ]
        if language:
            return [v for v in voices if v.language.startswith(language)]
        return voices


class TestProviderImplementations:
    """Test that mock implementations work correctly"""

    @pytest.fixture
    def stt_provider(self):
        return MockSTTProvider(api_key="test_key")

    @pytest.fixture
    def tts_provider(self):
        return MockTTSProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_mock_stt_transcribe_file(self, stt_provider):
        """Test mock STT file transcription"""
        result = await stt_provider.transcribe_file("mock_file.wav")
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Mock transcription"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_mock_stt_streaming(self, stt_provider):
        """Test mock STT streaming"""
        session_id = await stt_provider.start_streaming()
        assert session_id == "mock_session_123"
        
        await stt_provider.stream_audio(session_id, b"audio_chunk")
        
        # Get streaming results
        results = []
        async for result in stt_provider.get_streaming_results(session_id):
            results.append(result)
            break  # Just get one result for testing
        
        assert len(results) == 1
        assert isinstance(results[0], StreamingResult)
        assert results[0].session_id == session_id
        
        await stt_provider.stop_streaming(session_id)

    def test_mock_stt_supported_languages(self, stt_provider):
        """Test mock STT supported languages"""
        languages = stt_provider.supported_languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages

    @pytest.mark.asyncio
    async def test_mock_tts_synthesis(self, tts_provider):
        """Test mock TTS synthesis"""
        result = await tts_provider.synthesize("Hello world")
        
        assert isinstance(result, SynthesisResult)
        assert result.audio_data == b"mock_audio"
        assert result.format == "wav"
        assert result.duration == 1.1  # 11 chars * 0.1

    @pytest.mark.asyncio
    async def test_mock_tts_voices(self, tts_provider):
        """Test mock TTS available voices"""
        voices = await tts_provider.get_available_voices()
        
        assert len(voices) == 1
        assert isinstance(voices[0], Voice)
        assert voices[0].voice_id == "mock_voice_1"

    @pytest.mark.asyncio
    async def test_mock_tts_streaming(self, tts_provider):
        """Test mock TTS streaming"""
        # Test streaming synthesis
        audio_chunks = []
        async for chunk in tts_provider.synthesize_streaming("Hello"):
            audio_chunks.append(chunk)
            break  # Just get one chunk for testing
        
        assert len(audio_chunks) == 1
        assert audio_chunks[0] == b"mock_streaming_audio_chunk"