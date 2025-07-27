"""
Base Speech-to-Text Provider Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime


@dataclass
class WordTiming:
    """Word-level timing information for transcription"""
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription"""
    text: str
    confidence: float = 0.0
    language_detected: str = "unknown"
    duration: float = 0.0
    words: List[WordTiming] = None
    is_final: bool = True
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tokens: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.words is None:
            self.words = []


@dataclass
class StreamingResult:
    """Streaming transcription result with session info"""
    session_id: str
    is_final: bool
    text: str
    confidence: float
    timestamp: Optional[datetime] = None
    processing_time_ms: int = 0


class STTProvider(ABC):
    """
    Abstract base class for Speech-to-Text providers
    
    All STT providers must implement these methods to be compatible with Debabelizer.
    """
    
    def __init__(self, api_key: Optional[str] = None, **config):
        self.api_key = api_key
        self.config = config
        self.is_connected = False
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'soniox', 'deepgram')"""
        pass
        
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List of supported language codes (e.g., ['en', 'es', 'fr'])"""
        pass
        
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports real-time streaming"""
        pass
        
    @property
    @abstractmethod
    def supports_language_detection(self) -> bool:
        """Whether this provider can auto-detect languages"""
        pass
        
    @abstractmethod
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to audio file
            language: Expected language code (None for auto-detection)
            language_hints: List of possible languages to help detection
            **kwargs: Provider-specific options
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        pass
        
    @abstractmethod
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe raw audio data
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (wav, mp3, flac, mulaw, etc.)
            sample_rate: Sample rate in Hz
            language: Expected language code (None for auto-detection)
            language_hints: List of possible languages to help detection
            **kwargs: Provider-specific options
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        pass
        
    @abstractmethod
    async def start_streaming(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Start a streaming transcription session
        
        Args:
            audio_format: Audio format for streaming
            sample_rate: Sample rate in Hz
            language: Expected language code (None for auto-detection)
            language_hints: List of possible languages to help detection
            **kwargs: Provider-specific options
            
        Returns:
            Session ID for this streaming session
        """
        pass
        
    @abstractmethod
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """
        Send audio chunk to streaming session
        
        Args:
            session_id: Session ID from start_streaming()
            audio_chunk: Raw audio chunk bytes
        """
        pass
        
    @abstractmethod
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """
        Get streaming transcription results
        
        Args:
            session_id: Session ID from start_streaming()
            
        Yields:
            StreamingResult objects with transcription updates
        """
        pass
        
    @abstractmethod
    async def stop_streaming(self, session_id: str) -> None:
        """
        Stop streaming session and cleanup
        
        Args:
            session_id: Session ID to stop
        """
        pass
        
    async def test_connection(self) -> bool:
        """
        Test if provider is accessible and API key is valid
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test - try to transcribe a tiny bit of silence
            test_audio = b'\x00' * 1024  # 1KB of silence
            result = await self.transcribe_audio(
                test_audio, 
                audio_format="pcm",
                sample_rate=16000
            )
            return True
        except Exception:
            return False
            
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """
        Estimate cost for transcribing given duration
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            Estimated cost in USD (override in subclasses for accurate pricing)
        """
        # Default generic estimate - override in subclasses
        return duration_seconds * 0.001  # $0.001 per second default
        
    async def cleanup(self) -> None:
        """Cleanup resources (connections, sessions, etc.)"""
        self.is_connected = False