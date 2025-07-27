"""
Base Text-to-Speech Provider Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


@dataclass
class AudioFormat:
    """Audio format specification"""
    format: str
    sample_rate: int
    channels: int = 1
    bit_depth: int = 16


@dataclass
class Voice:
    """Voice configuration for TTS"""
    voice_id: str
    name: str
    language: str
    gender: Optional[str] = None
    description: Optional[str] = None
    age: Optional[str] = None
    accent: Optional[str] = None
    style: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis"""
    audio_data: bytes
    format: str
    sample_rate: int
    duration: float
    size_bytes: int
    voice_used: Optional[Voice] = None
    text_processed: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TTSProvider(ABC):
    """
    Abstract base class for Text-to-Speech providers
    
    All TTS providers must implement these methods to be compatible with Debabelizer.
    """
    
    def __init__(self, api_key: str, **config):
        self.api_key = api_key
        self.config = config
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'elevenlabs', 'deepgram', 'azure')"""
        pass
        
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List of supported language codes"""
        pass
        
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming synthesis"""
        pass
        
    @property
    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Whether this provider supports custom voice cloning"""
        pass
        
    @abstractmethod
    async def get_available_voices(
        self, 
        language: Optional[str] = None
    ) -> List[Voice]:
        """
        Get list of available voices
        
        Args:
            language: Filter by language code (None for all)
            
        Returns:
            List of available Voice objects
        """
        pass
        
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice object to use
            voice_id: Voice ID string (alternative to voice object)
            audio_format: Desired output format
            sample_rate: Sample rate in Hz
            **kwargs: Provider-specific options (speed, pitch, etc.)
            
        Returns:
            SynthesisResult with audio data and metadata
        """
        pass
        
    @abstractmethod
    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ):
        """
        Stream synthesis results in real-time
        
        Args:
            text: Text to synthesize
            voice: Voice object to use
            voice_id: Voice ID string (alternative to voice object)
            audio_format: Desired output format
            sample_rate: Sample rate in Hz
            **kwargs: Provider-specific options
            
        Yields:
            Audio chunks as they become available
        """
        pass
        
    async def synthesize_long_text(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        chunk_size: int = 1000,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize long text by breaking into chunks
        
        Args:
            text: Long text to synthesize
            voice: Voice to use
            voice_id: Voice ID string
            audio_format: Output format
            sample_rate: Sample rate in Hz
            chunk_size: Maximum characters per chunk
            **kwargs: Provider-specific options
            
        Returns:
            SynthesisResult with concatenated audio
        """
        # Default implementation - can be overridden for provider-specific chunking
        chunks = self._split_text_intelligently(text, chunk_size)
        audio_parts = []
        total_duration = 0.0
        
        for chunk in chunks:
            result = await self.synthesize(
                chunk, voice, voice_id, audio_format, sample_rate, **kwargs
            )
            audio_parts.append(result.audio_data)
            total_duration += result.duration
            
        # Concatenate audio (basic implementation)
        combined_audio = b''.join(audio_parts)
        
        return SynthesisResult(
            audio_data=combined_audio,
            format=audio_format.format if audio_format else "wav",
            sample_rate=sample_rate,
            duration=total_duration,
            size_bytes=len(combined_audio),
            voice_used=result.voice_used,  # Use last voice
            text_processed=text,
            metadata={"chunks": len(chunks)}
        )
        
    def _split_text_intelligently(self, text: str, max_chars: int) -> List[str]:
        """Split text at natural boundaries"""
        if len(text) <= max_chars:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = text.split('. ')
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chars:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
        
    async def test_connection(self) -> bool:
        """
        Test if provider is accessible and API key is valid
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            voices = await self.get_available_voices()
            return len(voices) > 0
        except Exception:
            return False
            
    def get_cost_estimate(self, text: str) -> float:
        """
        Estimate cost for synthesizing given text
        
        Args:
            text: Text to estimate cost for
            
        Returns:
            Estimated cost in USD (override in subclasses)
        """
        # Default generic estimate - override in subclasses
        char_count = len(text)
        return char_count * 0.00001  # $0.00001 per character default
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass