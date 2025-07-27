"""
OpenAI Text-to-Speech Provider for Debabelizer

Features:
- High-quality voice synthesis using TTS-1 and TTS-1-HD models
- 6 built-in voices (alloy, echo, fable, onyx, nova, shimmer)
- Multiple audio formats (MP3, OPUS, AAC, FLAC, WAV, PCM)
- Real-time streaming support
- Optimized for both speed (TTS-1) and quality (TTS-1-HD)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import tempfile

from ..base import TTSProvider, SynthesisResult, Voice, AudioFormat
from ..base.exceptions import (
    ProviderError, AuthenticationError, VoiceNotFoundError,
    TextTooLongError, RateLimitError, UnsupportedFormatError
)

logger = logging.getLogger(__name__)


class OpenAITTSProvider(TTSProvider):
    """OpenAI Text-to-Speech Provider"""
    
    def __init__(self, api_key: str, **config):
        super().__init__(api_key, **config)
        self.default_model = config.get("tts_model", "tts-1")
        self.default_voice = config.get("tts_voice", "alloy")
        
        # Available voices with metadata
        self._available_voices = [
            Voice(
                voice_id="alloy",
                name="Alloy",
                language="en",
                gender="neutral",
                description="Neutral, balanced voice suitable for most content"
            ),
            Voice(
                voice_id="echo",
                name="Echo",
                language="en", 
                gender="male",
                description="Male voice with clear articulation"
            ),
            Voice(
                voice_id="fable",
                name="Fable",
                language="en",
                gender="male", 
                description="Male voice with storytelling quality"
            ),
            Voice(
                voice_id="onyx",
                name="Onyx",
                language="en",
                gender="male",
                description="Deep male voice with authority"
            ),
            Voice(
                voice_id="nova",
                name="Nova",
                language="en",
                gender="female",
                description="Female voice with clarity and warmth"
            ),
            Voice(
                voice_id="shimmer",
                name="Shimmer", 
                language="en",
                gender="female",
                description="Female voice with bright, energetic tone"
            )
        ]
        
        # Supported audio formats
        self._supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        
    @property
    def name(self) -> str:
        return "openai"
        
    @property
    def supported_languages(self) -> List[str]:
        # OpenAI TTS supports many languages but voices are primarily English
        # It can handle multilingual content with the same voices
        return [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
            "ar", "hi", "th", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ]
        
    @property 
    def supports_streaming(self) -> bool:
        return True
        
    @property
    def supports_voice_cloning(self) -> bool:
        return False
        
    def _validate_text_length(self, text: str) -> None:
        """Validate text length (OpenAI has 4096 character limit)"""
        if len(text) > 4096:
            raise TextTooLongError(
                f"Text too long ({len(text)} characters). OpenAI TTS supports max 4096 characters.",
                "openai",
                max_length=4096
            )
            
    def _get_voice_by_id(self, voice_id: str) -> Optional[Voice]:
        """Get voice object by ID"""
        for voice in self._available_voices:
            if voice.voice_id == voice_id:
                return voice
        return None
        
    async def get_available_voices(
        self, 
        language: Optional[str] = None
    ) -> List[Voice]:
        """Get list of available voices"""
        voices = self._available_voices.copy()
        
        # Filter by language if specified
        if language:
            # OpenAI voices work with multiple languages, so we return all
            # since they can handle multilingual content
            return voices
            
        return voices
        
    async def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> SynthesisResult:
        """Synthesize speech from text"""
        
        # Validate text length
        self._validate_text_length(text)
        
        # Determine voice
        actual_voice_id = voice_id or (voice.voice_id if voice else self.default_voice)
        if actual_voice_id not in [v.voice_id for v in self._available_voices]:
            raise VoiceNotFoundError(f"Voice '{actual_voice_id}' not found", "openai")
            
        # Determine audio format
        output_format = "mp3"  # Default
        if audio_format:
            output_format = audio_format.format
        elif "response_format" in kwargs:
            output_format = kwargs["response_format"]
            
        if output_format not in self._supported_formats:
            raise UnsupportedFormatError(f"Unsupported audio format: {output_format}", "openai")
            
        # Determine model
        model = kwargs.get("model", self.default_model)
        if model not in ["tts-1", "tts-1-hd"]:
            model = self.default_model
            
        try:
            # Dynamic import to avoid requiring openai if not used
            from openai import AsyncOpenAI
            
            # Initialize client
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Make TTS request
            response = await client.audio.speech.create(
                model=model,
                voice=actual_voice_id,
                input=text,
                response_format=output_format,
                speed=kwargs.get("speed", 1.0)
            )
            
            # Get audio data
            audio_data = response.content
            
            # Get voice info
            used_voice = self._get_voice_by_id(actual_voice_id) or Voice(
                voice_id=actual_voice_id,
                name=actual_voice_id.title(),
                language="en"
            )
            
            # Estimate duration (rough approximation based on text length and speed)
            # Average speaking rate: ~150 words per minute
            words = len(text.split())
            speed_factor = kwargs.get("speed", 1.0)
            duration_seconds = (words / 150) * 60 / speed_factor
            
            return SynthesisResult(
                audio_data=audio_data,
                format=output_format,
                sample_rate=sample_rate,
                duration=duration_seconds,
                size_bytes=len(audio_data),
                voice_used=used_voice,
                text_processed=text,
                metadata={
                    "model": model,
                    "voice": actual_voice_id,
                    "speed": kwargs.get("speed", 1.0),
                    "character_count": len(text),
                    "word_count": words,
                    "response_format": output_format
                }
            )
            
        except ImportError:
            raise ProviderError(
                "OpenAI library not installed. Install with: pip install openai>=1.0.0",
                "openai"
            )
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "unauthorized" in error_str.lower():
                raise AuthenticationError("Invalid OpenAI API key", "openai")
            elif "429" in error_str or "rate_limit" in error_str.lower():
                raise RateLimitError("OpenAI rate limit exceeded", "openai")
            elif "quota" in error_str.lower() or "billing" in error_str.lower():
                raise ProviderError("OpenAI quota exceeded or billing issue", "openai")
            else:
                raise ProviderError(f"OpenAI TTS synthesis failed: {e}", "openai")
                
    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis results in real-time"""
        
        # Validate text length
        self._validate_text_length(text)
        
        # Determine voice
        actual_voice_id = voice_id or (voice.voice_id if voice else self.default_voice)
        if actual_voice_id not in [v.voice_id for v in self._available_voices]:
            raise VoiceNotFoundError(f"Voice '{actual_voice_id}' not found", "openai")
            
        # Determine audio format
        output_format = "mp3"  # Default for streaming
        if audio_format:
            output_format = audio_format.format
        elif "response_format" in kwargs:
            output_format = kwargs["response_format"]
            
        # Determine model
        model = kwargs.get("model", self.default_model)
        if model not in ["tts-1", "tts-1-hd"]:
            model = self.default_model
            
        try:
            # Dynamic import to avoid requiring openai if not used
            from openai import AsyncOpenAI
            
            # Initialize client
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Make streaming TTS request
            response = await client.audio.speech.create(
                model=model,
                voice=actual_voice_id,
                input=text,
                response_format=output_format,
                speed=kwargs.get("speed", 1.0)
            )
            
            # Stream the response in chunks
            chunk_size = kwargs.get("chunk_size", 1024)
            
            # For OpenAI, we get the full response and then chunk it
            # This is because OpenAI doesn't provide true streaming TTS yet
            audio_data = response.content
            
            # Yield chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
                
        except ImportError:
            raise ProviderError(
                "OpenAI library not installed. Install with: pip install openai>=1.0.0",
                "openai"
            )
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "unauthorized" in error_str.lower():
                raise AuthenticationError("Invalid OpenAI API key", "openai")
            elif "429" in error_str or "rate_limit" in error_str.lower():
                raise RateLimitError("OpenAI rate limit exceeded", "openai")
            else:
                raise ProviderError(f"OpenAI TTS streaming failed: {e}", "openai")
                
    def get_cost_estimate(self, text: str) -> float:
        """Estimate cost for OpenAI TTS synthesis"""
        # OpenAI TTS pricing: $0.015 per 1K characters for TTS-1
        # $0.030 per 1K characters for TTS-1-HD
        char_count = len(text)
        
        if self.default_model == "tts-1-hd":
            return (char_count / 1000) * 0.030
        else:
            return (char_count / 1000) * 0.015
            
    async def test_connection(self) -> bool:
        """Test OpenAI TTS API connection"""
        try:
            # Test with a short text
            result = await self.synthesize("Test")
            return len(result.audio_data) > 0
        except Exception:
            return False
            
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass  # No persistent connections to clean up