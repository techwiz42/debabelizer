"""
ElevenLabs Text-to-Speech Provider for Debabelizer

Features:
- High-quality voice synthesis
- 1000+ voices available
- Voice cloning support
- Streaming synthesis
- Intelligent text chunking for long content
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import aiohttp
import json

from ..base import TTSProvider, SynthesisResult, Voice, AudioFormat
from ..base.exceptions import (
    ProviderError, AuthenticationError, VoiceNotFoundError,
    TextTooLongError, RateLimitError
)

logger = logging.getLogger(__name__)


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs Text-to-Speech Provider"""
    
    def __init__(self, api_key: str, **config):
        super().__init__(api_key, **config)
        self.base_url = "https://api.elevenlabs.io/v1"
        self.default_voice_id = config.get("default_voice_id", "21m00Tcm4TlvDq8ikWAM")  # Rachel
        self.default_model = config.get("model", "eleven_turbo_v2_5")
        self.optimize_streaming_latency = config.get("optimize_streaming_latency", 1)
        
        # Supported languages (ElevenLabs supports many via different voices)
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
            "cs", "ar", "zh", "ja", "hi", "ko", "fi", "sk", "uk", "el",
            "bg", "hr", "ro", "da", "hu", "ta", "no", "he", "th", "ur",
            "id", "ms", "vi", "sw", "fil", "is", "lv", "lt", "sl", "et"
        ]
        
    @property
    def name(self) -> str:
        return "elevenlabs"
        
    @property
    def supported_languages(self) -> List[str]:
        return self._supported_languages.copy()
        
    @property
    def supports_streaming(self) -> bool:
        return True
        
    @property
    def supports_voice_cloning(self) -> bool:
        return True
        
    def _get_headers(self, content_type: str = "application/json") -> Dict[str, str]:
        """Get request headers"""
        return {
            "Accept": "audio/mpeg",
            "Content-Type": content_type,
            "xi-api-key": self.api_key
        }
        
    async def get_available_voices(
        self, 
        language: Optional[str] = None
    ) -> List[Voice]:
        """Get list of available voices"""
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{self.base_url}/voices", 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key", "elevenlabs")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"API error: {response.status} - {error_text}", "elevenlabs")
                        
                    data = await response.json()
                    voices = []
                    
                    for voice_data in data.get("voices", []):
                        # Extract voice metadata
                        labels = voice_data.get("labels", {})
                        
                        voice = Voice(
                            voice_id=voice_data["voice_id"],
                            name=voice_data["name"],
                            language=labels.get("language", "en"),  # Default to English
                            gender=labels.get("gender"),
                            age=labels.get("age"),
                            accent=labels.get("accent"),
                            style=labels.get("use_case"),
                            metadata={
                                "category": voice_data.get("category"),
                                "description": labels.get("description"),
                                "preview_url": voice_data.get("preview_url"),
                                "available_for_tiers": voice_data.get("available_for_tiers", [])
                            }
                        )
                        
                        # Filter by language if specified
                        if language is None or voice.language == language:
                            voices.append(voice)
                            
                    return voices
                    
        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError(f"Failed to get voices: {e}", "elevenlabs")
            
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
        
        # Determine voice ID
        actual_voice_id = voice_id or (voice.voice_id if voice else self.default_voice_id)
        
        # Validate text length (ElevenLabs has limits)
        if len(text) > 5000:  # ElevenLabs limit
            raise TextTooLongError(
                f"Text too long ({len(text)} chars). Use synthesize_long_text() for longer content.",
                "elevenlabs",
                max_length=5000
            )
            
        # Prepare request
        payload = {
            "text": text,
            "model_id": kwargs.get("model", self.default_model),
            "voice_settings": {
                "stability": kwargs.get("stability", 0.75),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True)
            }
        }
        
        # Add optimize_streaming_latency if specified
        if self.optimize_streaming_latency:
            payload["optimize_streaming_latency"] = self.optimize_streaming_latency
            
        try:
            timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for synthesis
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}/text-to-speech/{actual_voice_id}"
                
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key", "elevenlabs")
                    elif response.status == 404:
                        raise VoiceNotFoundError(f"Voice {actual_voice_id} not found", "elevenlabs")
                    elif response.status == 429:
                        raise RateLimitError("Rate limit exceeded", "elevenlabs")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"Synthesis failed: {response.status} - {error_text}", "elevenlabs")
                        
                    audio_data = await response.read()
                    
                    # Get voice info for result
                    used_voice = voice or Voice(
                        voice_id=actual_voice_id,
                        name="Unknown",
                        language="en"
                    )
                    
                    # Estimate duration (rough approximation)
                    # Average speaking rate: ~150 words per minute, ~5 chars per word
                    words = len(text.split())
                    duration_seconds = (words / 150) * 60
                    
                    return SynthesisResult(
                        audio_data=audio_data,
                        format="mp3",  # ElevenLabs returns MP3
                        sample_rate=sample_rate,
                        duration=duration_seconds,
                        size_bytes=len(audio_data),
                        voice_used=used_voice,
                        text_processed=text,
                        metadata={
                            "model": payload["model_id"],
                            "voice_settings": payload["voice_settings"],
                            "character_count": len(text),
                            "word_count": words
                        }
                    )
                    
        except Exception as e:
            if isinstance(e, (AuthenticationError, VoiceNotFoundError, RateLimitError, ProviderError)):
                raise
            raise ProviderError(f"Synthesis failed: {e}", "elevenlabs")
            
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
        
        actual_voice_id = voice_id or (voice.voice_id if voice else self.default_voice_id)
        
        payload = {
            "text": text,
            "model_id": kwargs.get("model", self.default_model),
            "voice_settings": {
                "stability": kwargs.get("stability", 0.75),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True)
            },
            "optimize_streaming_latency": self.optimize_streaming_latency
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.base_url}/text-to-speech/{actual_voice_id}/stream"
                
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"Streaming synthesis failed: {response.status} - {error_text}", "elevenlabs")
                        
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            yield chunk
                            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Streaming synthesis failed: {e}", "elevenlabs")
            
    async def synthesize_long_text(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        chunk_size: int = 4000,  # Conservative chunk size for ElevenLabs
        **kwargs
    ) -> SynthesisResult:
        """Synthesize long text by breaking into chunks"""
        
        if len(text) <= 5000:
            # Short enough for single request
            return await self.synthesize(text, voice, voice_id, audio_format, sample_rate, **kwargs)
            
        # Split text intelligently
        chunks = self._split_text_intelligently(text, chunk_size)
        audio_parts = []
        total_duration = 0.0
        total_chars = len(text)
        
        logger.info(f"Synthesizing long text ({total_chars} chars) in {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Synthesizing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
            
            result = await self.synthesize(
                chunk, voice, voice_id, audio_format, sample_rate, **kwargs
            )
            audio_parts.append(result.audio_data)
            total_duration += result.duration
            
            # Small delay to respect rate limits
            if i < len(chunks) - 1:
                await asyncio.sleep(0.1)
                
        # Combine audio data (simple concatenation for MP3)
        combined_audio = b''.join(audio_parts)
        
        return SynthesisResult(
            audio_data=combined_audio,
            format=audio_format.format if audio_format else "mp3",
            sample_rate=sample_rate,
            duration=total_duration,
            size_bytes=len(combined_audio),
            voice_used=result.voice_used,  # Use last voice
            text_processed=text,
            metadata={
                "chunks": len(chunks),
                "total_characters": total_chars,
                "synthesis_method": "chunked"
            }
        )
        
    def get_cost_estimate(self, text: str) -> float:
        """Estimate cost for ElevenLabs synthesis"""
        # ElevenLabs pricing: ~$0.30 per 1K characters for standard voices
        char_count = len(text)
        return (char_count / 1000) * 0.30
        
    async def test_connection(self) -> bool:
        """Test ElevenLabs API connection"""
        try:
            voices = await self.get_available_voices()
            return len(voices) > 0
        except Exception:
            return False
            
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass