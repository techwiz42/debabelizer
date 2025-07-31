"""
OpenAI Whisper API Speech-to-Text Provider

Implements STT using OpenAI's cloud-based Whisper API with support for:
- High-quality transcription via OpenAI API
- 99+ languages supported  
- Word-level timestamps (when available)
- Automatic language detection
- No local model downloads required
- Pay-per-use pricing
"""

import asyncio
import logging
import tempfile
import os
import wave
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from ..base import STTProvider, TranscriptionResult, StreamingResult, WordTiming
from ..base.exceptions import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class OpenAIWhisperSTTProvider(STTProvider):
    """
    OpenAI Whisper API Speech-to-Text Provider
    
    Cloud-based speech recognition using OpenAI's Whisper API.
    No local model downloads required.
    """
    
    name = "openai_whisper"
    
    # Whisper API supports many languages (using ISO 639-1 codes)
    LANGUAGE_MAP = {
        "en": "en",
        "es": "es", 
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "ja": "ja",
        "ko": "ko",
        "zh": "zh",
        "ar": "ar",
        "hi": "hi",
        "nl": "nl",
        "pl": "pl",
        "tr": "tr",
        "sv": "sv",
        "da": "da",
        "no": "no",
        "fi": "fi",
        "cs": "cs",
        "hu": "hu",
        "el": "el",
        "he": "he",
        "th": "th",
        "vi": "vi",
        "id": "id",
        "ms": "ms",
        "ro": "ro",
        "uk": "uk",
        "bg": "bg",
        "hr": "hr",
        "sk": "sk",
        "sl": "sl",
        "et": "et",
        "lv": "lv",
        "lt": "lt",
        "ca": "ca",
        "eu": "eu",
        "gl": "gl",
        "af": "af",
        "sq": "sq",
        "am": "am",
        "hy": "hy",
        "az": "az",
        "be": "be",
        "bn": "bn",
        "bs": "bs",
        "my": "my",
        "cy": "cy",
        "eo": "eo",
        "fa": "fa",
        "fo": "fo",
        "gu": "gu",
        "ha": "ha",
        "is": "is",
        "jw": "jw",
        "ka": "ka",
        "kk": "kk",
        "km": "km",
        "kn": "kn",
        "la": "la",
        "lo": "lo",
        "lb": "lb",
        "mk": "mk",
        "mg": "mg",
        "ml": "ml",
        "mt": "mt",
        "mi": "mi",
        "mr": "mr",
        "mn": "mn",
        "ne": "ne",
        "nn": "nn",
        "oc": "oc",
        "ps": "ps",
        "sa": "sa",
        "sd": "sd",
        "si": "si",
        "so": "so",
        "su": "su",
        "sw": "sw",
        "tl": "tl",
        "tg": "tg",
        "ta": "ta",
        "tt": "tt",
        "te": "te",
        "tk": "tk",
        "ur": "ur",
        "uz": "uz",
        "yi": "yi",
        "yo": "yo",
        "zu": "zu"
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        temperature: float = 0.0,
        response_format: str = "json",
        **kwargs
    ):
        """
        Initialize OpenAI Whisper API STT Provider
        
        Args:
            api_key: OpenAI API key
            model: Model to use ("whisper-1")
            temperature: Temperature for sampling (0.0 = deterministic)
            response_format: Response format ("json", "text", "srt", "verbose_json", "vtt")
        """
        super().__init__()
        
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI library not available. Install with: pip install openai",
                self.name
            )
        
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key is required",
                self.name
            )
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        
        # Initialize OpenAI client
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}", self.name)
        
        logger.info(f"Initialized OpenAI Whisper API with model: {model}")
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supports_streaming(self) -> bool:
        """OpenAI Whisper API doesn't support real-time streaming"""
        return False
    
    @property
    def supports_language_detection(self) -> bool:
        """OpenAI Whisper API supports automatic language detection"""
        return True
    
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using OpenAI Whisper API
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            language_hints: Not used by OpenAI API
            **kwargs: Additional OpenAI API options
            
        Returns:
            TranscriptionResult with transcription
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Map language code
            whisper_language = None
            if language:
                whisper_language = self._map_language_to_whisper(language)
            
            # Prepare API parameters
            api_params = {
                "model": self.model,
                "response_format": self.response_format,
                "temperature": kwargs.get("temperature", self.temperature)
            }
            
            # Add language if specified
            if whisper_language:
                api_params["language"] = whisper_language
            
            # Add any additional parameters
            for key in ["prompt"]:  # OpenAI API specific parameters
                if key in kwargs:
                    api_params[key] = kwargs[key]
            
            logger.info(f"Transcribing file with OpenAI Whisper API: {file_path}")
            start_time = datetime.now()
            
            # Perform transcription
            with open(file_path, 'rb') as audio_file:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.audio.transcriptions.create(
                        file=audio_file,
                        **api_params
                    )
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Process response based on format
            if self.response_format == "json":
                text = response.text
                confidence = 0.9  # OpenAI doesn't provide confidence scores
                
                # Get usage information if available
                usage_seconds = 0
                if hasattr(response, 'usage') and response.usage:
                    usage_seconds = getattr(response.usage, 'seconds', 0)
                
                return TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language_detected=language or "en",  # API doesn't return detected language in basic mode
                    duration=duration,
                    words=[],  # Basic mode doesn't include word timestamps
                    metadata={
                        "model": self.model,
                        "api_usage_seconds": usage_seconds,
                        "response_format": self.response_format,
                        "processing_time": duration
                    }
                )
            else:
                # For other formats (text, srt, vtt), return as text
                return TranscriptionResult(
                    text=str(response),
                    confidence=0.9,
                    language_detected=language or "en",
                    duration=duration,
                    words=[],
                    metadata={
                        "model": self.model,
                        "response_format": self.response_format,
                        "processing_time": duration
                    }
                )
            
        except Exception as e:
            logger.error(f"OpenAI Whisper API transcription failed: {e}")
            raise ProviderError(f"OpenAI Whisper API transcription failed: {e}", self.name)
    
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
        Transcribe raw audio data by saving to temporary file
        
        OpenAI API works with files, so we save audio data to a temp file first.
        """
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Write WAV file with proper header
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                
                temp_path = tmp_file.name
            
            try:
                # Transcribe the temporary file
                result = await self.transcribe_file(
                    temp_path,
                    language=language,
                    language_hints=language_hints,
                    **kwargs
                )
                
                # Update metadata to indicate it was from raw audio
                result.metadata["source"] = "raw_audio"
                result.metadata["original_format"] = audio_format
                result.metadata["sample_rate"] = sample_rate
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise ProviderError(f"Audio transcription failed: {e}", self.name)
    
    async def start_streaming(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """OpenAI Whisper API doesn't support real-time streaming"""
        raise ProviderError(
            "OpenAI Whisper API doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
    
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """OpenAI Whisper API doesn't support real-time streaming"""
        raise ProviderError(
            "OpenAI Whisper API doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
    
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """OpenAI Whisper API doesn't support real-time streaming"""
        raise ProviderError(
            "OpenAI Whisper API doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
        yield  # This is unreachable but makes the type checker happy
    
    async def stop_streaming(self, session_id: str) -> None:
        """OpenAI Whisper API doesn't support real-time streaming"""
        pass  # No-op since streaming isn't supported
    
    def _map_language_to_whisper(self, language: str) -> str:
        """Map standard language code to OpenAI Whisper format"""
        return self.LANGUAGE_MAP.get(language, language)
    
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """
        Estimate cost for transcription
        
        OpenAI Whisper API pricing (as of 2024):
        - $0.006 per minute ($0.0001 per second)
        """
        return duration_seconds * 0.0001
    
    async def test_connection(self) -> bool:
        """Test if OpenAI Whisper API is accessible"""
        try:
            # Create a minimal test audio file (1 second of silence)
            import struct
            import numpy as np
            
            sample_rate = 16000
            duration = 1.0
            silence = np.zeros(int(sample_rate * duration), dtype=np.int16)
            audio_data = struct.pack(f'<{len(silence)}h', *silence)
            
            # Test transcription with minimal audio
            result = await self.transcribe_audio(
                audio_data=audio_data,
                audio_format="wav",
                sample_rate=sample_rate,
                language="en"
            )
            
            # If we get a result (even empty), the API is working
            return True
            
        except Exception as e:
            logger.error(f"OpenAI Whisper API connection test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # No persistent resources to clean up for API-based provider
        pass