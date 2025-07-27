"""
OpenAI Whisper Speech-to-Text Provider

Implements STT using OpenAI's Whisper models with support for:
- Local/offline transcription (no API calls required)
- 99 languages supported
- Multiple model sizes (tiny, base, small, medium, large)
- Word-level timestamps
- Automatic language detection
- Robust noise handling
- VAD (Voice Activity Detection)
- Batch processing for long audio
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path
import json

try:
    import whisper
    import torch
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    torch = None
    np = None

from ..base import STTProvider, TranscriptionResult, StreamingResult, WordTiming
from ..base.exceptions import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class WhisperSTTProvider(STTProvider):
    """
    OpenAI Whisper Speech-to-Text Provider
    
    Local/offline speech recognition with high accuracy across 99 languages.
    No internet connection required after model download.
    """
    
    name = "whisper"
    
    # Whisper language codes (subset of most common ones)
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
        "cy": "cy",
        "yi": "yi",
        "yo": "yo",
        "zu": "zu"
    }
    
    # Model sizes and their characteristics
    MODEL_INFO = {
        "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x", "accuracy": "lowest"},
        "base": {"params": "74M", "vram": "~1GB", "speed": "~16x", "accuracy": "low"},
        "small": {"params": "244M", "vram": "~2GB", "speed": "~6x", "accuracy": "medium"},
        "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x", "accuracy": "high"},
        "large": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "highest"},
        "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "highest"},
        "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "highest"}
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
        in_memory: bool = True,
        fp16: bool = True,
        temperature: float = 0.0,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        **kwargs
    ):
        """
        Initialize OpenAI Whisper STT Provider
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cpu', 'cuda', 'auto')
            download_root: Directory to store models
            in_memory: Keep model in memory for faster inference
            fp16: Use float16 precision (faster, less memory)
            temperature: Temperature for sampling (0.0 = deterministic)
            compression_ratio_threshold: Threshold for compression ratio
            logprob_threshold: Threshold for log probability
            no_speech_threshold: Threshold for no speech detection
            condition_on_previous_text: Use previous text as context
        """
        super().__init__()
        
        if not WHISPER_AVAILABLE:
            raise ConfigurationError(
                "OpenAI Whisper not available. Install with: pip install openai-whisper",
                self.name
            )
        
        self.model_size = model_size
        self.device = device or self._get_best_device()
        self.download_root = download_root
        self.in_memory = in_memory
        self.fp16 = fp16 and self.device != "cpu"
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        
        # Model instance
        self._model = None
        self._model_loaded = False
        
        # Validate model size
        if model_size not in self.MODEL_INFO:
            raise ConfigurationError(
                f"Invalid model size: {model_size}. "
                f"Available: {list(self.MODEL_INFO.keys())}",
                self.name
            )
        
        # Load model if in_memory is True
        if self.in_memory:
            self._load_model()
        
        logger.info(f"Initialized Whisper STT with model: {model_size} on {self.device}")
    
    def _get_best_device(self) -> str:
        """Determine the best device to use"""
        if torch and torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the Whisper model"""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=self.download_root
            )
            self._model_loaded = True
            logger.info(f"Whisper model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ConfigurationError(f"Failed to load Whisper model: {e}", self.name)
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supports_streaming(self) -> bool:
        """Whisper doesn't support real-time streaming (file-based only)"""
        return False
    
    @property
    def supports_language_detection(self) -> bool:
        """Whisper supports automatic language detection"""
        return True
    
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Whisper
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            language_hints: Not used by Whisper (auto-detection is built-in)
            **kwargs: Additional Whisper options
            
        Returns:
            TranscriptionResult with transcription
        """
        try:
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model()
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Map language code
            whisper_language = None
            if language:
                whisper_language = self._map_language_to_whisper(language)
                if whisper_language not in whisper.tokenizer.LANGUAGES:
                    logger.warning(f"Language '{language}' not supported by Whisper, using auto-detection")
                    whisper_language = None
            
            # Prepare transcription options
            transcribe_options = {
                "language": whisper_language,
                "task": "transcribe",  # vs "translate"
                "temperature": kwargs.get("temperature", self.temperature),
                "compression_ratio_threshold": kwargs.get("compression_ratio_threshold", self.compression_ratio_threshold),
                "logprob_threshold": kwargs.get("logprob_threshold", self.logprob_threshold),
                "no_speech_threshold": kwargs.get("no_speech_threshold", self.no_speech_threshold),
                "condition_on_previous_text": kwargs.get("condition_on_previous_text", self.condition_on_previous_text),
                "word_timestamps": True,  # Always get word timestamps
                "fp16": self.fp16
            }
            
            # Remove None values
            transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
            
            logger.info(f"Transcribing file with Whisper: {file_path}")
            start_time = datetime.now()
            
            # Run transcription in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.transcribe(str(file_path), **transcribe_options)
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Process results
            text = result.get("text", "").strip()
            
            # Extract segments and words
            segments = result.get("segments", [])
            all_words = []
            total_confidence = 0.0
            confidence_count = 0
            
            for segment in segments:
                # Get segment-level confidence if available
                if "avg_logprob" in segment:
                    # Convert log probability to confidence (approximate)
                    segment_confidence = max(0.0, min(1.0, np.exp(segment["avg_logprob"])))
                    total_confidence += segment_confidence
                    confidence_count += 1
                
                # Extract word-level timestamps
                if "words" in segment:
                    for word_info in segment["words"]:
                        all_words.append(WordTiming(
                            word=word_info.get("word", "").strip(),
                            start_time=word_info.get("start", 0.0),
                            end_time=word_info.get("end", 0.0),
                            confidence=getattr(word_info, "probability", None)
                        ))
            
            # Calculate average confidence
            avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.8
            
            # Detect language from result
            detected_language = result.get("language")
            if detected_language:
                detected_language = self._map_whisper_language_to_standard(detected_language)
            
            return TranscriptionResult(
                text=text,
                confidence=avg_confidence,
                language_detected=detected_language or language,
                duration=duration,
                words=all_words,
                metadata={
                    "model": self.model_size,
                    "device": self.device,
                    "segments": len(segments),
                    "whisper_language": result.get("language"),
                    "processing_time": duration
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise ProviderError(f"Whisper transcription failed: {e}", self.name)
    
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
        
        Whisper works with files, so we need to save audio data to a temp file first.
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
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
        """Whisper doesn't support real-time streaming"""
        raise ProviderError(
            "Whisper doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
    
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Whisper doesn't support real-time streaming"""
        raise ProviderError(
            "Whisper doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
    
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Whisper doesn't support real-time streaming"""
        raise ProviderError(
            "Whisper doesn't support real-time streaming. Use transcribe_file or transcribe_audio instead.",
            self.name
        )
        yield  # This is unreachable but makes the type checker happy
    
    async def stop_streaming(self, session_id: str) -> None:
        """Whisper doesn't support real-time streaming"""
        pass  # No-op since streaming isn't supported
    
    def _map_language_to_whisper(self, language: str) -> str:
        """Map standard language code to Whisper format"""
        # Whisper uses ISO 639-1 codes directly for most languages
        return self.LANGUAGE_MAP.get(language, language)
    
    def _map_whisper_language_to_standard(self, whisper_lang: str) -> str:
        """Map Whisper language code back to standard format"""
        # Reverse lookup in language map
        for std_lang, w_lang in self.LANGUAGE_MAP.items():
            if w_lang == whisper_lang:
                return std_lang
        return whisper_lang
    
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """
        Estimate cost for transcription
        
        Whisper is free to use (local processing), only compute costs.
        Return 0.0 since there are no API charges.
        """
        return 0.0  # Free local processing
    
    async def test_connection(self) -> bool:
        """Test if Whisper model can be loaded and used"""
        try:
            # Try to load the model
            if not self._model_loaded:
                self._load_model()
            
            # Try a simple operation to verify the model works
            if self._model is not None:
                # Create a small test audio (1 second of silence)
                sample_rate = 16000
                duration = 1.0
                silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
                
                # Test transcription (should return empty or minimal result)
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._model.transcribe(silence, fp16=self.fp16, temperature=0.0)
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Whisper connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = self.MODEL_INFO.get(self.model_size, {}).copy()
        info.update({
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self._model_loaded,
            "fp16": self.fp16,
            "languages_supported": len(self.LANGUAGE_MAP),
            "local_processing": True,
            "requires_internet": False
        })
        return info
    
    def list_available_models(self) -> List[str]:
        """List all available Whisper model sizes"""
        return list(self.MODEL_INFO.keys())
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self._model is not None:
            # Clear model from memory
            del self._model
            self._model = None
            self._model_loaded = False
            
            # Clear CUDA cache if using GPU
            if torch and torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("Whisper model cleaned up")