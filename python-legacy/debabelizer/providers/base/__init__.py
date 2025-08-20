"""
Base provider interfaces for Debabelizer
"""

from .stt_provider import STTProvider, TranscriptionResult, StreamingResult, WordTiming
from .tts_provider import TTSProvider, SynthesisResult, Voice, AudioFormat
from .exceptions import ProviderError, AuthenticationError, RateLimitError, ConfigurationError

__all__ = [
    "STTProvider",
    "TranscriptionResult", 
    "StreamingResult",
    "WordTiming",
    "TTSProvider",
    "SynthesisResult",
    "Voice",
    "AudioFormat",
    "ProviderError",
    "AuthenticationError", 
    "RateLimitError",
    "ConfigurationError",
]