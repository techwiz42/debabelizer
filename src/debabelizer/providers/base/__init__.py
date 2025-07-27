"""
Base provider interfaces for Debabelizer
"""

from .stt_provider import STTProvider, TranscriptionResult, StreamingResult
from .tts_provider import TTSProvider, SynthesisResult, Voice, AudioFormat
from .exceptions import ProviderError, AuthenticationError, RateLimitError

__all__ = [
    "STTProvider",
    "TranscriptionResult", 
    "StreamingResult",
    "TTSProvider",
    "SynthesisResult",
    "Voice",
    "AudioFormat",
    "ProviderError",
    "AuthenticationError", 
    "RateLimitError",
]