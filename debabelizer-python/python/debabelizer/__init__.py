"""
Debabelizer - Universal Voice Processing Library

Breaking down language barriers, one voice at a time.
"""

from ._internal import (
    AudioFormat,
    AudioData,
    WordTiming,
    TranscriptionResult,
    Voice,
    SynthesisResult,
    StreamingResult,
    DebabelizerConfig,
    VoiceProcessor,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError,
    create_processor,
)

# Create aliases to match the original Python API exactly
STTProvider = None  # Base class - not directly exposed in Rust bindings
TTSProvider = None  # Base class - not directly exposed in Rust bindings

# Version
__version__ = "0.1.45"

# Main exports - matching the original exactly
__all__ = [
    "VoiceProcessor",
    "DebabelizerConfig", 
    "STTProvider",
    "TTSProvider",
    "TranscriptionResult",
    "SynthesisResult", 
    "StreamingResult",
    "Voice",
    "AudioFormat",
    "AudioData",
    "ProviderError",
    # "WordTiming",  # Not exposed in legacy API at module level
    "AuthenticationError",
    "RateLimitError", 
    "ConfigurationError",
    "create_processor",
]