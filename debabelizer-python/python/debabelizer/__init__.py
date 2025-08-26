"""
Debabelizer - Universal Voice Processing Library

Breaking down language barriers, one voice at a time.

⚠️ DEPRECATED: This package is no longer maintained.
Neither the Python nor Rust implementations are production-ready.
"""

import warnings

warnings.warn(
    "The 'debabelizer' package is deprecated and no longer maintained. "
    "This was an experimental project that did not achieve production readiness. "
    "Please consider using established voice processing alternatives.",
    DeprecationWarning,
    stacklevel=2
)

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
__version__ = "0.2.5"

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