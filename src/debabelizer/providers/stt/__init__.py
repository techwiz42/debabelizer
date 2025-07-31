"""
Speech-to-Text Provider Implementations
"""

from .soniox import SonioxSTTProvider

__all__ = ["SonioxSTTProvider"]

# Conditional imports for optional providers
try:
    from .deepgram import DeepgramSTTProvider
    __all__.append("DeepgramSTTProvider")
except ImportError:
    pass

try:
    from .google import GoogleSTTProvider
    __all__.append("GoogleSTTProvider")
except ImportError:
    pass

try:
    from .azure import AzureSTTProvider
    __all__.append("AzureSTTProvider")
except ImportError:
    pass

try:
    from .whisper import WhisperSTTProvider
    __all__.append("WhisperSTTProvider")
except ImportError:
    pass

try:
    from .openai_whisper import OpenAIWhisperSTTProvider
    __all__.append("OpenAIWhisperSTTProvider")
except ImportError:
    pass