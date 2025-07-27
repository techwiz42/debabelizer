"""
Debabelizer - Universal Voice Processing Library

Breaking down language barriers, one voice at a time.
"""

from .core.processor import VoiceProcessor
from .core.config import DebabelizerConfig
from .providers.base import (
    STTProvider, TTSProvider, TranscriptionResult, SynthesisResult,
    StreamingResult, Voice, AudioFormat, ProviderError
)

# Version
__version__ = "0.1.0"

# Main exports
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
    "ProviderError",
]

# Convenience function for quick setup
def create_processor(
    stt_provider: str = "soniox",
    tts_provider: str = "elevanlabs", 
    **config
) -> VoiceProcessor:
    """
    Convenience function to create a VoiceProcessor with common defaults
    
    Args:
        stt_provider: STT provider name
        tts_provider: TTS provider name
        **config: Additional configuration
        
    Returns:
        Configured VoiceProcessor instance
    """
    return VoiceProcessor(
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        config=config
    )