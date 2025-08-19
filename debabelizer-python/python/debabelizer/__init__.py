"""
Debabelizer: Universal voice processing library with support for multiple STT/TTS providers

This package provides a unified interface for Speech-to-Text (STT) and Text-to-Speech (TTS)
services across multiple providers including OpenAI, Google Cloud, Azure, Deepgram, and more.
"""

from ._debabelizer import (
    PyAudioFormat as AudioFormat,
    PyAudioData as AudioData,
    PyWordTiming as WordTiming,
    PyTranscriptionResult as TranscriptionResult,
    PyVoice as Voice,
    PySynthesisResult as SynthesisResult,
    PySynthesisOptions as SynthesisOptions,
    PyVoiceProcessor as VoiceProcessor,
    DebabelizerException,
)

__version__ = "0.1.0"
__author__ = "Debabelizer Contributors"
__email__ = "contributors@debabelizer.com"

__all__ = [
    "AudioFormat",
    "AudioData", 
    "WordTiming",
    "TranscriptionResult",
    "Voice",
    "SynthesisResult",
    "SynthesisOptions",
    "VoiceProcessor",
    "DebabelizerException",
]