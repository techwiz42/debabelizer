# Debabelizer Python Bindings

High-performance Python bindings for the Debabelizer universal voice processing library, built with Rust and PyO3.

## Features

- **Speech-to-Text (STT)**: Support for multiple providers (Soniox, Deepgram, Google Cloud, Azure, OpenAI Whisper)
- **Text-to-Speech (TTS)**: Support for multiple providers (ElevenLabs, OpenAI, Google Cloud, Azure)
- **High Performance**: Native Rust implementation with zero-copy operations where possible
- **Async Support**: Full async/await support with Python asyncio compatibility
- **Type Safety**: Comprehensive type hints and runtime validation
- **Easy Configuration**: Environment variable and programmatic configuration

## Installation

### From PyPI (when published)

```bash
pip install debabelizer
```

### From Source

1. Install Rust and maturin:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
```

2. Build and install:
```bash
cd debabelizer-python
maturin develop --release
```

## Quick Start

```python
import debabelizer
import asyncio

# Configure with environment variables or programmatically
config = {
    "preferences": {
        "stt_provider": "deepgram",
        "tts_provider": "openai"
    },
    "deepgram": {
        "api_key": "your_deepgram_key"
    },
    "openai": {
        "api_key": "your_openai_key"
    }
}

# Create processor
processor = debabelizer.VoiceProcessor(config=config)

# Load audio file
with open("audio.wav", "rb") as f:
    audio_data = f.read()

# Create audio format
audio_format = debabelizer.AudioFormat(
    format="wav",
    sample_rate=16000,
    channels=1,
    bit_depth=16
)

# Create audio data object
audio = debabelizer.AudioData(audio_data, audio_format)

# Transcribe audio
result = processor.transcribe(audio)
print(f"Transcription: {result.text}")
print(f"Confidence: {result.confidence}")

# Synthesize speech
synthesis_options = debabelizer.SynthesisOptions(
    voice="alloy",
    speed=1.0,
    format="mp3"
)

synthesis_result = processor.synthesize("Hello, world!", synthesis_options)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(synthesis_result.audio_data)
```

## Configuration

### Environment Variables

```bash
# Core preferences
export DEBABELIZER_STT_PROVIDER=deepgram
export DEBABELIZER_TTS_PROVIDER=openai
export DEBABELIZER_AUTO_SELECT=false
export DEBABELIZER_OPTIMIZE_FOR=quality

# Provider API keys
export DEEPGRAM_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ELEVENLABS_API_KEY=your_key
export SONIOX_API_KEY=your_key

# Google Cloud (using API key or service account)
export GOOGLE_API_KEY=your_key
# OR
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Azure
export AZURE_SPEECH_KEY=your_key
export AZURE_SPEECH_REGION=eastus
```

### Programmatic Configuration

```python
import debabelizer

# Using environment variables
processor = debabelizer.VoiceProcessor()

# Using explicit configuration
config = {
    "preferences": {
        "stt_provider": "deepgram",
        "tts_provider": "elevenlabs",
        "optimize_for": "quality"
    },
    "deepgram": {
        "api_key": "your_key",
        "model": "nova-2"
    },
    "elevenlabs": {
        "api_key": "your_key",
        "voice_id": "21m00Tcm4TlvDq8ikWAM"
    }
}

processor = debabelizer.VoiceProcessor(config=config)

# Override specific providers
processor = debabelizer.VoiceProcessor(
    stt_provider="deepgram",
    tts_provider="openai"
)
```

## Supported Providers

### Speech-to-Text (STT)
- **Soniox**: Telephony-optimized, real-time streaming
- **Deepgram**: Nova-2 model, 40+ languages, <300ms latency
- **Google Cloud**: 125+ languages, streaming, diarization
- **Azure**: 140+ languages, streaming, speaker ID
- **OpenAI Whisper**: 99+ languages, local/offline, zero cost

### Text-to-Speech (TTS)
- **ElevenLabs**: 1000+ voices, voice cloning, streaming
- **OpenAI**: 6 voices, TTS-1/TTS-1-HD models
- **Google Cloud**: 220+ voices, WaveNet/Neural2, SSML
- **Azure**: 300+ neural voices, SSML, voice styles

## API Reference

### VoiceProcessor

Main class for voice processing operations.

```python
class VoiceProcessor:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None
    )
    
    def transcribe(self, audio: AudioData) -> TranscriptionResult
    def synthesize(self, text: str, options: Optional[SynthesisOptions] = None) -> SynthesisResult
    def get_available_voices(self) -> List[Voice]
    def has_stt_provider(self) -> bool
    def has_tts_provider(self) -> bool
    def get_stt_provider_name(self) -> Optional[str]
    def get_tts_provider_name(self) -> Optional[str]
```

### Data Types

```python
class AudioFormat:
    format: str
    sample_rate: int
    channels: int
    bit_depth: int

class AudioData:
    data: bytes
    format: AudioFormat

class TranscriptionResult:
    text: str
    confidence: Optional[float]
    language_detected: Optional[str]
    duration: Optional[float]
    words: Optional[List[WordTiming]]

class SynthesisResult:
    audio_data: bytes
    format: str
    duration: Optional[float]
    size_bytes: int

class SynthesisOptions:
    voice: Optional[str]
    speed: Optional[float]  # 0.25-4.0
    pitch: Optional[float]  # -20.0 to 20.0 semitones
    volume: Optional[float] # 0.0-1.0
    format: Optional[str]
```

## Utilities

```python
from debabelizer.utils import (
    load_audio_file,
    get_audio_format_from_extension,
    create_config_from_env,
    create_synthesis_options,
)

# Load audio file
audio_data = load_audio_file("speech.wav")
format_name = get_audio_format_from_extension("speech.wav")

# Create configuration from environment
config = create_config_from_env()

# Create synthesis options
options = create_synthesis_options(
    voice="alloy",
    speed=1.2,
    format="mp3"
)
```

## Performance

The Rust implementation provides significant performance benefits:

- **10-100x faster** than pure Python implementations
- **Lower memory usage** with zero-cost abstractions
- **True parallelism** without GIL limitations
- **Memory safety** with compile-time guarantees

## License

MIT OR Apache-2.0