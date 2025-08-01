# 🗣️ Debabelizer

**Voice Processing Library - Breaking Down Language Barriers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Debabelizer is a voice processing library that provides a unified interface for speech-to-text (STT) and text-to-speech (TTS) operations across multiple cloud providers and local engines. Break down language barriers with support for 100+ languages and dialects.

## 🌟 Features

### 🎯 **Pluggable Provider Support**
- **6 STT Providers**: Soniox, Deepgram, Google Cloud, Azure, OpenAI Whisper (local), OpenAI Whisper (API)
- **4 TTS Providers**: ElevenLabs, OpenAI, Google Cloud, Azure
- **Unified API**: Switch providers without changing code
- **Provider-specific optimizations**: Each provider uses its optimal streaming/processing approach

### 🌍 **Comprehensive Language Support**
- **100+ languages and dialects** across all providers
- **Automatic language detection** 
- **Multi-language processing** in single workflows
- **Custom language hints** for improved accuracy

### ⚡ **Advanced Processing**
- **Real-time streaming** transcription (Soniox, Deepgram with true WebSocket streaming)
- **Chunk-based transcription** for reliable web application audio processing
- **File-based transcription** for batch processing
- **Word-level timestamps** and confidence scores
- **Speaker diarization** and voice identification (provider-dependent)
- **Custom voice training** and cloning (ElevenLabs)

### 🏠 **Local & Cloud Options**
- **OpenAI Whisper**: Complete offline processing (FREE)
- **Cloud APIs**: Enterprise-grade accuracy and features
- **Hybrid workflows**: Mix local and cloud processing
- **Cost optimization**: Automatic provider selection by cost/quality

### 🛠️ **Enterprise Ready**
- **Async/await support** for high-performance applications
- **Session management** for long-running processes
- **Error handling** with provider-specific fallbacks
- **Usage tracking** and cost estimation
- **Extensive configuration** options

## 📦 Installation

### Basic Installation
```bash
pip install debabelizer
```

### Provider-Specific Installation
```bash
# Individual providers
pip install debabelizer[soniox]      # Soniox STT
pip install debabelizer[deepgram]    # Deepgram STT
pip install debabelizer[google]      # Google Cloud STT & TTS
pip install debabelizer[azure]       # Azure STT & TTS
pip install debabelizer[whisper]     # OpenAI Whisper STT (local)
pip install debabelizer[elevenlabs]  # ElevenLabs TTS
pip install debabelizer[openai]      # OpenAI TTS & Whisper API

# All providers
pip install debabelizer[all]

# Development
pip install debabelizer[dev]
```

### Development Installation
```bash
git clone https://github.com/your-org/debabelizer.git
cd debabelizer
pip install -e .[dev]
```

## 🚀 Quick Start

### Basic Speech-to-Text
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def transcribe_audio():
    # Configure with your preferred provider
    config = DebabelizerConfig({
        "deepgram": {"api_key": "your_deepgram_key"},
        "preferences": {"stt_provider": "deepgram"}
    })
    
    # Create processor
    processor = VoiceProcessor(config=config)
    
    # Transcribe audio file
    result = await processor.transcribe_file("audio.wav")
    
    print(f"Text: {result.text}")
    print(f"Language: {result.language_detected}")
    print(f"Confidence: {result.confidence}")

# Run transcription
asyncio.run(transcribe_audio())
```

### Basic Text-to-Speech
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def synthesize_speech():
    # Configure TTS provider
    config = DebabelizerConfig({
        "elevenlabs": {"api_key": "your_elevenlabs_key"}
    })
    
    processor = VoiceProcessor(
        tts_provider="elevenlabs", 
        config=config
    )
    
    # Synthesize speech
    result = await processor.synthesize(
        text="Hello world! This is Debabelizer speaking.",
        voice="Rachel"  # ElevenLabs voice
    )
    
    # Save audio
    with open("output.mp3", "wb") as f:
        f.write(result.audio_data)

asyncio.run(synthesize_speech())
```

### Local Processing (FREE with Whisper)
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def local_transcription():
    # No API key needed for Whisper!
    config = DebabelizerConfig({
        "whisper": {
            "model_size": "base",  # tiny, base, small, medium, large
            "device": "auto"       # auto-detects GPU/CPU
        }
    })
    
    processor = VoiceProcessor(stt_provider="whisper", config=config)
    
    # Completely offline transcription
    result = await processor.transcribe_file("audio.wav")
    print(f"Offline transcription: {result.text}")

asyncio.run(local_transcription())
```

### Real-time Streaming (Provider-Specific)
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def streaming_transcription():
    # Note: True streaming varies by provider
    # Soniox: True real-time WebSocket streaming
    # Deepgram: True real-time WebSocket streaming  
    # Google/Azure: Session-based streaming with optimizations
    
    config = DebabelizerConfig({
        "soniox": {"api_key": "your_key"}  # Best for true streaming
    })
    
    processor = VoiceProcessor(stt_provider="soniox", config=config)
    
    # Start streaming session
    session_id = await processor.start_streaming_transcription(
        audio_format="pcm",  # Raw PCM preferred for streaming
        sample_rate=16000,
        language="en"
    )
    
    # Stream audio chunks (typically 16ms - 100ms chunks)
    with open("audio.wav", "rb") as f:
        chunk_size = 1024  # Small chunks for real-time
        while chunk := f.read(chunk_size):
            await processor.stream_audio(session_id, chunk)
    
    # Get results as they arrive
    async for result in processor.get_streaming_results(session_id):
        if result.is_final:
            print(f"Final: {result.text}")
        else:
            print(f"Interim: {result.text}")
    
    await processor.stop_streaming_transcription(session_id)

asyncio.run(streaming_transcription())
```

### File-Based Transcription (Alternative to Streaming)
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def file_transcription():
    """
    Process complete audio files or buffered audio chunks.
    Alternative to streaming for applications that can buffer audio.
    """
    config = DebabelizerConfig({
        "deepgram": {"api_key": "your_key"}
    })
    
    processor = VoiceProcessor(stt_provider="deepgram", config=config)
    
    # Process complete audio file
    result = await processor.transcribe_file("audio.wav")
    
    # Or process audio data from memory (e.g., from web upload)
    with open("audio_chunk.webm", "rb") as f:
        chunk_data = f.read()  # WebM/Opus from MediaRecorder
    
    # Process audio data directly
    result = await processor.transcribe_audio(
        audio_data=chunk_data,
        audio_format="webm",     # Browser WebM/Opus format
        sample_rate=48000,       # Browser standard
        language="en"
    )
    
    print(f"Result: {result.text}")
    print(f"Confidence: {result.confidence}")
    print(f"Language: {result.language_detected}")

asyncio.run(file_transcription())
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
# Provider API Keys
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
SONIOX_API_KEY=your_soniox_key

# Azure (requires key + region)
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=eastus

# Google Cloud (requires service account JSON)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json

# Preferences
DEBABELIZER_STT_PROVIDER=deepgram
DEBABELIZER_TTS_PROVIDER=elevenlabs
DEBABELIZER_OPTIMIZE_FOR=quality  # cost, latency, quality, balanced
```

### Authentication Requirements by Provider

#### Google Cloud STT/TTS
**Requires**: Service account JSON file or Application Default Credentials
```bash
# Option 1: Service account file
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Option 2: Use gcloud CLI (for development)
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

#### Azure STT/TTS
**Requires**: API key + region
```bash
AZURE_SPEECH_KEY=your_api_key_here
AZURE_SPEECH_REGION=eastus  # or your preferred region
```

#### OpenAI (TTS & Whisper API)
**Requires**: OpenAI API key
```bash
OPENAI_API_KEY=your_openai_api_key
```

#### ElevenLabs TTS
**Requires**: ElevenLabs API key
```bash
ELEVENLABS_API_KEY=your_elevenlabs_key
```

#### Deepgram STT
**Requires**: Deepgram API key
```bash
DEEPGRAM_API_KEY=your_deepgram_key
```

#### Soniox STT
**Requires**: Soniox API key
```bash
SONIOX_API_KEY=your_soniox_key
```

#### OpenAI Whisper (Local)
**Requires**: No API key (completely offline)
- Automatically downloads models on first use
- Supports GPU acceleration with CUDA/MPS

## 🎯 Provider Comparison & Testing Status

### Speech-to-Text (STT) Providers

| Provider | Status | Streaming | Language Auto-Detection | Testing | Authentication | Best For |
|----------|--------|-----------|-------------------------|---------|----------------|----------|
| **Soniox** | ✅ **Verified** | True WebSocket streaming | ✅  **Verified** | ✅ **Tested & Fixed** | API Key | Real-time applications |
| **Deepgram** | ✅ **Verified** | True WebSocket streaming | ✅ **Claimed** | ✅ **Tested & Fixed** | API Key | High accuracy & speed |
| **Google Cloud** | ✅ **Code Fixed** | Session-based streaming | ⚠️ **Limited** | ⚠️ **Needs Auth Setup** | Service Account JSON | Enterprise features |
| **Azure** | ✅ **Code Fixed** | Session-based streaming | ✅ **Claimed** | ⚠️ **Needs Auth Setup** | API Key + Region | Microsoft ecosystem |
| **OpenAI Whisper (Local)** | ✅ **Verified** | File-based only | ❓ **Unclear** | ✅ **Tested** | None (offline) | Cost-free processing |
| **OpenAI Whisper (API)** | ✅ **Available** | File-based only | ❓ **Unclear** | ⚠️ **Not tested** | OpenAI API Key | Cloud Whisper |

### Text-to-Speech (TTS) Providers

| Provider | Status | Streaming | Testing | Authentication | Best For |
|----------|--------|-----------|---------|----------------|----------|
| **ElevenLabs** | ✅ **Verified** | Simulated streaming | ✅ **Tested & Working** | API Key | Voice cloning & quality |
| **OpenAI** | ✅ **Verified** | Simulated streaming | ✅ **Tested & Fixed** | OpenAI API Key | Natural voices |
| **Google Cloud** | ✅ **Available** | TBD | ⚠️ **Not tested** | Service Account JSON | Enterprise features |
| **Azure** | ✅ **Available** | TBD | ⚠️ **Not tested** | API Key + Region | Microsoft ecosystem |

### Key Testing Results

#### ✅ **Fully Tested & Verified**
- **OpenAI TTS**: All features working, issues fixed (sample rate accuracy, duration estimation, streaming transparency)
- **ElevenLabs TTS**: All features working, fully tested and verified
- **Soniox STT**: Streaming implementation fixed (method names, session management)
- **Deepgram STT**: True WebSocket streaming implemented and working

#### ✅ **Code Issues Fixed (Ready for Testing)**
- **Google Cloud STT**: Fixed critical async/sync mixing bugs in streaming implementation
- **Azure STT**: Fixed critical async/sync mixing bugs in event handlers

#### ⚠️ **Available but Needs Testing**
- **Google Cloud TTS**: Implementation exists but not tested  
- **Azure TTS**: Implementation exists but not tested
- **OpenAI Whisper API**: Implementation exists but not tested

## 🔧 Advanced Usage


### Provider-Specific Optimizations

```python
# Soniox: Best for true real-time streaming
soniox_config = DebabelizerConfig({
    "soniox": {
        "api_key": "your_key",
        "model": "en_v2",
        "include_profanity": False,
        "enable_global_speaker_diarization": True
    }
})

# Deepgram: High accuracy with true streaming
deepgram_config = DebabelizerConfig({
    "deepgram": {
        "api_key": "your_key", 
        "model": "nova-2",
        "language": "en",
        "interim_results": True,
        "vad_events": True
    }
})

# Google Cloud: Enterprise features (requires service account)
google_config = DebabelizerConfig({
    "google": {
        "credentials_path": "/path/to/service-account.json",
        "project_id": "your-project-id",
        "model": "latest_long",
        "enable_speaker_diarization": True,
        "enable_word_time_offsets": True
    }
})

# Azure: Microsoft ecosystem integration
azure_config = DebabelizerConfig({
    "azure": {
        "api_key": "your_key",
        "region": "eastus",
        "language": "en-US",
        "enable_dictation": True,
        "profanity_filter": True
    }
})

# OpenAI Whisper: Free local processing
whisper_config = DebabelizerConfig({
    "whisper": {
        "model_size": "medium",  # tiny, base, small, medium, large
        "device": "cuda",        # cpu, cuda, mps, auto
        "fp16": True,           # Faster inference with GPU
        "language": None        # Auto-detect
    }
})
```

### Web Application Integration

```python
from fastapi import FastAPI, UploadFile, File, WebSocket
from debabelizer import VoiceProcessor, DebabelizerConfig
import asyncio

app = FastAPI()

# Initialize processor globally
config = DebabelizerConfig()
processor = VoiceProcessor(config=config)

@app.post("/transcribe-chunk")
async def transcribe_chunk(file: UploadFile = File(...)):
    """
    Recommended approach for web applications.
    Process audio chunks from browser MediaRecorder.
    """
    content = await file.read()
    
    # Use audio transcription for buffered chunks
    result = await processor.transcribe_audio(
        audio_data=content,
        audio_format="webm",    # Common browser format
        sample_rate=48000,      # Browser standard
        language="en"
    )
    
    return {
        "text": result.text,
        "language": result.language_detected,
        "confidence": result.confidence,
        "duration": result.duration,
        "method": "chunk_transcription"
    }

@app.websocket("/transcribe-stream")
async def transcribe_stream(websocket: WebSocket):
    """
    True streaming approach for specialized applications.
    Requires careful connection management.
    """
    await websocket.accept()
    
    # Start streaming session
    session_id = await processor.start_streaming_transcription(
        audio_format="pcm",
        sample_rate=16000,
        language="en"
    )
    
    try:
        while True:
            # Receive audio chunk from WebSocket
            audio_chunk = await websocket.receive_bytes()
            
            # Stream to STT provider
            await processor.stream_audio(session_id, audio_chunk)
            
            # Get results and send back
            async for result in processor.get_streaming_results(session_id):
                await websocket.send_json({
                    "text": result.text,
                    "is_final": result.is_final,
                    "confidence": result.confidence
                })
                
                if result.is_final:
                    break
                    
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        await processor.stop_streaming_transcription(session_id)
```

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest

# Specific test categories
python -m pytest tests/test_voice_processor.py  # Core functionality
python -m pytest tests/test_config.py          # Configuration
python -m pytest tests/test_providers/         # Provider tests

# Integration tests (requires API keys)
python -m pytest tests/test_integration.py

# With coverage
python -m pytest --cov=debabelizer --cov-report=html
```

### Test Results
Current test status: **150/165 tests passing, 15 skipped** ✅

```bash
# Test specific providers (requires API keys in .env)
python tests/test_openai_tts.py      # OpenAI TTS (tested ✅)
python tests/test_soniox_stt.py      # Soniox STT (tested ✅) 
python tests/test_deepgram_stt.py    # Deepgram STT (tested ✅)
python tests/test_google_stt.py      # Google STT (needs auth setup)
python tests/test_azure_stt.py       # Azure STT (needs auth setup)
```

## 🚨 Known Issues & Limitations

### Current Limitations
1. **Google Cloud & Azure**: Code is fixed but requires proper authentication setup for testing
2. **TTS Streaming**: Most providers simulate streaming (download full audio, then chunk) - only true for specialized streaming TTS APIs
3. **OpenAI TTS**: Correctly reports 24kHz output, but doesn't support custom sample rates
4. **WebM Audio**: Some providers may need audio format conversion for browser-generated WebM/Opus

### Fixed Issues
- ✅ **Google STT**: Fixed critical async/sync mixing in streaming implementation
- ✅ **Azure STT**: Fixed async/sync mixing in event handlers  
- ✅ **OpenAI TTS**: Fixed sample rate accuracy, duration estimation, and streaming transparency
- ✅ **Soniox STT**: Fixed method name mismatches and session management
- ✅ **Deepgram STT**: Implemented true WebSocket streaming

## 🤝 Contributing

We welcome contributions! 

### Development Setup
```bash
git clone https://github.com/your-org/debabelizer.git
cd debabelizer
pip install -e .[dev]
pre-commit install
```

### Testing New Providers
1. Add comprehensive test coverage
2. Follow the systematic debugging approach documented in CLAUDE.md
3. Test both file-based and streaming implementations
4. Verify error handling and edge cases

### Adding New Providers
1. Implement the provider interface in `src/debabelizer/providers/`
2. Add configuration support in `src/debabelizer/core/config.py`
3. Update processor in `src/debabelizer/core/processor.py`
4. Add comprehensive tests in `tests/`
5. Update documentation

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/techwiz42/debabelizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/techwiz42/debabelizer/discussions)

## 🙏 Acknowledgments

- OpenAI for Whisper models and TTS API
- All provider teams for their excellent APIs
- Contributors and testers
- The open-source community

---

**Debabelizer** - *Breaking down language barriers, one voice at a time* 🌍🗣️

*Last updated: 2025-07-31 - Comprehensive testing and bug fixes for OpenAI TTS, Soniox STT, Deepgram STT, Google Cloud STT, and Azure STT implementations*
