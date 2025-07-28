# 🗣️ Debabelizer

**Universal Voice Processing Library - Breaking Down Language Barriers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-105%2F120%20passing-green.svg)](#testing)

Debabelizer is a powerful, universal voice processing library that provides a unified interface for speech-to-text (STT) and text-to-speech (TTS) operations across multiple cloud providers and local engines. Break down language barriers with support for 400+ languages and dialects.

## 🌟 Features

### 🎯 **Pluggable Provider Support**
- **5 STT Providers**: Soniox, Deepgram, Google Cloud, Azure, OpenAI Whisper
- **4 TTS Providers**: ElevenLabs, OpenAI, Google Cloud, Azure
- **Unified API**: Switch providers without changing code
- **Smart Fallbacks**: Automatic provider selection based on optimization strategy

### 🌍 **Comprehensive Language Support**
- **40+ languages and dialects** across all providers
- **Automatic language detection** 
- **Multi-language processing** in single workflows
- **Custom language hints** for improved accuracy

### ⚡ **Advanced Processing**
- **Real-time streaming** transcription and synthesis
- **Batch processing** for large files
- **Word-level timestamps** and confidence scores
- **Speaker diarization** and voice identification
- **Custom voice training** and cloning

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
pip install debabelizer[openai]      # OpenAI TTS

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

## 🔨 Building from Source

### Prerequisites
```bash
# Required build tools
pip install build twine setuptools wheel

# For development
pip install -e .[dev]
```

### Build Distribution
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build source and wheel distributions
python -m build

# Verify the build
twine check dist/*
```

### Local Installation from Build
```bash
# Install from local wheel
pip install dist/debabelizer-*.whl

# Or install from source distribution
pip install dist/debabelizer-*.tar.gz
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

### Real-time Streaming
```python
import asyncio
from debabelizer import VoiceProcessor, DebabelizerConfig

async def streaming_transcription():
    config = DebabelizerConfig({
        "deepgram": {"api_key": "your_key"}
    })
    
    processor = VoiceProcessor(stt_provider="deepgram", config=config)
    
    # Start streaming session
    session_id = await processor.start_streaming_transcription(
        audio_format="wav",
        sample_rate=16000
    )
    
    # Stream audio chunks
    with open("audio.wav", "rb") as f:
        chunk_size = 1024
        while chunk := f.read(chunk_size):
            await processor.stream_audio(session_id, chunk)
    
    # Get results
    async for result in processor.get_streaming_results(session_id):
        if result.is_final:
            print(f"Final: {result.text}")
        else:
            print(f"Interim: {result.text}")
    
    await processor.stop_streaming_transcription(session_id)

asyncio.run(streaming_transcription())
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
# Provider API Keys
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=eastus
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json

# Preferences
DEBABELIZER_STT_PROVIDER=deepgram
DEBABELIZER_TTS_PROVIDER=elevenlabs
DEBABELIZER_OPTIMIZE_FOR=quality  # cost, latency, quality, balanced
```

### Programmatic Configuration
```python
from debabelizer import DebabelizerConfig, VoiceProcessor

# Method 1: Direct configuration
config = DebabelizerConfig({
    "deepgram": {
        "api_key": "your_key",
        "model": "nova-2",
        "language": "en-US"
    },
    "elevenlabs": {
        "api_key": "your_key",
        "voice": "Rachel",
        "stability": 0.5,
        "similarity_boost": 0.75
    },
    "preferences": {
        "stt_provider": "deepgram",
        "tts_provider": "elevenlabs",
        "optimize_for": "quality"
    }
})

# Method 2: Environment-based (recommended)
config = DebabelizerConfig()  # Auto-loads from environment

processor = VoiceProcessor(config=config)
```

### Provider-Specific Configuration

#### Whisper (Local/Offline)
```python
config = DebabelizerConfig({
    "whisper": {
        "model_size": "medium",      # tiny, base, small, medium, large
        "device": "cuda",            # cpu, cuda, mps, auto
        "fp16": True,                # Faster inference
        "temperature": 0.0,          # Deterministic output
        "language": None             # Auto-detect language
    }
})
```

#### Google Cloud
```python
config = DebabelizerConfig({
    "google": {
        "credentials_path": "/path/to/credentials.json",
        "project_id": "your-project-id",
        "model": "latest_long",      # STT model
        "voice_type": "Neural2",     # TTS voice type
        "enable_speaker_diarization": True
    }
})
```

#### Azure
```python
config = DebabelizerConfig({
    "azure": {
        "api_key": "your_key",
        "region": "eastus",
        "language": "en-US",
        "voice": "en-US-JennyNeural",
        "enable_speaker_identification": True
    }
})
```

## 📦 Including in Your Projects

### As a Dependency

#### requirements.txt
```txt
# Basic installation
debabelizer

# With specific providers
debabelizer[deepgram,elevenlabs]

# All providers
debabelizer[all]

# From GitHub (latest development)
git+https://github.com/your-org/debabelizer.git

# Specific version
debabelizer==1.0.0
```

#### pyproject.toml
```toml
[project]
dependencies = [
    "debabelizer[deepgram,openai]>=1.0.0"
]

# Optional dependencies for different use cases
[project.optional-dependencies]
voice = ["debabelizer[all]"]
transcription-only = ["debabelizer[whisper,deepgram]"]
synthesis-only = ["debabelizer[elevenlabs,openai]"]
```

#### setup.py
```python
from setuptools import setup

setup(
    name="your-project",
    install_requires=[
        "debabelizer[whisper,elevenlabs]>=1.0.0",
    ],
    extras_require={
        "full-voice": ["debabelizer[all]>=1.0.0"],
    }
)
```

#### Poetry
```toml
[tool.poetry.dependencies]
python = "^3.8"
debabelizer = {extras = ["deepgram", "openai"], version = "^1.0.0"}

# Or for all providers
# debabelizer = {extras = ["all"], version = "^1.0.0"}
```

### Docker Integration
```dockerfile
FROM python:3.9-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install your application with Debabelizer
COPY requirements.txt .
RUN pip install -r requirements.txt

# Example requirements.txt content:
# debabelizer[whisper,elevenlabs]==1.0.0
# your-other-dependencies

COPY . .
CMD ["python", "your_app.py"]
```

## 🏗️ Integration Examples

### Web API Integration
```python
from fastapi import FastAPI, UploadFile, File
from debabelizer import VoiceProcessor, DebabelizerConfig
import asyncio

app = FastAPI()

# Initialize processor globally
config = DebabelizerConfig()
processor = VoiceProcessor(config=config)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded file
    content = await file.read()
    
    # Transcribe using raw audio data
    result = await processor.transcribe_audio(
        audio_data=content,
        audio_format="wav"
    )
    
    return {
        "text": result.text,
        "language": result.language_detected,
        "confidence": result.confidence,
        "duration": result.duration
    }

@app.post("/synthesize")
async def synthesize_text(text: str, voice: str = "default"):
    result = await processor.synthesize(text=text, voice=voice)
    
    return {
        "audio_size": result.size_bytes,
        "duration": result.duration,
        "format": result.format
        # Note: Return audio_data as base64 or save to file
    }
```

### Background Task Processing
```python
import asyncio
from celery import Celery
from debabelizer import VoiceProcessor, DebabelizerConfig

app = Celery('voice_processing')

@app.task
def process_audio_file(file_path: str, options: dict):
    """Background task for processing large audio files"""
    
    async def _process():
        config = DebabelizerConfig()
        processor = VoiceProcessor(config=config)
        
        # Process with specified options
        result = await processor.transcribe_file(
            file_path,
            language=options.get('language'),
            **options.get('provider_options', {})
        )
        
        return {
            "text": result.text,
            "language": result.language_detected,
            "confidence": result.confidence,
            "words": [
                {
                    "word": w.word,
                    "start": w.start_time,
                    "end": w.end_time,
                    "confidence": w.confidence
                }
                for w in result.words
            ]
        }
    
    # Run async function in sync context
    return asyncio.run(_process())
```

### Django Integration
```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from debabelizer import VoiceProcessor, DebabelizerConfig
import asyncio
import json

# Initialize once at module level
config = DebabelizerConfig()
processor = VoiceProcessor(config=config)

@csrf_exempt
async def transcribe_view(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        
        if audio_file:
            # Read audio data
            audio_data = audio_file.read()
            
            # Transcribe
            result = await processor.transcribe_audio(
                audio_data=audio_data,
                audio_format="wav"
            )
            
            return JsonResponse({
                'success': True,
                'text': result.text,
                'language': result.language_detected,
                'confidence': result.confidence
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/transcribe/', views.transcribe_view, name='transcribe'),
]
```

### Batch Processing Script
```python
#!/usr/bin/env python3
"""
Batch audio processing script
Usage: python batch_process.py /path/to/audio/files --output /path/to/results
"""

import asyncio
import argparse
from pathlib import Path
import json
from debabelizer import VoiceProcessor, DebabelizerConfig

async def process_files(input_dir: Path, output_dir: Path, provider: str = None):
    """Process all audio files in a directory"""
    
    config = DebabelizerConfig()
    processor = VoiceProcessor(
        stt_provider=provider,
        config=config
    )
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    audio_files = [
        f for f in input_dir.rglob('*') 
        if f.suffix.lower() in audio_extensions
    ]
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process files
    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        
        try:
            result = await processor.transcribe_file(str(audio_file))
            
            # Save result
            output_file = output_dir / f"{audio_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "file": str(audio_file),
                    "text": result.text,
                    "language": result.language_detected,
                    "confidence": result.confidence,
                    "duration": result.duration,
                    "word_count": len(result.text.split())
                }, f, indent=2)
                
            print(f"✅ Saved: {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["soniox", "deepgram", "google", "azure", "whisper"])
    
    args = parser.parse_args()
    
    args.output.mkdir(exist_ok=True)
    asyncio.run(process_files(args.input_dir, args.output, args.provider))
```

## 🎯 Advanced Usage

### Multi-Provider Workflow
```python
async def multi_provider_transcription(audio_file: str):
    """Use multiple providers for comparison or fallback"""
    
    config = DebabelizerConfig()
    
    # Primary: High accuracy
    primary = VoiceProcessor(stt_provider="google", config=config)
    
    # Fallback: Local processing
    fallback = VoiceProcessor(stt_provider="whisper", config=config)
    
    try:
        # Try primary provider first
        result = await primary.transcribe_file(audio_file)
        print(f"Primary result: {result.text}")
        return result
        
    except Exception as e:
        print(f"Primary failed: {e}, using fallback...")
        
        # Use local fallback
        result = await fallback.transcribe_file(audio_file)
        print(f"Fallback result: {result.text}")
        return result
```

### Cost Optimization
```python
async def cost_optimized_processing():
    """Automatically select cheapest provider for your needs"""
    
    config = DebabelizerConfig({
        "preferences": {
            "optimize_for": "cost",  # Will prefer cheaper providers
            "auto_select": True
        }
    })
    
    processor = VoiceProcessor(config=config)
    
    # Get cost estimates
    duration = 300  # 5 minutes
    
    estimates = {
        "whisper": 0.0,  # Free
        "soniox": processor._get_provider_cost("soniox", duration),
        "deepgram": processor._get_provider_cost("deepgram", duration),
    }
    
    print("Cost estimates:", estimates)
    
    # Process with auto-selected provider
    result = await processor.transcribe_file("audio.wav")
    return result
```

### Voice Cloning Pipeline
```python
async def voice_cloning_pipeline():
    """Complete pipeline: Record → Transcribe → Clone → Synthesize"""
    
    config = DebabelizerConfig({
        "deepgram": {"api_key": "your_key"},
        "elevenlabs": {"api_key": "your_key"}
    })
    
    stt_processor = VoiceProcessor(stt_provider="deepgram", config=config)
    tts_processor = VoiceProcessor(tts_provider="elevenlabs", config=config)
    
    # 1. Transcribe original audio
    transcript = await stt_processor.transcribe_file("speaker_sample.wav")
    print(f"Original text: {transcript.text}")
    
    # 2. Get available voices
    voices = await tts_processor.get_available_voices()
    target_voice = voices[0]  # Select voice
    
    # 3. Synthesize new text with target voice
    new_text = "This is new text in the cloned voice!"
    result = await tts_processor.synthesize(
        text=new_text,
        voice=target_voice
    )
    
    # 4. Save cloned speech
    with open("cloned_speech.mp3", "wb") as f:
        f.write(result.audio_data)
```

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest

# Specific test categories
python -m pytest tests/test_voice_processor.py  # Core functionality
python -m pytest tests/test_config.py          # Configuration
python -m pytest tests/test_base_providers.py  # Provider interfaces

# Integration tests (requires API keys)
python -m pytest tests/test_integration.py

# With coverage
python -m pytest --cov=debabelizer --cov-report=html
```

### Test Configuration
Set up test environment:
```bash
# Copy example config
cp .env.example .env

# Add your API keys for integration tests
export DEEPGRAM_API_KEY="your_key"
export ELEVENLABS_API_KEY="your_key"
# ... other keys

# Run integration tests
python -m pytest tests/test_integration.py -v
```

## 🤝 Contributing

We welcome contributions! 

### Development Setup
```bash
git clone https://github.com/your-org/debabelizer.git
cd debabelizer
pip install -e .[dev]
pre-commit install
```

### Adding New Providers
1. Implement the provider interface in `src/debabelizer/providers/`
2. Add configuration support in `src/debabelizer/core/config.py`
3. Update processor in `src/debabelizer/core/processor.py`
4. Add tests in `tests/`
5. Update documentation

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/techwiz42/debabelizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/techwiz42/debabelizer/discussions)

## 🙏 Acknowledgments

- OpenAI for Whisper models
- All provider teams for their excellent APIs
- Contributors and testers
- The open-source community

---

**Debabelizer** - *Breaking down language barriers, one voice at a time* 🌍🗣️
