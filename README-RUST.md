# Debabelizer Rust Implementation

This is the Rust implementation of Debabelizer - a universal voice processing library with support for multiple STT/TTS providers.

## Project Structure

```
debabelizer-rust/
├── debabelizer/                 # Main library crate
├── debabelizer-core/           # Core traits and types
├── debabelizer-utils/          # Audio utilities
├── providers/                  # Provider implementations
│   ├── soniox/                # Soniox STT (with auto language detection)
│   ├── elevenlabs/            # ElevenLabs TTS
│   ├── deepgram/              # (TODO) Deepgram STT
│   ├── openai/                # (TODO) OpenAI TTS & Whisper
│   ├── google/                # (TODO) Google Cloud STT/TTS
│   ├── azure/                 # (TODO) Azure Cognitive Services
│   └── whisper-local/         # (TODO) Local Whisper
└── examples/                   # Usage examples
```

## Building

```bash
# Build all crates
cargo build

# Build with all providers
cargo build --features all-providers

# Build release version
cargo build --release
```

## Running Examples

```bash
# Basic transcription
cargo run --example basic_transcription

# Voice synthesis
cargo run --example voice_synthesis

# Streaming STT
cargo run --example streaming_stt

# Full pipeline (STT -> TTS)
cargo run --example full_pipeline
```

## Configuration

Set environment variables or create a configuration file:

```bash
# Environment variables
export SONIOX_API_KEY=your_api_key
export ELEVENLABS_API_KEY=your_api_key
export DEBABELIZER_STT_PROVIDER=soniox
export DEBABELIZER_TTS_PROVIDER=elevenlabs
```

Or create `.debabelizer.toml`:

```toml
[preferences]
stt_provider = "soniox"
tts_provider = "elevenlabs"
auto_select = false
optimize_for = "balanced"

[soniox]
api_key = "your_api_key"
auto_detect_language = true

[elevenlabs]
api_key = "your_api_key"
model_id = "eleven_monolingual_v1"
```

## Features

### Implemented
- ✅ Core architecture and traits
- ✅ Configuration system
- ✅ VoiceProcessor with provider selection
- ✅ Session management
- ✅ Soniox STT provider (with language auto-detection)
- ✅ ElevenLabs TTS provider
- ✅ Basic examples

### TODO
- ⏳ Deepgram STT provider
- ⏳ OpenAI providers (TTS & Whisper)
- ⏳ Google Cloud providers
- ⏳ Azure providers
- ⏳ Local Whisper provider
- ⏳ Audio format conversion utilities
- ⏳ Integration tests
- ⏳ Documentation
- ⏳ CI/CD setup

## Testing

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p debabelizer-core

# Run with verbose output
cargo test -- --nocapture
```

## License

This project is dual-licensed under MIT OR Apache-2.0.