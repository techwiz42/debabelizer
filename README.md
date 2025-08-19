# Debabelizer - Universal Voice Processing Library in Rust

A high-performance, memory-safe Rust implementation of the Debabelizer voice processing library, providing a unified interface for Speech-to-Text (STT) and Text-to-Speech (TTS) services across multiple providers.

## 🚀 Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Google Cloud, Azure, Deepgram, ElevenLabs, and more
- **Async-First Design**: Built on Tokio for high-performance concurrent operations
- **Type Safety**: Leverages Rust's type system to prevent runtime errors
- **Memory Safety**: Zero-cost abstractions with guaranteed memory safety
- **Streaming Support**: Real-time WebSocket streaming for both STT and TTS
- **Python Bindings**: PyO3-based Python API for easy integration
- **Extensible Architecture**: Easy to add new providers through trait implementations

## 📁 Project Structure

The project uses a Cargo workspace architecture for modularity and clear separation of concerns:

```
debabelizer/
├── Cargo.toml                    # Workspace configuration
├── debabelizer/                  # Main library crate
│   ├── src/
│   │   ├── lib.rs               # Library entry point and re-exports
│   │   ├── config.rs            # Configuration management
│   │   ├── processor.rs         # Main VoiceProcessor implementation
│   │   ├── providers.rs         # Provider registry and management
│   │   └── session.rs           # Session lifecycle management
│   └── Cargo.toml
├── debabelizer-core/            # Core traits and types
│   ├── src/
│   │   ├── lib.rs              # Core module exports
│   │   ├── audio.rs            # Audio data structures
│   │   ├── error.rs            # Error types and handling
│   │   ├── stt.rs              # STT traits and types
│   │   └── tts.rs              # TTS traits and types
│   └── Cargo.toml
├── debabelizer-utils/           # Utility functions
│   └── src/lib.rs
├── debabelizer-python/          # Python bindings (PyO3)
│   ├── src/lib.rs              # PyO3 wrapper implementation
│   └── python/debabelizer/     # Python package
└── providers/                   # Provider implementations
    ├── soniox/                 # Soniox STT provider
    ├── elevenlabs/             # ElevenLabs TTS provider
    ├── deepgram/               # Deepgram STT provider
    ├── openai/                 # OpenAI TTS provider
    ├── google/                 # Google Cloud STT/TTS
    ├── azure/                  # Azure Cognitive Services
    └── whisper/                # Local Whisper STT

```

## 🦀 Rust Features Employed

### 1. **Async/Await with Tokio**
The entire library is built on async foundations for non-blocking I/O operations:
```rust
#[async_trait]
pub trait SttProvider: Send + Sync {
    async fn transcribe(&self, audio: &AudioData) -> Result<TranscriptionResult>;
    async fn create_stream(&self, config: &StreamConfig) -> Result<Box<dyn SttStream>>;
}
```

### 2. **Trait-Based Abstraction**
Core functionality is defined through traits, enabling polymorphism and extensibility:
```rust
// Provider traits enable runtime polymorphism
pub trait TtsProvider: Send + Sync {
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult>;
    async fn list_voices(&self) -> Result<Vec<Voice>>;
}
```

### 3. **Zero-Cost Abstractions**
Smart pointers and ownership system for memory safety without runtime overhead:
```rust
pub struct VoiceProcessor {
    stt_provider: Option<Arc<dyn SttProvider>>,
    tts_provider: Option<Arc<dyn TtsProvider>>,
    config: Arc<DebabelizerConfig>,
    session_manager: Arc<SessionManager>,
}
```

### 4. **Error Handling with `thiserror`**
Type-safe error handling with automatic error conversions:
```rust
#[derive(Debug, thiserror::Error)]
pub enum DebabelizerError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    
    #[error("Audio format error: {0}")]
    AudioFormat(String),
}
```

### 5. **Serde for Serialization**
Automatic serialization/deserialization for configuration and data structures:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub language_detected: Option<String>,
    pub duration: Option<f32>,
    pub words: Option<Vec<WordTiming>>,
}
```

### 6. **Builder Pattern**
Ergonomic API design with builder patterns:
```rust
let processor = VoiceProcessor::builder()
    .with_config(config)
    .with_stt_provider("deepgram")
    .with_tts_provider("openai")
    .build()
    .await?;
```

### 7. **Interior Mutability with Arc<Mutex<T>>**
Thread-safe shared state management:
```rust
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<Uuid, Session>>>,
    config: Arc<DebabelizerConfig>,
}
```

### 8. **Feature Flags**
Conditional compilation for optional dependencies:
```toml
[features]
default = []
soniox = ["dep:soniox-provider"]
elevenlabs = ["dep:elevenlabs-provider"]
all-providers = ["soniox", "elevenlabs", "openai", "deepgram", "google", "azure", "whisper"]
```

### 9. **WebSocket Streaming with `tokio-tungstenite`**
Real-time bidirectional communication for streaming:
```rust
pub struct DeepgramStream {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    session_id: Uuid,
}

impl Stream for DeepgramStream {
    type Item = Result<StreamingResult>;
    // Stream implementation for real-time transcription
}
```

### 10. **Workspace Dependencies**
Centralized dependency management across crates:
```toml
[workspace.dependencies]
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"
serde = { version = "1.0", features = ["derive"] }
```

## 🚀 Getting Started

### Prerequisites

- Rust 1.80+ (install via [rustup](https://rustup.rs/))
- OpenSSL development libraries
- Python 3.8+ (for Python bindings)

### Installation

```bash
# Clone the repository
git clone https://github.com/techwiz42/debabelizer.git
cd debabelizer

# Build the entire workspace
cargo build --release

# Run tests
cargo test

# Build with specific providers only
cargo build --features="deepgram,openai"

# Build with all providers
cargo build --features="all-providers"
```

### Basic Usage

```rust
use debabelizer::{VoiceProcessor, DebabelizerConfig, AudioData, AudioFormat};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize from environment variables
    let config = DebabelizerConfig::from_env()?;
    
    // Create processor with auto-selected providers
    let processor = VoiceProcessor::new(config).await?;
    
    // Load audio file
    let audio_data = std::fs::read("speech.wav")?;
    let audio = AudioData::new(
        audio_data,
        AudioFormat::wav(16000)
    );
    
    // Transcribe audio
    let result = processor.transcribe(&audio).await?;
    println!("Transcription: {}", result.text);
    println!("Confidence: {}", result.confidence);
    
    // Synthesize speech
    let synthesis_result = processor.synthesize(
        "Hello from Rust!",
        &Default::default()
    ).await?;
    
    // Save audio
    std::fs::write("output.mp3", synthesis_result.audio_data)?;
    
    Ok(())
}
```

### Configuration

Set environment variables for provider API keys:

```bash
# Provider API Keys
export DEEPGRAM_API_KEY="your-deepgram-key"
export OPENAI_API_KEY="your-openai-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google-credentials.json"
export AZURE_SPEECH_KEY="your-azure-key"
export AZURE_SPEECH_REGION="your-region"

# Provider Preferences
export DEBABELIZER_STT_PROVIDER="deepgram"
export DEBABELIZER_TTS_PROVIDER="openai"
export DEBABELIZER_AUTO_SELECT="true"
export DEBABELIZER_OPTIMIZE_FOR="quality"  # cost|latency|quality|balanced
```

Or use a configuration file (`config.toml`):

```toml
[preferences]
stt_provider = "deepgram"
tts_provider = "elevenlabs"
auto_select = true
optimize_for = "balanced"

[deepgram]
api_key = "your-api-key"
model = "nova-2"
language = "en-US"

[openai]
api_key = "your-api-key"
model = "tts-1-hd"
voice = "alloy"
```

## 🔌 Provider Support

### Speech-to-Text (STT) Providers

| Provider | Streaming | Languages | Key Features |
|----------|-----------|-----------|--------------|
| **Deepgram** | ✅ WebSocket | 40+ | Nova-2 model, <300ms latency, real-time streaming |
| **Google Cloud** | ✅ Simulated | 125+ | Advanced punctuation, speaker diarization |
| **Azure** | ✅ WebSocket | 140+ | Custom models, speaker identification |
| **Soniox** | ✅ WebSocket | Multiple | Low-latency, telephony optimized |
| **Whisper** | ❌ Batch | 99+ | Local processing, zero API costs |

### Text-to-Speech (TTS) Providers

| Provider | Voices | Key Features |
|----------|--------|--------------|
| **ElevenLabs** | 1000+ | Voice cloning, ultra-realistic voices |
| **OpenAI** | 6 | TTS-1/TTS-1-HD models, multiple formats |
| **Google Cloud** | 220+ | WaveNet/Neural2, SSML support |
| **Azure** | 300+ | Neural voices, custom voice creation |

## 🐍 Python Bindings

The library includes Python bindings via PyO3 for easy integration with Python applications:

```python
import debabelizer

# Create processor
processor = debabelizer.VoiceProcessor()

# Transcribe audio
with open("audio.wav", "rb") as f:
    audio = debabelizer.AudioData(
        f.read(),
        debabelizer.AudioFormat("wav", 16000, 1, 16)
    )
    
result = processor.transcribe(audio)
print(f"Text: {result.text}")

# Synthesize speech
options = debabelizer.SynthesisOptions(voice="alloy", speed=1.0)
result = processor.synthesize("Hello from Python!", options)

with open("output.mp3", "wb") as f:
    f.write(result.audio_data)
```

### Building Python Wheels

```bash
cd debabelizer-python
pip install maturin

# Development build
maturin develop --release

# Build wheel for distribution
maturin build --release

# Install the wheel
pip install target/wheels/debabelizer-*.whl
```

## 🧪 Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p debabelizer-core

# Run tests with specific features
cargo test --features="deepgram,openai"

# Run tests with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

## 🏗️ Architecture Highlights

### Provider Registry
Dynamic provider registration and selection based on availability and user preferences:

```rust
pub struct ProviderRegistry {
    pub stt_providers: Vec<(String, Arc<dyn SttProvider>)>,
    pub tts_providers: Vec<(String, Arc<dyn TtsProvider>)>,
}
```

### Session Management
Automatic session lifecycle management with cleanup:

```rust
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<Uuid, Session>>>,
    cleanup_interval: Duration,
}
```

### Stream Processing
Efficient audio streaming with backpressure handling:

```rust
impl Stream for SttStreamWrapper {
    type Item = Result<StreamingResult>;
    
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Efficient async stream processing
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT OR Apache-2.0 license - see the LICENSE-MIT and LICENSE-APACHE files for details.

## 🙏 Acknowledgments

- Built with love using Rust's amazing ecosystem
- Inspired by the need for a unified voice processing interface
- Thanks to all the provider APIs that make this possible

## 📚 Resources

- [API Documentation](https://docs.rs/debabelizer)
- [Examples](./examples)
- [Python Package](./debabelizer-python)
- [Original Python Implementation](https://github.com/techwiz42/debabelizer/tree/main-python)