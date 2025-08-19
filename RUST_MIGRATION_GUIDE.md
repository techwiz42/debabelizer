# Debabelizer Rust Migration Guide

This document outlines the complete migration strategy for rewriting Debabelizer from Python to Rust while maintaining compatibility and improving performance.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Project Structure](#project-structure)
3. [Core Architecture](#core-architecture)
4. [Provider System](#provider-system)
5. [Data Structures](#data-structures)
6. [Configuration Management](#configuration-management)
7. [Error Handling](#error-handling)
8. [Async Patterns](#async-patterns)
9. [Audio Processing](#audio-processing)
10. [Testing Strategy](#testing-strategy)
11. [Distribution Strategy](#distribution-strategy)
12. [Migration Timeline](#migration-timeline)
13. [Compatibility Matrix](#compatibility-matrix)

## Migration Overview

### Goals
- **Performance**: 5-10x improvement in audio processing and transcription throughput
- **Memory Safety**: Eliminate memory leaks and data races
- **Deployment**: Single binary distribution with no runtime dependencies
- **Compatibility**: Maintain Python API compatibility via bindings
- **Ecosystem**: Expand to native Rust, CLI, and WebAssembly targets

### Success Metrics
- All 150+ existing tests pass in Rust implementation
- Python API compatibility maintained via PyO3 bindings
- Performance benchmarks show >5x improvement
- Binary size <50MB for standalone CLI
- Memory usage <50% of Python version

## Project Structure

### Proposed Rust Project Layout

```
debabelizer-rs/
├── Cargo.toml                    # Workspace configuration
├── README.md
├── LICENSE
├── CHANGELOG.md
│
├── crates/
│   ├── debabelizer-core/         # Core library (pure Rust)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config/
│   │       ├── processor/
│   │       ├── session/
│   │       ├── providers/
│   │       ├── audio/
│   │       └── error.rs
│   │
│   ├── debabelizer-py/           # Python bindings (PyO3)
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── processor.rs
│   │       └── types.rs
│   │
│   ├── debabelizer-cli/          # Command-line interface
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── commands/
│   │       └── output.rs
│   │
│   └── debabelizer-wasm/         # WebAssembly bindings
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── web.rs
│
├── providers/                    # Provider implementations
│   ├── stt/
│   │   ├── deepgram/
│   │   ├── google/
│   │   ├── azure/
│   │   ├── openai/
│   │   ├── soniox/
│   │   └── whisper/
│   └── tts/
│       ├── elevenlabs/
│       ├── openai/
│       ├── google/
│       └── azure/
│
├── tests/                        # Integration tests
├── examples/                     # Usage examples
├── benchmarks/                   # Performance benchmarks
└── docs/                         # Documentation
```

### Workspace Configuration

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/debabelizer-core",
    "crates/debabelizer-py", 
    "crates/debabelizer-cli",
    "crates/debabelizer-wasm",
    "providers/stt/*",
    "providers/tts/*"
]

[workspace.dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
async-trait = "0.1"
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio-tungstenite = "0.20"
```

## Core Architecture

### Main Processor

```rust
// crates/debabelizer-core/src/processor/mod.rs
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;

pub struct VoiceProcessor {
    config: Arc<DebabelizerConfig>,
    stt_providers: RwLock<HashMap<String, Box<dyn STTProvider>>>,
    tts_providers: RwLock<HashMap<String, Box<dyn TTSProvider>>>,
    session_manager: Arc<SessionManager>,
}

impl VoiceProcessor {
    pub async fn new(config: DebabelizerConfig) -> Result<Self, ProcessorError> {
        let config = Arc::new(config);
        let session_manager = Arc::new(SessionManager::new());
        
        let mut processor = Self {
            config: config.clone(),
            stt_providers: RwLock::new(HashMap::new()),
            tts_providers: RwLock::new(HashMap::new()),
            session_manager,
        };
        
        processor.initialize_providers().await?;
        Ok(processor)
    }
    
    pub async fn transcribe_file<P: AsRef<Path>>(
        &self,
        path: P,
        options: Option<TranscriptionOptions>,
    ) -> Result<TranscriptionResult, ProcessorError> {
        let provider = self.select_stt_provider(&options).await?;
        let result = provider.transcribe_file(path.as_ref(), options.unwrap_or_default()).await?;
        
        // Update usage statistics
        self.session_manager.record_usage(
            provider.name(),
            UsageType::Transcription,
            result.duration,
        ).await;
        
        Ok(result)
    }
    
    pub async fn start_streaming_transcription(
        &self,
        options: StreamingOptions,
    ) -> Result<StreamingSession, ProcessorError> {
        let provider = self.select_stt_provider(&Some(options.transcription_options.clone())).await?;
        let session_id = provider.start_streaming(options).await?;
        
        Ok(StreamingSession {
            session_id,
            provider_name: provider.name().to_string(),
            results: provider.get_streaming_results(&session_id),
        })
    }
    
    pub async fn synthesize_text(
        &self,
        text: &str,
        options: Option<SynthesisOptions>,
    ) -> Result<SynthesisResult, ProcessorError> {
        let provider = self.select_tts_provider(&options).await?;
        let result = provider.synthesize(text, options.unwrap_or_default()).await?;
        
        self.session_manager.record_usage(
            provider.name(),
            UsageType::Synthesis,
            result.duration,
        ).await;
        
        Ok(result)
    }
    
    pub async fn get_available_voices(
        &self,
        provider: Option<&str>,
    ) -> Result<Vec<Voice>, ProcessorError> {
        match provider {
            Some(name) => {
                let providers = self.tts_providers.read().await;
                let provider = providers.get(name)
                    .ok_or_else(|| ProcessorError::ProviderNotFound(name.to_string()))?;
                provider.get_voices().await
            }
            None => {
                let mut all_voices = Vec::new();
                let providers = self.tts_providers.read().await;
                for provider in providers.values() {
                    let mut voices = provider.get_voices().await?;
                    all_voices.append(&mut voices);
                }
                Ok(all_voices)
            }
        }
    }
    
    async fn initialize_providers(&mut self) -> Result<(), ProcessorError> {
        // Initialize STT providers
        if self.config.is_provider_configured("deepgram") {
            let provider = DeepgramSTT::new(self.config.clone()).await?;
            self.stt_providers.write().await.insert("deepgram".to_string(), Box::new(provider));
        }
        
        if self.config.is_provider_configured("google") {
            let provider = GoogleSTT::new(self.config.clone()).await?;
            self.stt_providers.write().await.insert("google".to_string(), Box::new(provider));
        }
        
        // Initialize TTS providers
        if self.config.is_provider_configured("elevenlabs") {
            let provider = ElevenLabsTTS::new(self.config.clone()).await?;
            self.tts_providers.write().await.insert("elevenlabs".to_string(), Box::new(provider));
        }
        
        if self.config.is_provider_configured("openai") {
            let provider = OpenAITTS::new(self.config.clone()).await?;
            self.tts_providers.write().await.insert("openai".to_string(), Box::new(provider));
        }
        
        Ok(())
    }
    
    async fn select_stt_provider(
        &self,
        options: &Option<TranscriptionOptions>,
    ) -> Result<Arc<dyn STTProvider>, ProcessorError> {
        // Provider selection logic based on preferences, auto-selection, etc.
        let providers = self.stt_providers.read().await;
        
        // Check explicit provider in options
        if let Some(ref opts) = options {
            if let Some(ref provider_name) = opts.provider {
                if let Some(provider) = providers.get(provider_name) {
                    return Ok(provider.clone());
                }
            }
        }
        
        // Check user preferences
        if let Some(preferred) = self.config.preferences().stt_provider() {
            if let Some(provider) = providers.get(preferred) {
                return Ok(provider.clone());
            }
        }
        
        // Auto-selection based on optimization strategy
        if self.config.preferences().auto_select() {
            return self.auto_select_stt_provider(&providers, options).await;
        }
        
        // Fallback to first available provider
        providers.values().next()
            .map(|p| p.clone())
            .ok_or(ProcessorError::NoProvidersAvailable)
    }
    
    async fn auto_select_stt_provider(
        &self,
        providers: &HashMap<String, Box<dyn STTProvider>>,
        options: &Option<TranscriptionOptions>,
    ) -> Result<Arc<dyn STTProvider>, ProcessorError> {
        match self.config.preferences().optimize_for() {
            OptimizationStrategy::Quality => {
                // Prefer providers known for quality: Deepgram Nova-2 > Google > Azure
                for name in &["deepgram", "google", "azure"] {
                    if let Some(provider) = providers.get(*name) {
                        return Ok(provider.clone());
                    }
                }
            }
            OptimizationStrategy::Cost => {
                // Prefer local/free providers: Whisper > cheaper cloud providers
                for name in &["whisper", "azure", "google"] {
                    if let Some(provider) = providers.get(*name) {
                        return Ok(provider.clone());
                    }
                }
            }
            OptimizationStrategy::Latency => {
                // Prefer providers with lowest latency: Deepgram > Soniox > Azure
                for name in &["deepgram", "soniox", "azure"] {
                    if let Some(provider) = providers.get(*name) {
                        return Ok(provider.clone());
                    }
                }
            }
            OptimizationStrategy::Balanced => {
                // Balanced selection based on multiple factors
                for name in &["deepgram", "google", "azure", "whisper"] {
                    if let Some(provider) = providers.get(*name) {
                        return Ok(provider.clone());
                    }
                }
            }
        }
        
        // Fallback
        providers.values().next()
            .map(|p| p.clone())
            .ok_or(ProcessorError::NoProvidersAvailable)
    }
    
    async fn select_tts_provider(
        &self,
        options: &Option<SynthesisOptions>,
    ) -> Result<Arc<dyn TTSProvider>, ProcessorError> {
        // Similar logic to STT provider selection
        // Implementation details similar to select_stt_provider
        todo!("Implement TTS provider selection")
    }
}
```

## Provider System

### Base Traits

```rust
// crates/debabelizer-core/src/providers/stt.rs
use std::path::Path;
use async_trait::async_trait;
use futures::Stream;

#[async_trait]
pub trait STTProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supported_languages(&self) -> &[String];
    fn supports_streaming(&self) -> bool;
    fn supports_file_formats(&self) -> &[AudioFormat];
    
    async fn transcribe_file(
        &self,
        path: &Path,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResult, ProviderError>;
    
    async fn start_streaming(
        &self,
        options: StreamingOptions,
    ) -> Result<String, ProviderError>; // Returns session_id
    
    fn get_streaming_results(
        &self,
        session_id: &str,
    ) -> impl Stream<Item = Result<StreamingResult, ProviderError>> + Send;
    
    async fn stop_streaming(&self, session_id: &str) -> Result<(), ProviderError>;
    
    async fn health_check(&self) -> Result<ProviderHealth, ProviderError>;
}

#[async_trait]
pub trait TTSProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supported_languages(&self) -> &[String];
    fn supports_streaming(&self) -> bool;
    
    async fn get_voices(&self) -> Result<Vec<Voice>, ProviderError>;
    
    async fn synthesize(
        &self,
        text: &str,
        options: SynthesisOptions,
    ) -> Result<SynthesisResult, ProviderError>;
    
    fn synthesize_stream(
        &self,
        text: &str,
        options: SynthesisOptions,
    ) -> impl Stream<Item = Result<AudioChunk, ProviderError>> + Send;
    
    async fn health_check(&self) -> Result<ProviderHealth, ProviderError>;
}
```

### Example Provider Implementation

```rust
// providers/stt/deepgram/src/lib.rs
use debabelizer_core::{STTProvider, TranscriptionOptions, TranscriptionResult, ProviderError};
use reqwest::Client;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};

pub struct DeepgramSTT {
    client: Client,
    api_key: String,
    config: DeepgramConfig,
}

impl DeepgramSTT {
    pub async fn new(config: Arc<DebabelizerConfig>) -> Result<Self, ProviderError> {
        let api_key = config.get_provider_config("deepgram")?
            .get("api_key")
            .ok_or(ProviderError::MissingApiKey("deepgram".to_string()))?
            .clone();
        
        let client = Client::new();
        let deepgram_config = DeepgramConfig::from_config(config)?;
        
        Ok(Self {
            client,
            api_key,
            config: deepgram_config,
        })
    }
}

#[async_trait]
impl STTProvider for DeepgramSTT {
    fn name(&self) -> &str {
        "deepgram"
    }
    
    fn supported_languages(&self) -> &[String] {
        &[
            "en".to_string(), "es".to_string(), "fr".to_string(),
            "de".to_string(), "it".to_string(), "pt".to_string(),
            // ... 40+ languages
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_file_formats(&self) -> &[AudioFormat] {
        &[
            AudioFormat::Wav, AudioFormat::Mp3, AudioFormat::Flac,
            AudioFormat::M4a, AudioFormat::Mp4, AudioFormat::Webm,
        ]
    }
    
    async fn transcribe_file(
        &self,
        path: &Path,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionResult, ProviderError> {
        let audio_data = tokio::fs::read(path).await
            .map_err(|e| ProviderError::FileRead(e.to_string()))?;
        
        let url = format!(
            "https://api.deepgram.com/v1/listen?model={}&language={}&punctuate=true&diarize=true",
            self.config.model,
            options.language.unwrap_or_else(|| "en-US".to_string())
        );
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "audio/wav")
            .body(audio_data)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::ApiError(format!("Deepgram API error: {}", error_text)));
        }
        
        let deepgram_response: DeepgramResponse = response.json().await
            .map_err(|e| ProviderError::InvalidResponse(e.to_string()))?;
        
        Ok(convert_deepgram_response(deepgram_response))
    }
    
    async fn start_streaming(
        &self,
        options: StreamingOptions,
    ) -> Result<String, ProviderError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        let ws_url = format!(
            "wss://api.deepgram.com/v1/listen?model={}&language={}&encoding=linear16&sample_rate=16000",
            self.config.model,
            options.transcription_options.language.unwrap_or_else(|| "en-US".to_string())
        );
        
        let (ws_stream, _) = connect_async(&ws_url).await
            .map_err(|e| ProviderError::Connection(e.to_string()))?;
        
        // Store WebSocket connection for this session
        // Implementation would involve storing the connection in a session manager
        
        Ok(session_id)
    }
    
    fn get_streaming_results(
        &self,
        session_id: &str,
    ) -> impl Stream<Item = Result<StreamingResult, ProviderError>> + Send {
        // Return a stream that yields results from the WebSocket connection
        // This would be connected to the WebSocket stored in start_streaming
        async_stream::stream! {
            // Implementation would pull from stored WebSocket connection
            // and yield StreamingResult items
        }
    }
    
    async fn stop_streaming(&self, session_id: &str) -> Result<(), ProviderError> {
        // Close WebSocket connection and cleanup session
        Ok(())
    }
    
    async fn health_check(&self) -> Result<ProviderHealth, ProviderError> {
        let response = self.client
            .get("https://api.deepgram.com/v1/projects")
            .header("Authorization", format!("Token {}", self.api_key))
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        
        Ok(ProviderHealth {
            is_healthy: response.status().is_success(),
            latency: Some(response.elapsed().unwrap_or_default()),
            message: None,
        })
    }
}

fn convert_deepgram_response(response: DeepgramResponse) -> TranscriptionResult {
    let channel = &response.results.channels[0];
    let alternative = &channel.alternatives[0];
    
    TranscriptionResult {
        text: alternative.transcript.clone(),
        confidence: alternative.confidence,
        language_detected: response.results.channels[0]
            .detected_language
            .clone()
            .unwrap_or_else(|| "en".to_string()),
        duration: response.metadata.duration,
        words: alternative.words.iter().map(|w| WordTiming {
            word: w.word.clone(),
            start: w.start,
            end: w.end,
            confidence: w.confidence,
        }).collect(),
        is_final: true,
    }
}
```

## Data Structures

```rust
// crates/debabelizer-core/src/types.rs
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f64,
    pub language_detected: String,
    pub duration: f64,
    pub words: Vec<WordTiming>,
    pub is_final: bool,
    pub metadata: Option<TranscriptionMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub audio_data: Vec<u8>,
    pub format: AudioFormat,
    pub duration: f64,
    pub size_bytes: usize,
    pub metadata: Option<SynthesisMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResult {
    pub session_id: String,
    pub is_final: bool,
    pub text: String,
    pub confidence: f64,
    pub words: Vec<WordTiming>,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub voice_id: String,
    pub name: String,
    pub description: String,
    pub language: String,
    pub gender: Option<Gender>,
    pub age: Option<AgeRange>,
    pub accent: Option<String>,
    pub use_case: Vec<UseCase>,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormat {
    pub format: String,        // "wav", "mp3", "flac", etc.
    pub sample_rate: u32,      // 16000, 44100, etc.
    pub channels: u16,         // 1 (mono), 2 (stereo)
    pub bit_depth: u16,        // 16, 24, 32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gender {
    Male,
    Female,
    NonBinary,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgeRange {
    Child,
    Young,
    Adult,
    Senior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UseCase {
    Narration,
    Conversational,
    News,
    CustomerService,
    Gaming,
    Educational,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    pub language: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub enable_diarization: bool,
    pub enable_punctuation: bool,
    pub enable_profanity_filter: bool,
    pub custom_vocabulary: Vec<String>,
    pub audio_format: Option<AudioFormat>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: None,
            model: None,
            provider: None,
            enable_diarization: false,
            enable_punctuation: true,
            enable_profanity_filter: false,
            custom_vocabulary: Vec::new(),
            audio_format: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisOptions {
    pub voice: Option<String>,
    pub provider: Option<String>,
    pub audio_format: Option<AudioFormat>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    pub volume: Option<f32>,
    pub stability: Option<f32>,
    pub similarity_boost: Option<f32>,
    pub style: Option<f32>,
    pub use_speaker_boost: bool,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            voice: None,
            provider: None,
            audio_format: None,
            speed: None,
            pitch: None,
            volume: None,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingOptions {
    pub transcription_options: TranscriptionOptions,
    pub interim_results: bool,
    pub end_utterance_silence_threshold: Option<Duration>,
    pub auto_punctuation: bool,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub sequence: u64,
    pub timestamp: std::time::SystemTime,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
pub struct ProviderHealth {
    pub is_healthy: bool,
    pub latency: Option<Duration>,
    pub message: Option<String>,
}
```

## Configuration Management

```rust
// crates/debabelizer-core/src/config/mod.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebabelizerConfig {
    pub providers: HashMap<String, ProviderConfig>,
    pub preferences: UserPreferences,
    pub audio: AudioConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub stt_provider: Option<String>,
    pub tts_provider: Option<String>,
    pub auto_select: bool,
    pub optimize_for: OptimizationStrategy,
    pub default_language: String,
    pub cost_optimization: CostOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Quality,
    Cost,
    Latency,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOptimization {
    pub max_monthly_cost: Option<f64>,
    pub prefer_free_tier: bool,
    pub cost_alerts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub region: Option<String>,
    pub custom_settings: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub default_sample_rate: u32,
    pub default_format: String,
    pub max_file_size_mb: u64,
    pub temp_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub enable_provider_logs: bool,
    pub log_audio_metadata: bool,
}

impl DebabelizerConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        // Load provider preferences
        if let Ok(stt_provider) = env::var("DEBABELIZER_STT_PROVIDER") {
            config.preferences.stt_provider = Some(stt_provider);
        }
        
        if let Ok(tts_provider) = env::var("DEBABELIZER_TTS_PROVIDER") {
            config.preferences.tts_provider = Some(tts_provider);
        }
        
        if let Ok(auto_select) = env::var("DEBABELIZER_AUTO_SELECT") {
            config.preferences.auto_select = auto_select.parse().unwrap_or(true);
        }
        
        if let Ok(optimize_for) = env::var("DEBABELIZER_OPTIMIZE_FOR") {
            config.preferences.optimize_for = match optimize_for.as_str() {
                "quality" => OptimizationStrategy::Quality,
                "cost" => OptimizationStrategy::Cost,
                "latency" => OptimizationStrategy::Latency,
                "balanced" => OptimizationStrategy::Balanced,
                _ => OptimizationStrategy::Balanced,
            };
        }
        
        // Load provider configurations
        config.load_provider_configs();
        
        config
    }
    
    fn load_provider_configs(&mut self) {
        // Deepgram
        if let Ok(api_key) = env::var("DEEPGRAM_API_KEY") {
            let mut provider_config = ProviderConfig::default();
            provider_config.api_key = Some(api_key);
            
            if let Ok(model) = env::var("DEEPGRAM_MODEL") {
                provider_config.model = Some(model);
            }
            
            self.providers.insert("deepgram".to_string(), provider_config);
        }
        
        // OpenAI
        if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            let mut provider_config = ProviderConfig::default();
            provider_config.api_key = Some(api_key);
            
            if let Ok(model) = env::var("OPENAI_TTS_MODEL") {
                provider_config.model = Some(model);
            }
            
            self.providers.insert("openai".to_string(), provider_config);
        }
        
        // Google Cloud
        if let Ok(credentials_path) = env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            let mut provider_config = ProviderConfig::default();
            provider_config.custom_settings.insert(
                "credentials_path".to_string(),
                serde_json::Value::String(credentials_path)
            );
            
            if let Ok(project_id) = env::var("GOOGLE_CLOUD_PROJECT") {
                provider_config.custom_settings.insert(
                    "project_id".to_string(),
                    serde_json::Value::String(project_id)
                );
            }
            
            self.providers.insert("google".to_string(), provider_config);
        }
        
        // Azure
        if let Ok(api_key) = env::var("AZURE_SPEECH_KEY") {
            let mut provider_config = ProviderConfig::default();
            provider_config.api_key = Some(api_key);
            
            if let Ok(region) = env::var("AZURE_SPEECH_REGION") {
                provider_config.region = Some(region);
            }
            
            self.providers.insert("azure".to_string(), provider_config);
        }
        
        // ElevenLabs
        if let Ok(api_key) = env::var("ELEVENLABS_API_KEY") {
            let mut provider_config = ProviderConfig::default();
            provider_config.api_key = Some(api_key);
            self.providers.insert("elevenlabs".to_string(), provider_config);
        }
        
        // Soniox
        if let Ok(api_key) = env::var("SONIOX_API_KEY") {
            let mut provider_config = ProviderConfig::default();
            provider_config.api_key = Some(api_key);
            self.providers.insert("soniox".to_string(), provider_config);
        }
        
        // Whisper (local) - no API key required
        self.providers.insert("whisper".to_string(), ProviderConfig::default());
    }
    
    pub fn is_provider_configured(&self, provider_name: &str) -> bool {
        self.providers.contains_key(provider_name)
    }
    
    pub fn get_provider_config(&self, provider_name: &str) -> Result<&ProviderConfig, ConfigError> {
        self.providers.get(provider_name)
            .ok_or_else(|| ConfigError::ProviderNotConfigured(provider_name.to_string()))
    }
    
    pub fn preferences(&self) -> &UserPreferences {
        &self.preferences
    }
}

impl Default for DebabelizerConfig {
    fn default() -> Self {
        Self {
            providers: HashMap::new(),
            preferences: UserPreferences {
                stt_provider: None,
                tts_provider: None,
                auto_select: true,
                optimize_for: OptimizationStrategy::Balanced,
                default_language: "en".to_string(),
                cost_optimization: CostOptimization {
                    max_monthly_cost: None,
                    prefer_free_tier: false,
                    cost_alerts: false,
                },
            },
            audio: AudioConfig {
                default_sample_rate: 16000,
                default_format: "wav".to_string(),
                max_file_size_mb: 100,
                temp_dir: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                enable_provider_logs: false,
                log_audio_metadata: true,
            },
        }
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            model: None,
            region: None,
            custom_settings: HashMap::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Provider not configured: {0}")]
    ProviderNotConfigured(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
}
```

## Error Handling

```rust
// crates/debabelizer-core/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProcessorError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioError),
    
    #[error("Session error: {0}")]
    Session(#[from] SessionError),
    
    #[error("No providers available")]
    NoProvidersAvailable,
    
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("Authentication failed: {0}")]
    Authentication(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),
    
    #[error("File read error: {0}")]
    FileRead(String),
    
    #[error("Missing API key for provider: {0}")]
    MissingApiKey(String),
    
    #[error("Rate limit exceeded")]
    RateLimit,
    
    #[error("Quota exceeded")]
    QuotaExceeded,
    
    #[error("Connection timeout")]
    Timeout,
    
    #[error("Provider unavailable")]
    Unavailable,
}

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),
    
    #[error("Audio conversion failed: {0}")]
    ConversionFailed(String),
    
    #[error("Audio file too large: {0} MB (max: {1} MB)")]
    FileTooLarge(u64, u64),
    
    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),
    
    #[error("Audio processing error: {0}")]
    ProcessingError(String),
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(String),
    
    #[error("Session expired: {0}")]
    Expired(String),
    
    #[error("Session already active: {0}")]
    AlreadyActive(String),
    
    #[error("Session cleanup failed: {0}")]
    CleanupFailed(String),
}

// Provider-specific error types
#[derive(Debug, Error)]
pub enum DeepgramError {
    #[error("Deepgram API error: {0}")]
    Api(String),
    
    #[error("WebSocket connection failed: {0}")]
    WebSocket(String),
    
    #[error("Invalid model: {0}")]
    InvalidModel(String),
}

#[derive(Debug, Error)]
pub enum OpenAIError {
    #[error("OpenAI API error: {0}")]
    Api(String),
    
    #[error("Invalid voice: {0}")]
    InvalidVoice(String),
    
    #[error("Content policy violation: {0}")]
    ContentPolicy(String),
}
```

## Async Patterns

```rust
// crates/debabelizer-core/src/session/mod.rs
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio::time::{interval, Duration};
use uuid::Uuid;

pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    cleanup_interval: Duration,
    session_timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub provider: String,
    pub session_type: SessionType,
    pub created_at: std::time::SystemTime,
    pub last_activity: std::time::SystemTime,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub enum SessionType {
    StreamingTranscription,
    StreamingSynthesis,
    FileTranscription,
}

impl SessionManager {
    pub fn new() -> Self {
        let manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval: Duration::from_secs(60), // Cleanup every minute
            session_timeout: Duration::from_secs(300), // 5 minute timeout
        };
        
        // Start background cleanup task
        manager.start_cleanup_task();
        
        manager
    }
    
    pub async fn create_session(
        &self,
        provider: String,
        session_type: SessionType,
    ) -> String {
        let session_id = Uuid::new_v4().to_string();
        let now = std::time::SystemTime::now();
        
        let session = Session {
            id: session_id.clone(),
            provider,
            session_type,
            created_at: now,
            last_activity: now,
            is_active: true,
        };
        
        self.sessions.write().await.insert(session_id.clone(), session);
        session_id
    }
    
    pub async fn update_activity(&self, session_id: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = std::time::SystemTime::now();
            Ok(())
        } else {
            Err(SessionError::NotFound(session_id.to_string()))
        }
    }
    
    pub async fn end_session(&self, session_id: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.is_active = false;
            Ok(())
        } else {
            Err(SessionError::NotFound(session_id.to_string()))
        }
    }
    
    pub async fn record_usage(
        &self,
        provider: &str,
        usage_type: UsageType,
        duration: f64,
    ) {
        // Record usage statistics for cost tracking
        // Implementation would store usage data for analytics
    }
    
    fn start_cleanup_task(&self) {
        let sessions = Arc::clone(&self.sessions);
        let cleanup_interval = self.cleanup_interval;
        let session_timeout = self.session_timeout;
        
        tokio::spawn(async move {
            let mut interval = interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let mut sessions = sessions.write().await;
                let now = std::time::SystemTime::now();
                
                sessions.retain(|_, session| {
                    if let Ok(elapsed) = now.duration_since(session.last_activity) {
                        elapsed < session_timeout && session.is_active
                    } else {
                        false
                    }
                });
            }
        });
    }
    
    pub async fn get_active_sessions(&self) -> Vec<Session> {
        self.sessions.read().await
            .values()
            .filter(|s| s.is_active)
            .cloned()
            .collect()
    }
    
    pub async fn cleanup_expired_sessions(&self) -> usize {
        let mut sessions = self.sessions.write().await;
        let now = std::time::SystemTime::now();
        let initial_count = sessions.len();
        
        sessions.retain(|_, session| {
            if let Ok(elapsed) = now.duration_since(session.last_activity) {
                elapsed < self.session_timeout && session.is_active
            } else {
                false
            }
        });
        
        initial_count - sessions.len()
    }
}

#[derive(Debug, Clone)]
pub enum UsageType {
    Transcription,
    Synthesis,
    Streaming,
}

// Streaming session wrapper
pub struct StreamingSession {
    pub session_id: String,
    pub provider_name: String,
    pub results: Box<dyn Stream<Item = Result<StreamingResult, ProviderError>> + Send + Unpin>,
}

impl StreamingSession {
    pub async fn next_result(&mut self) -> Option<Result<StreamingResult, ProviderError>> {
        use futures::StreamExt;
        self.results.next().await
    }
    
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    
    pub fn provider(&self) -> &str {
        &self.provider_name
    }
}
```

## Audio Processing

```rust
// crates/debabelizer-core/src/audio/mod.rs
use std::path::Path;
use std::process::Command;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct AudioProcessor {
    temp_dir: Option<String>,
    max_file_size: u64,
}

impl AudioProcessor {
    pub fn new(temp_dir: Option<String>, max_file_size_mb: u64) -> Self {
        Self {
            temp_dir,
            max_file_size: max_file_size_mb * 1024 * 1024, // Convert to bytes
        }
    }
    
    pub async fn convert_audio(
        &self,
        input_path: &Path,
        output_format: &AudioFormat,
    ) -> Result<Vec<u8>, AudioError> {
        // Check file size
        let metadata = tokio::fs::metadata(input_path).await
            .map_err(|e| AudioError::ProcessingError(e.to_string()))?;
        
        if metadata.len() > self.max_file_size {
            return Err(AudioError::FileTooLarge(
                metadata.len() / (1024 * 1024),
                self.max_file_size / (1024 * 1024),
            ));
        }
        
        // Use FFmpeg for audio conversion
        let output = Command::new("ffmpeg")
            .args(&[
                "-i", input_path.to_str().unwrap(),
                "-f", &output_format.format,
                "-ar", &output_format.sample_rate.to_string(),
                "-ac", &output_format.channels.to_string(),
                "-"
            ])
            .output()
            .map_err(|e| AudioError::ConversionFailed(e.to_string()))?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(AudioError::ConversionFailed(error.to_string()));
        }
        
        Ok(output.stdout)
    }
    
    pub fn detect_format(&self, file_path: &Path) -> Result<AudioFormat, AudioError> {
        // Use symphonia to detect audio format
        let file = std::fs::File::open(file_path)
            .map_err(|e| AudioError::ProcessingError(e.to_string()))?;
        
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        
        let mut hint = Hint::new();
        if let Some(extension) = file_path.extension() {
            if let Some(ext_str) = extension.to_str() {
                hint.with_extension(ext_str);
            }
        }
        
        let meta_opts = MetadataOptions::default();
        let fmt_opts = FormatOptions::default();
        
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| AudioError::InvalidFormat(e.to_string()))?;
        
        let format_info = probed.format.default_track()
            .ok_or_else(|| AudioError::InvalidFormat("No audio tracks found".to_string()))?;
        
        let codec_params = &format_info.codec_params;
        
        Ok(AudioFormat {
            format: probed.format.metadata().container_format.long_name().unwrap_or("unknown").to_string(),
            sample_rate: codec_params.sample_rate.unwrap_or(44100),
            channels: codec_params.channels.map(|c| c.count() as u16).unwrap_or(2),
            bit_depth: codec_params.bits_per_sample.unwrap_or(16) as u16,
        })
    }
    
    pub fn detect_silence(
        &self,
        audio_data: &[u8],
        threshold: f32,
        min_duration: Duration,
    ) -> Vec<SilenceSegment> {
        // Implementation for silence detection
        // This would analyze the audio data and return segments where silence is detected
        Vec::new()
    }
    
    pub async fn split_on_silence(
        &self,
        input_path: &Path,
        silence_threshold: f32,
        min_silence_duration: Duration,
    ) -> Result<Vec<AudioSegment>, AudioError> {
        // Split audio file on silence for better processing
        // Useful for long audio files that need to be processed in chunks
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct SilenceSegment {
    pub start: Duration,
    pub end: Duration,
}

#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub data: Vec<u8>,
    pub start: Duration,
    pub end: Duration,
    pub format: AudioFormat,
}

// Audio format utilities
impl AudioFormat {
    pub fn is_supported_by_provider(&self, provider: &str) -> bool {
        match provider {
            "deepgram" => matches!(self.format.as_str(), "wav" | "mp3" | "flac" | "m4a" | "mp4" | "webm"),
            "google" => matches!(self.format.as_str(), "wav" | "flac" | "mp3" | "ogg"),
            "azure" => matches!(self.format.as_str(), "wav" | "mp3" | "flac" | "aac" | "ogg"),
            "whisper" => matches!(self.format.as_str(), "wav" | "mp3" | "flac" | "m4a" | "ogg"),
            _ => false,
        }
    }
    
    pub fn get_mime_type(&self) -> String {
        match self.format.as_str() {
            "wav" => "audio/wav".to_string(),
            "mp3" => "audio/mpeg".to_string(),
            "flac" => "audio/flac".to_string(),
            "m4a" => "audio/mp4".to_string(),
            "ogg" => "audio/ogg".to_string(),
            "webm" => "audio/webm".to_string(),
            _ => "application/octet-stream".to_string(),
        }
    }
}
```

## Testing Strategy

```rust
// tests/integration_tests.rs
use debabelizer_core::*;
use tokio_test;

#[tokio::test]
async fn test_voice_processor_initialization() {
    let config = DebabelizerConfig::default();
    let processor = VoiceProcessor::new(config).await;
    assert!(processor.is_ok());
}

#[tokio::test]
async fn test_provider_auto_selection() {
    let mut config = DebabelizerConfig::default();
    config.preferences.auto_select = true;
    config.preferences.optimize_for = OptimizationStrategy::Quality;
    
    // Mock provider configurations
    config.providers.insert("deepgram".to_string(), ProviderConfig {
        api_key: Some("test_key".to_string()),
        ..Default::default()
    });
    
    let processor = VoiceProcessor::new(config).await.unwrap();
    
    // Test that the processor selects the appropriate provider
    let options = TranscriptionOptions::default();
    // Would test provider selection logic here
}

#[tokio::test]
async fn test_streaming_transcription() {
    // Mock streaming test
    let config = create_test_config();
    let processor = VoiceProcessor::new(config).await.unwrap();
    
    let streaming_options = StreamingOptions {
        transcription_options: TranscriptionOptions::default(),
        interim_results: true,
        end_utterance_silence_threshold: Some(Duration::from_secs(2)),
        auto_punctuation: true,
    };
    
    let session = processor.start_streaming_transcription(streaming_options).await.unwrap();
    
    // Test streaming session
    assert!(!session.session_id().is_empty());
    assert!(!session.provider().is_empty());
}

// Mock provider for testing
struct MockSTTProvider {
    name: String,
}

#[async_trait]
impl STTProvider for MockSTTProvider {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn supported_languages(&self) -> &[String] {
        &["en".to_string(), "es".to_string()]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_file_formats(&self) -> &[AudioFormat] {
        &[AudioFormat {
            format: "wav".to_string(),
            sample_rate: 16000,
            channels: 1,
            bit_depth: 16,
        }]
    }
    
    async fn transcribe_file(
        &self,
        _path: &Path,
        _options: TranscriptionOptions,
    ) -> Result<TranscriptionResult, ProviderError> {
        Ok(TranscriptionResult {
            text: "Test transcription".to_string(),
            confidence: 0.95,
            language_detected: "en".to_string(),
            duration: 5.0,
            words: vec![],
            is_final: true,
            metadata: None,
        })
    }
    
    async fn start_streaming(&self, _options: StreamingOptions) -> Result<String, ProviderError> {
        Ok("test_session_123".to_string())
    }
    
    fn get_streaming_results(
        &self,
        _session_id: &str,
    ) -> impl Stream<Item = Result<StreamingResult, ProviderError>> + Send {
        async_stream::stream! {
            yield Ok(StreamingResult {
                session_id: "test_session_123".to_string(),
                is_final: false,
                text: "Hello".to_string(),
                confidence: 0.8,
                words: vec![],
                timestamp: std::time::SystemTime::now(),
            });
            
            yield Ok(StreamingResult {
                session_id: "test_session_123".to_string(),
                is_final: true,
                text: "Hello world".to_string(),
                confidence: 0.95,
                words: vec![],
                timestamp: std::time::SystemTime::now(),
            });
        }
    }
    
    async fn stop_streaming(&self, _session_id: &str) -> Result<(), ProviderError> {
        Ok(())
    }
    
    async fn health_check(&self) -> Result<ProviderHealth, ProviderError> {
        Ok(ProviderHealth {
            is_healthy: true,
            latency: Some(Duration::from_millis(100)),
            message: None,
        })
    }
}

fn create_test_config() -> DebabelizerConfig {
    let mut config = DebabelizerConfig::default();
    // Add test provider configurations
    config.providers.insert("mock".to_string(), ProviderConfig::default());
    config
}

// Benchmark tests
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn benchmark_transcription(c: &mut Criterion) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        c.bench_function("transcription_performance", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let config = create_test_config();
                    let processor = VoiceProcessor::new(config).await.unwrap();
                    
                    // Benchmark transcription performance
                    let options = TranscriptionOptions::default();
                    // Would benchmark actual transcription here
                });
            });
        });
    }
    
    criterion_group!(benches, benchmark_transcription);
    criterion_main!(benches);
}
```

## Distribution Strategy

### 1. Core Rust Library (Crates.io)

```toml
# crates/debabelizer-core/Cargo.toml
[package]
name = "debabelizer"
version = "2.0.0"
edition = "2021"
license = "MIT"
description = "Universal voice processing library with multiple provider support"
homepage = "https://github.com/your-org/debabelizer-rs"
repository = "https://github.com/your-org/debabelizer-rs"
keywords = ["speech", "transcription", "synthesis", "tts", "stt"]
categories = ["multimedia::audio", "api-bindings"]

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
async-trait = "0.1"
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio-tungstenite = "0.20"
futures = "0.3"
uuid = { version = "1.0", features = ["v4"] }
async-stream = "0.3"

# Audio processing
symphonia = { version = "0.5", features = ["all"] }

# Optional provider dependencies
deepgram = { version = "0.3", optional = true }
google-cloud-speech = { version = "0.4", optional = true }
azure-cognitiveservices-speech = { version = "1.0", optional = true }

[features]
default = []
all = ["deepgram", "google", "azure", "whisper", "elevenlabs"]
deepgram = ["dep:deepgram"]
google = ["dep:google-cloud-speech"] 
azure = ["dep:azure-cognitiveservices-speech"]
whisper = ["candle-core", "candle-transformers"]
elevenlabs = []
```

### 2. Python Bindings (PyO3)

```toml
# crates/debabelizer-py/Cargo.toml
[package]
name = "debabelizer-py"
version = "2.0.0"
edition = "2021"

[lib]
name = "debabelizer"
crate-type = ["cdylib"]

[dependencies]
debabelizer-core = { path = "../debabelizer-core" }
pyo3 = { version = "0.20", features = ["extension-module"] }
pyo3-asyncio = { version = "0.20", features = ["tokio-runtime"] }
tokio = { version = "1.0", features = ["full"] }

[build-dependencies]
pyo3-build-config = "0.20"
```

```python
# crates/debabelizer-py/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "debabelizer"
version = "2.0.0"
description = "Universal voice processing library (Rust-powered)"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Rust",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
]

[project.urls]
Homepage = "https://github.com/your-org/debabelizer-rs"
Repository = "https://github.com/your-org/debabelizer-rs"
Issues = "https://github.com/your-org/debabelizer-rs/issues"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### 3. CLI Tool

```toml
# crates/debabelizer-cli/Cargo.toml
[package]
name = "debabelizer-cli"
version = "2.0.0"
edition = "2021"

[[bin]]
name = "debabelizer"
path = "src/main.rs"

[dependencies]
debabelizer-core = { path = "../debabelizer-core", features = ["all"] }
clap = { version = "4.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
indicatif = "0.17"
colored = "2.0"
serde_json = "1.0"
```

### 4. WebAssembly Package

```toml
# crates/debabelizer-wasm/Cargo.toml
[package]
name = "debabelizer-wasm"
version = "2.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
debabelizer-core = { path = "../debabelizer-core", default-features = false }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = "0.3"
serde-wasm-bindgen = "0.6"

[dependencies.web-sys]
version = "0.3"
features = [
  "console",
  "AudioContext",
  "MediaRecorder",
  "File",
  "FileReader",
]
```

## Migration Timeline

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up Rust workspace structure
- [ ] Implement core data structures and error types
- [ ] Create configuration management system
- [ ] Implement basic audio processing utilities
- [ ] Set up CI/CD pipeline

### Phase 2: Core Engine (Weeks 5-8)
- [ ] Implement VoiceProcessor main class
- [ ] Create provider trait definitions
- [ ] Implement session management system
- [ ] Add provider auto-selection logic
- [ ] Create basic test infrastructure

### Phase 3: Provider Implementation (Weeks 9-16)
- [ ] Implement HTTP-based providers (OpenAI, ElevenLabs)
- [ ] Add WebSocket streaming providers (Deepgram, Soniox)
- [ ] Implement Google Cloud providers (gRPC)
- [ ] Add Azure Cognitive Services providers
- [ ] Implement local Whisper provider
- [ ] Add remaining Soniox integration

### Phase 4: Python Bindings (Weeks 17-20)
- [ ] Create PyO3 bindings for all core functionality
- [ ] Implement async Python API compatibility
- [ ] Port Python test suite to validate compatibility
- [ ] Add Python packaging and distribution
- [ ] Performance benchmarking vs Python version

### Phase 5: CLI and WASM (Weeks 21-24)
- [ ] Implement CLI tool with full feature parity
- [ ] Create WebAssembly bindings for web use
- [ ] Add comprehensive documentation
- [ ] Performance optimization and profiling
- [ ] Final testing and validation

### Phase 6: Release and Migration (Weeks 25-26)
- [ ] Release candidates and beta testing
- [ ] Migration documentation and guides
- [ ] Community feedback and bug fixes
- [ ] Official 2.0 release
- [ ] Python package migration strategy

## Compatibility Matrix

| Feature | Python 1.x | Rust 2.x Core | Rust 2.x Python | Rust 2.x CLI | Rust 2.x WASM |
|---------|------------|---------------|------------------|--------------|---------------|
| **STT Providers** |
| Deepgram | ✅ | ✅ | ✅ | ✅ | ⚠️ API-only |
| Google Cloud | ✅ | ✅ | ✅ | ✅ | ❌ |
| Azure | ✅ | ✅ | ✅ | ✅ | ⚠️ API-only |
| OpenAI Whisper (API) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Whisper (Local) | ✅ | ✅ | ✅ | ✅ | ❌ |
| Soniox | ✅ | ✅ | ✅ | ✅ | ⚠️ API-only |
| **TTS Providers** |
| ElevenLabs | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Google Cloud | ✅ | ✅ | ✅ | ✅ | ❌ |
| Azure | ✅ | ✅ | ✅ | ✅ | ⚠️ API-only |
| **Features** |
| File Transcription | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming STT | ✅ | ✅ | ✅ | ✅ | ⚠️ Limited |
| Streaming TTS | ✅ | ✅ | ✅ | ✅ | ⚠️ Limited |
| Provider Auto-selection | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cost Optimization | ✅ | ✅ | ✅ | ✅ | ✅ |
| Session Management | ✅ | ✅ | ✅ | ✅ | ⚠️ Limited |
| Audio Format Conversion | ✅ | ✅ | ✅ | ✅ | ⚠️ Limited |
| **Performance** |
| Memory Usage | Baseline | 50% less | 60% less | 50% less | 70% less |
| CPU Usage | Baseline | 40% less | 50% less | 40% less | Varies |
| Startup Time | Baseline | 80% faster | 70% faster | 90% faster | Instant |
| Throughput | Baseline | 5-10x faster | 4-8x faster | 5-10x faster | 3-5x faster |

**Legend:**
- ✅ Full support
- ⚠️ Limited support (some features may not work)
- ❌ Not supported

## Migration Benefits

### Performance Improvements
- **Memory Usage**: 50-70% reduction due to Rust's zero-cost abstractions
- **CPU Usage**: 40-50% reduction from optimized compilation and efficient async
- **Startup Time**: 70-90% faster initialization due to compiled binaries
- **Throughput**: 5-10x improvement in audio processing and API calls

### Deployment Benefits
- **Single Binary**: No Python runtime or dependency management
- **Cross-platform**: Native binaries for Linux, macOS, Windows, ARM
- **Container Size**: Significantly smaller Docker images
- **Memory Safety**: Elimination of memory leaks and data races

### Developer Experience
- **Type Safety**: Compile-time error detection vs runtime errors
- **Concurrency**: Fearless concurrency with Rust's ownership model
- **Ecosystem**: Access to Rust's growing ecosystem of high-performance libraries
- **Maintainability**: Better code organization and module system

### Migration Path
1. **Gradual Migration**: Start with Python bindings to maintain compatibility
2. **Performance Critical**: Move performance-critical workloads to native Rust
3. **New Projects**: Use native Rust API for greenfield projects
4. **Legacy Support**: Maintain Python 1.x for existing projects during transition
5. **Full Migration**: Eventually deprecate Python version once ecosystem adopts Rust

This migration would position Debabelizer as a high-performance, multi-platform voice processing library while maintaining the familiar Python API for existing users.