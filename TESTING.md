# Testing Guide for Debabelizer Rust

This document provides comprehensive guidance on testing the Debabelizer voice processing library, covering unit tests, integration tests, mocking strategies, and best practices.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Mocking and Test Doubles](#mocking-and-test-doubles)
5. [Async Testing](#async-testing)
6. [Provider Testing](#provider-testing)
7. [Test Configuration](#test-configuration)
8. [Running Tests](#running-tests)
9. [Coverage and Quality](#coverage-and-quality)
10. [Best Practices](#best-practices)

## Testing Overview

Debabelizer uses Rust's built-in testing framework along with additional testing libraries for comprehensive test coverage:

### Testing Dependencies

```toml
[dev-dependencies]
tokio-test = "0.4"
mockall = "0.13"
tempfile = "3.12"
criterion = "0.5"
proptest = "1.0"
wiremock = "0.6"
serial_test = "3.1"
```

### Test Structure

```
debabelizer/
├── debabelizer-core/
│   └── src/
│       ├── lib.rs           # Unit tests inline
│       ├── stt.rs           # Unit tests inline
│       └── tts.rs           # Unit tests inline
├── debabelizer/
│   ├── src/
│   │   ├── lib.rs           # Unit tests inline
│   │   ├── config.rs        # Unit tests inline
│   │   └── processor.rs     # Unit tests inline
│   └── tests/
│       ├── integration_tests.rs
│       ├── config_tests.rs
│       └── common/
│           └── mod.rs       # Test utilities
├── providers/
│   ├── soniox/
│   │   ├── src/lib.rs       # Unit tests inline
│   │   └── tests/
│   │       └── integration.rs
│   └── elevenlabs/
│       ├── src/lib.rs       # Unit tests inline
│       └── tests/
│           └── integration.rs
└── tests/                   # Workspace-level integration tests
    ├── end_to_end.rs
    └── provider_compatibility.rs
```

## Unit Testing

### Basic Unit Tests

Unit tests are placed in the same file as the code they test, using the `#[cfg(test)]` attribute:

```rust
// src/audio.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub format: String,
    pub sample_rate: u32,
    pub channels: u8,
    pub bit_depth: Option<u16>,
}

impl AudioFormat {
    pub fn wav(sample_rate: u32) -> Self {
        Self {
            format: "wav".to_string(),
            sample_rate,
            channels: 1,
            bit_depth: Some(16),
        }
    }
    
    pub fn is_compressed(&self) -> bool {
        matches!(self.format.as_str(), "mp3" | "opus" | "aac")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_format_creation() {
        let format = AudioFormat::wav(16000);
        assert_eq!(format.format, "wav");
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, Some(16));
    }

    #[test]
    fn test_compression_detection() {
        let wav = AudioFormat::wav(16000);
        let mp3 = AudioFormat {
            format: "mp3".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: None,
        };

        assert!(!wav.is_compressed());
        assert!(mp3.is_compressed());
    }

    #[test]
    fn test_format_equality() {
        let format1 = AudioFormat::wav(16000);
        let format2 = AudioFormat::wav(16000);
        let format3 = AudioFormat::wav(44100);

        assert_eq!(format1, format2);
        assert_ne!(format1, format3);
    }
}
```

### Testing Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },
    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },
}

pub fn validate_sample_rate(rate: u32) -> Result<u32, ConfigError> {
    match rate {
        8000 | 16000 | 22050 | 44100 | 48000 => Ok(rate),
        _ => Err(ConfigError::InvalidValue {
            field: "sample_rate".to_string(),
            value: rate.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_sample_rates() {
        assert_eq!(validate_sample_rate(16000), Ok(16000));
        assert_eq!(validate_sample_rate(44100), Ok(44100));
    }

    #[test]
    fn test_invalid_sample_rate() {
        let result = validate_sample_rate(12345);
        assert!(result.is_err());
        
        if let Err(ConfigError::InvalidValue { field, value }) = result {
            assert_eq!(field, "sample_rate");
            assert_eq!(value, "12345");
        } else {
            panic!("Expected InvalidValue error");
        }
    }
}
```

## Integration Testing

### Provider Integration Tests

```rust
// tests/provider_integration.rs
use debabelizer::{VoiceProcessor, AudioData, AudioFormat, SynthesisOptions};
use tempfile::NamedTempFile;
use std::fs;

#[tokio::test]
async fn test_stt_provider_integration() {
    let processor = VoiceProcessor::builder()
        .stt_provider("soniox")
        .build()
        .await
        .expect("Failed to create processor");

    // Create test audio data
    let audio_data = create_test_wav_data();
    let audio = AudioData::new(audio_data, AudioFormat::wav(16000));

    // Test transcription
    let result = processor.transcribe(audio).await;
    
    // Note: This will fail without valid API keys, so we check the error type
    match result {
        Ok(transcription) => {
            assert!(!transcription.text.is_empty());
            assert!(transcription.confidence >= 0.0);
            assert!(transcription.confidence <= 1.0);
        }
        Err(e) => {
            // Expected to fail without API key
            assert!(e.to_string().contains("API key") || e.to_string().contains("authentication"));
        }
    }
}

#[tokio::test]
async fn test_tts_provider_integration() {
    let processor = VoiceProcessor::builder()
        .tts_provider("elevenlabs")
        .build()
        .await
        .expect("Failed to create processor");

    let text = "Hello, this is a test.";
    
    // Get available voices
    let voices = processor.list_voices().await;
    
    match voices {
        Ok(voice_list) => {
            if !voice_list.is_empty() {
                let options = SynthesisOptions::new(voice_list[0].clone());
                let result = processor.synthesize(text, &options).await;
                
                match result {
                    Ok(synthesis) => {
                        assert!(!synthesis.audio_data.is_empty());
                        assert!(synthesis.size_bytes > 0);
                    }
                    Err(e) => {
                        // Expected to fail without API key
                        assert!(e.to_string().contains("API key") || e.to_string().contains("authentication"));
                    }
                }
            }
        }
        Err(e) => {
            // Expected to fail without API key
            assert!(e.to_string().contains("API key") || e.to_string().contains("authentication"));
        }
    }
}

fn create_test_wav_data() -> Vec<u8> {
    // Create minimal WAV header + some dummy audio data
    let mut data = Vec::new();
    
    // WAV header (44 bytes)
    data.extend_from_slice(b"RIFF");           // ChunkID
    data.extend_from_slice(&36u32.to_le_bytes());  // ChunkSize
    data.extend_from_slice(b"WAVE");           // Format
    data.extend_from_slice(b"fmt ");           // Subchunk1ID
    data.extend_from_slice(&16u32.to_le_bytes());  // Subchunk1Size
    data.extend_from_slice(&1u16.to_le_bytes());   // AudioFormat (PCM)
    data.extend_from_slice(&1u16.to_le_bytes());   // NumChannels
    data.extend_from_slice(&16000u32.to_le_bytes()); // SampleRate
    data.extend_from_slice(&32000u32.to_le_bytes()); // ByteRate
    data.extend_from_slice(&2u16.to_le_bytes());   // BlockAlign
    data.extend_from_slice(&16u16.to_le_bytes());  // BitsPerSample
    data.extend_from_slice(b"data");           // Subchunk2ID
    data.extend_from_slice(&8u32.to_le_bytes());   // Subchunk2Size
    
    // Dummy audio data (8 bytes of silence)
    data.extend_from_slice(&[0u8; 8]);
    
    data
}
```

### Configuration Integration Tests

```rust
// tests/config_integration.rs
use debabelizer::{DebabelizerConfig, VoiceProcessor};
use std::env;
use tempfile::tempdir;
use std::fs;

#[tokio::test]
async fn test_config_from_environment() {
    // Set test environment variables
    env::set_var("DEBABELIZER_STT_PROVIDER", "soniox");
    env::set_var("DEBABELIZER_TTS_PROVIDER", "elevenlabs");
    env::set_var("SONIOX_API_KEY", "test-key");
    env::set_var("ELEVENLABS_API_KEY", "test-key");

    let config = DebabelizerConfig::from_env().unwrap();
    
    assert_eq!(config.get_preferred_stt_provider(), Some("soniox"));
    assert_eq!(config.get_preferred_tts_provider(), Some("elevenlabs"));
    
    // Clean up
    env::remove_var("DEBABELIZER_STT_PROVIDER");
    env::remove_var("DEBABELIZER_TTS_PROVIDER");
    env::remove_var("SONIOX_API_KEY");
    env::remove_var("ELEVENLABS_API_KEY");
}

#[tokio::test]
async fn test_config_from_file() {
    let temp_dir = tempdir().unwrap();
    let config_path = temp_dir.path().join(".debabelizer.toml");
    
    let config_content = r#"
[preferences]
stt_provider = "deepgram"
tts_provider = "openai"
auto_select = true
optimize_for = "quality"

[deepgram]
api_key = "test-deepgram-key"
model = "nova-2"

[openai]
api_key = "test-openai-key"
model = "tts-1-hd"
"#;
    
    fs::write(&config_path, config_content).unwrap();
    
    // Change to temp directory
    let original_dir = env::current_dir().unwrap();
    env::set_current_dir(&temp_dir).unwrap();
    
    let config = DebabelizerConfig::new().unwrap();
    
    assert_eq!(config.get_preferred_stt_provider(), Some("deepgram"));
    assert_eq!(config.get_preferred_tts_provider(), Some("openai"));
    assert!(config.is_auto_select_enabled());
    
    // Restore original directory
    env::set_current_dir(original_dir).unwrap();
}
```

## Mocking and Test Doubles

### Using Mockall for Provider Mocking

```rust
// tests/mock_providers.rs
use mockall::predicate::*;
use mockall::*;
use debabelizer_core::*;
use async_trait::async_trait;

#[automock]
#[async_trait]
trait MockSttProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult>;
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>>;
    async fn list_models(&self) -> Result<Vec<Model>>;
    fn supported_formats(&self) -> Vec<AudioFormat>;
    fn supports_streaming(&self) -> bool;
}

#[tokio::test]
async fn test_with_mock_stt_provider() {
    let mut mock_provider = MockMockSttProvider::new();
    
    // Set up expectations
    mock_provider
        .expect_name()
        .returning(|| "mock-stt");
    
    mock_provider
        .expect_transcribe()
        .with(always())
        .times(1)
        .returning(|_| {
            Ok(TranscriptionResult {
                text: "Hello world".to_string(),
                confidence: 0.95,
                language_detected: Some("en".to_string()),
                duration: Some(1.5),
                words: None,
                metadata: None,
            })
        });

    // Test the mock
    let audio = AudioData::new(vec![0u8; 1000], AudioFormat::wav(16000));
    let result = mock_provider.transcribe(audio).await.unwrap();
    
    assert_eq!(result.text, "Hello world");
    assert_eq!(result.confidence, 0.95);
    assert_eq!(result.language_detected, Some("en".to_string()));
}
```

### Manual Test Doubles

```rust
// tests/test_doubles.rs
use debabelizer_core::*;
use async_trait::async_trait;
use std::collections::HashMap;

pub struct FakeSttProvider {
    pub responses: HashMap<String, TranscriptionResult>,
    pub default_response: TranscriptionResult,
}

impl FakeSttProvider {
    pub fn new() -> Self {
        Self {
            responses: HashMap::new(),
            default_response: TranscriptionResult {
                text: "Default transcription".to_string(),
                confidence: 0.8,
                language_detected: Some("en".to_string()),
                duration: None,
                words: None,
                metadata: None,
            },
        }
    }
    
    pub fn with_response(mut self, audio_key: String, response: TranscriptionResult) -> Self {
        self.responses.insert(audio_key, response);
        self
    }
}

#[async_trait]
impl SttProvider for FakeSttProvider {
    fn name(&self) -> &str {
        "fake-stt"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        // Use audio data hash as key for deterministic responses
        let key = format!("{:x}", md5::compute(&audio.data));
        
        Ok(self.responses
            .get(&key)
            .cloned()
            .unwrap_or_else(|| self.default_response.clone()))
    }
    
    async fn transcribe_stream(&self, _config: StreamConfig) -> Result<Box<dyn SttStream>> {
        unimplemented!("Stream not implemented for fake provider")
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![Model {
            id: "fake-model".to_string(),
            name: "Fake Model".to_string(),
            languages: vec!["en".to_string()],
            capabilities: vec!["batch".to_string()],
        }])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![AudioFormat::wav(16000)]
    }
    
    fn supports_streaming(&self) -> bool {
        false
    }
}

#[tokio::test]
async fn test_with_fake_provider() {
    let provider = FakeSttProvider::new()
        .with_response(
            "test-key".to_string(),
            TranscriptionResult {
                text: "Custom response".to_string(),
                confidence: 0.99,
                language_detected: Some("es".to_string()),
                duration: Some(2.0),
                words: None,
                metadata: None,
            }
        );
    
    let audio = AudioData::new(b"test-key".to_vec(), AudioFormat::wav(16000));
    let result = provider.transcribe(audio).await.unwrap();
    
    assert_eq!(result.text, "Custom response");
    assert_eq!(result.confidence, 0.99);
}
```

## Async Testing

### Testing Streaming Operations

```rust
// tests/streaming_tests.rs
use debabelizer_core::*;
use tokio::time::{timeout, Duration};
use tokio_stream::StreamExt;

#[tokio::test]
async fn test_streaming_transcription() {
    // This test would require a real or mocked streaming provider
    // For demonstration, we'll show the testing pattern
    
    let processor = create_test_processor().await;
    let config = StreamConfig::default();
    
    let mut stream = processor.transcribe_stream(config).await.unwrap();
    
    // Send audio chunks
    let chunk1 = vec![0u8; 1600]; // 100ms of 16kHz audio
    let chunk2 = vec![1u8; 1600];
    
    stream.send_audio(&chunk1).await.unwrap();
    stream.send_audio(&chunk2).await.unwrap();
    
    // Receive results with timeout
    let result = timeout(Duration::from_secs(5), stream.receive_transcript()).await;
    
    match result {
        Ok(Ok(Some(transcript))) => {
            assert!(!transcript.text.is_empty());
            assert!(transcript.confidence >= 0.0);
        }
        Ok(Ok(None)) => {
            // Stream ended
        }
        Ok(Err(e)) => {
            panic!("Stream error: {}", e);
        }
        Err(_) => {
            panic!("Timeout waiting for transcript");
        }
    }
    
    stream.close().await.unwrap();
}

async fn create_test_processor() -> debabelizer::VoiceProcessor {
    // Create processor with test configuration
    debabelizer::VoiceProcessor::new().unwrap()
}
```

### Testing Concurrent Operations

```rust
use tokio::task::JoinSet;
use serial_test::serial;

#[tokio::test]
#[serial] // Ensure this test runs alone to avoid resource conflicts
async fn test_concurrent_transcriptions() {
    let processor = Arc::new(create_test_processor().await);
    let mut join_set = JoinSet::new();
    
    // Spawn multiple concurrent transcription tasks
    for i in 0..5 {
        let processor_clone = processor.clone();
        join_set.spawn(async move {
            let audio = create_test_audio(i);
            processor_clone.transcribe(audio).await
        });
    }
    
    let mut results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        results.push(result.unwrap());
    }
    
    assert_eq!(results.len(), 5);
    
    // Check that all results are valid (or expected errors)
    for result in results {
        match result {
            Ok(transcription) => {
                assert!(transcription.confidence >= 0.0);
            }
            Err(e) => {
                // Expected errors (like missing API keys) are OK
                assert!(e.to_string().contains("API key"));
            }
        }
    }
}

fn create_test_audio(seed: usize) -> AudioData {
    let mut data = vec![0u8; 1000];
    // Make each audio unique
    data[0] = seed as u8;
    AudioData::new(data, AudioFormat::wav(16000))
}
```

## Provider Testing

### HTTP Provider Testing with WireMock

```rust
// Cargo.toml
[dev-dependencies]
wiremock = "0.6"

// tests/http_provider_tests.rs
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path, header};
use debabelizer_core::*;

#[tokio::test]
async fn test_elevenlabs_provider_with_mock_server() {
    // Start mock server
    let mock_server = MockServer::start().await;
    
    // Mock the voices endpoint
    Mock::given(method("GET"))
        .and(path("/v1/voices"))
        .and(header("xi-api-key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "voices": [
                {
                    "voice_id": "test-voice-id",
                    "name": "Test Voice",
                    "category": "premade",
                    "labels": {
                        "language": "en",
                        "gender": "female"
                    }
                }
            ]
        })))
        .mount(&mock_server)
        .await;
    
    // Mock the synthesis endpoint
    Mock::given(method("POST"))
        .and(path("/v1/text-to-speech/test-voice-id"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_bytes(vec![0u8; 1000]) // Mock MP3 data
            .append_header("content-type", "audio/mpeg"))
        .mount(&mock_server)
        .await;
    
    // Create provider with mock server URL
    let provider = create_mock_elevenlabs_provider(&mock_server.uri()).await;
    
    // Test listing voices
    let voices = provider.list_voices().await.unwrap();
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0].voice_id, "test-voice-id");
    assert_eq!(voices[0].name, "Test Voice");
    
    // Test synthesis
    let options = SynthesisOptions::new(voices[0].clone());
    let result = provider.synthesize("Hello world", &options).await.unwrap();
    
    assert_eq!(result.audio_data.len(), 1000);
    assert_eq!(result.format.format, "mp3");
}

async fn create_mock_elevenlabs_provider(base_url: &str) -> impl TtsProvider {
    // Implementation would create ElevenLabs provider with custom base URL
    // This requires modifying the provider to accept custom endpoints for testing
    unimplemented!("Requires provider modification for testing")
}
```

### WebSocket Provider Testing

```rust
// tests/websocket_provider_tests.rs
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use futures::{SinkExt, StreamExt};

#[tokio::test]
async fn test_soniox_websocket_protocol() {
    // This would require a mock WebSocket server
    // For demonstration, we'll show the testing pattern
    
    // Mock WebSocket server responses
    let mock_responses = vec![
        serde_json::json!({
            "result": {
                "transcript": "Hello",
                "is_final": false,
                "confidence": 0.8
            }
        }),
        serde_json::json!({
            "result": {
                "transcript": "Hello world",
                "is_final": true,
                "confidence": 0.95,
                "language": "en"
            }
        }),
    ];
    
    // Test the protocol implementation
    // This would require setting up a mock WebSocket server
    // and testing the provider's WebSocket message handling
}
```

## Test Configuration

### Environment-specific Test Configuration

```rust
// tests/common/mod.rs
use std::env;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn setup_test_environment() {
    INIT.call_once(|| {
        // Initialize tracing for tests
        if env::var("RUST_LOG").is_err() {
            env::set_var("RUST_LOG", "debug");
        }
        
        tracing_subscriber::fmt()
            .with_test_writer()
            .init();
    });
}

pub fn get_test_config() -> debabelizer::DebabelizerConfig {
    use std::collections::HashMap;
    
    let mut config_map = HashMap::new();
    
    // Use test API keys if available, otherwise use dummy values
    if let Ok(soniox_key) = env::var("SONIOX_TEST_API_KEY") {
        config_map.insert("soniox.api_key".to_string(), serde_json::Value::String(soniox_key));
    } else {
        config_map.insert("soniox.api_key".to_string(), serde_json::Value::String("test-key".to_string()));
    }
    
    if let Ok(elevenlabs_key) = env::var("ELEVENLABS_TEST_API_KEY") {
        config_map.insert("elevenlabs.api_key".to_string(), serde_json::Value::String(elevenlabs_key));
    } else {
        config_map.insert("elevenlabs.api_key".to_string(), serde_json::Value::String("test-key".to_string()));
    }
    
    debabelizer::DebabelizerConfig::from_map(config_map).unwrap()
}

pub fn skip_if_no_api_keys() {
    if env::var("SONIOX_TEST_API_KEY").is_err() && env::var("ELEVENLABS_TEST_API_KEY").is_err() {
        println!("Skipping test: No test API keys provided");
        return;
    }
}
```

### Test Feature Flags

```rust
// Cargo.toml
[features]
default = ["soniox", "elevenlabs"]
test-utils = ["tempfile", "wiremock"]
integration-tests = ["test-utils"]

# In test files
#[cfg(feature = "integration-tests")]
mod integration_tests {
    // Integration test code
}
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_audio_format_creation

# Run tests matching pattern
cargo test config

# Run tests in specific crate
cargo test -p debabelizer-core

# Run integration tests only
cargo test --test integration_tests

# Run with specific features
cargo test --features integration-tests

# Run with all features
cargo test --all-features
```

### Environment Variables for Testing

```bash
# Set test API keys
export SONIOX_TEST_API_KEY=your_test_key
export ELEVENLABS_TEST_API_KEY=your_test_key

# Enable debug logging for tests
export RUST_LOG=debug

# Run tests with environment
cargo test
```

### Parallel vs Sequential Testing

```bash
# Run tests in parallel (default)
cargo test

# Run tests sequentially
cargo test -- --test-threads=1

# Run specific tests sequentially with serial_test crate
# Tests marked with #[serial] will run sequentially
```

## Coverage and Quality

### Code Coverage with Tarpaulin

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# Coverage with specific features
cargo tarpaulin --features integration-tests --out Html

# Coverage excluding integration tests
cargo tarpaulin --lib --out Html
```

### Benchmark Testing

```rust
// benches/audio_processing.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use debabelizer_utils::detect_audio_format;

fn benchmark_format_detection(c: &mut Criterion) {
    let wav_data = include_bytes!("../tests/data/sample.wav");
    let mp3_data = include_bytes!("../tests/data/sample.mp3");
    
    c.bench_function("detect wav format", |b| {
        b.iter(|| detect_audio_format(black_box(wav_data)))
    });
    
    c.bench_function("detect mp3 format", |b| {
        b.iter(|| detect_audio_format(black_box(mp3_data)))
    });
}

criterion_group!(benches, benchmark_format_detection);
criterion_main!(benches);
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_audio_format_roundtrip(
        format in prop::sample::select(&["wav", "mp3", "opus"]),
        sample_rate in prop::sample::select(&[8000u32, 16000, 44100, 48000]),
        channels in 1u8..=8
    ) {
        let audio_format = AudioFormat {
            format: format.to_string(),
            sample_rate,
            channels,
            bit_depth: Some(16),
        };
        
        // Serialize and deserialize
        let json = serde_json::to_string(&audio_format).unwrap();
        let deserialized: AudioFormat = serde_json::from_str(&json).unwrap();
        
        prop_assert_eq!(audio_format, deserialized);
    }
}
```

## Best Practices

### Test Organization

1. **Group related tests** in modules
2. **Use descriptive test names** that explain what is being tested
3. **Test both success and failure cases**
4. **Keep tests independent** - each test should be able to run in isolation
5. **Use setup and teardown** functions for common test preparation

### Test Data Management

```rust
// tests/data/mod.rs
use std::path::PathBuf;

pub fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data")
}

pub fn load_test_audio(filename: &str) -> Vec<u8> {
    std::fs::read(test_data_dir().join(filename))
        .unwrap_or_else(|_| panic!("Failed to load test file: {}", filename))
}

pub fn create_temp_audio_file() -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&create_test_wav_data()).unwrap();
    file
}
```

### Async Test Guidelines

1. **Use `#[tokio::test]`** for async tests
2. **Set timeouts** for operations that might hang
3. **Test cancellation** scenarios where applicable
4. **Use `serial_test`** for tests that can't run concurrently

### Mock and Stub Guidelines

1. **Mock external dependencies** (HTTP APIs, WebSocket servers)
2. **Use dependency injection** to make components testable
3. **Create realistic test data** that matches production scenarios
4. **Test error conditions** with mocked failures

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Run tests
        run: cargo test --all-features
      
      - name: Run integration tests
        run: cargo test --features integration-tests
        env:
          SONIOX_TEST_API_KEY: ${{ secrets.SONIOX_TEST_API_KEY }}
          ELEVENLABS_TEST_API_KEY: ${{ secrets.ELEVENLABS_TEST_API_KEY }}
      
      - name: Generate coverage
        run: |
          cargo install cargo-tarpaulin
          cargo tarpaulin --out Xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

This comprehensive testing guide should help you implement robust testing for the Debabelizer project, ensuring reliability and maintainability as the codebase grows.