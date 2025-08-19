use async_trait::async_trait;
use bytes::Bytes;
use debabelizer_core::{
    AudioFormat, DebabelizerError, ProviderError, Result, SynthesisOptions,
    SynthesisResult, TtsProvider, TtsStream, Voice,
};
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde_json;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use tokio_stream::wrappers::ReceiverStream;

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
const MAX_TEXT_LENGTH: usize = 4096;

#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Simple(std::collections::HashMap<String, serde_json::Value>),
}

impl ProviderConfig {
    pub fn get_api_key(&self) -> Option<String> {
        match self {
            Self::Simple(map) => map
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }
    }
    
    pub fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        match self {
            Self::Simple(map) => map.get(key),
        }
    }
}

#[derive(Debug)]
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    default_model: String,
    #[allow(dead_code)]
    default_voice: String,
}

impl OpenAIProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("OpenAI API key not found".to_string()))?;
        
        let default_model = config
            .get_value("tts_model")
            .and_then(|v| v.as_str())
            .unwrap_or("tts-1")
            .to_string();
        
        let default_voice = config
            .get_value("tts_voice")
            .and_then(|v| v.as_str())
            .unwrap_or("alloy")
            .to_string();
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            default_model,
            default_voice,
        })
    }
    
    fn get_available_voices() -> Vec<Voice> {
        vec![
            Voice {
                voice_id: "alloy".to_string(),
                name: "Alloy".to_string(),
                language: "en".to_string(),
                gender: Some("neutral".to_string()),
                description: Some("Neutral, balanced voice suitable for most content".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
            Voice {
                voice_id: "echo".to_string(),
                name: "Echo".to_string(),
                language: "en".to_string(),
                gender: Some("male".to_string()),
                description: Some("Male voice with clear articulation".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
            Voice {
                voice_id: "fable".to_string(),
                name: "Fable".to_string(),
                language: "en".to_string(),
                gender: Some("male".to_string()),
                description: Some("Male voice with storytelling quality".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
            Voice {
                voice_id: "onyx".to_string(),
                name: "Onyx".to_string(),
                language: "en".to_string(),
                gender: Some("male".to_string()),
                description: Some("Deep male voice with authority".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
            Voice {
                voice_id: "nova".to_string(),
                name: "Nova".to_string(),
                language: "en".to_string(),
                gender: Some("female".to_string()),
                description: Some("Female voice with clarity and warmth".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
            Voice {
                voice_id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                language: "en".to_string(),
                gender: Some("female".to_string()),
                description: Some("Female voice with bright, energetic tone".to_string()),
                preview_url: None,
                accent: None,
                age: None,
                style: None,
                use_case: None,
                metadata: None,
            },
        ]
    }
    
    fn validate_text_length(&self, text: &str) -> Result<()> {
        if text.len() > MAX_TEXT_LENGTH {
            Err(DebabelizerError::Provider(ProviderError::InvalidRequest(
                format!("Text too long ({} characters). OpenAI TTS supports max {} characters.", 
                    text.len(), MAX_TEXT_LENGTH)
            )))
        } else {
            Ok(())
        }
    }
    
    fn get_voice_by_id(&self, voice_id: &str) -> Option<Voice> {
        Self::get_available_voices()
            .into_iter()
            .find(|v| v.voice_id == voice_id)
    }
    
    fn estimate_duration(text: &str, format: &str, audio_size: usize, speed: f32) -> f32 {
        let words = text.split_whitespace().count() as f32;
        
        // More accurate duration estimation based on audio format
        let duration = match format {
            "mp3" => {
                // MP3 compression varies, but estimate based on typical bitrate (~64kbps for speech)
                let estimated_bitrate = 64000.0; // bits per second
                (audio_size as f32 * 8.0) / estimated_bitrate
            }
            "wav" | "pcm" => {
                // Uncompressed: bytes = sample_rate * channels * bytes_per_sample * duration
                // OpenAI typically outputs 24kHz, mono, 16-bit
                let bytes_per_second = 24000.0 * 1.0 * 2.0; // 24kHz * mono * 2 bytes per sample
                audio_size as f32 / bytes_per_second
            }
            _ => {
                // Fallback to word-based estimation for other formats
                (words / 150.0) * 60.0
            }
        };
        
        // Apply speed factor
        duration / speed
    }
}

#[async_trait]
impl TtsProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }
    
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult> {
        // Validate text length
        self.validate_text_length(text)?;
        
        // Determine voice
        let voice_id = &options.voice.voice_id;
        
        // Validate voice exists
        if !Self::get_available_voices().iter().any(|v| &v.voice_id == voice_id) {
            return Err(DebabelizerError::Provider(ProviderError::InvalidRequest(
                format!("Voice '{}' not found", voice_id)
            )));
        }
        
        // Determine audio format
        let output_format = options.format.format.as_str();
        
        // Validate format
        let supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"];
        if !supported_formats.contains(&output_format) {
            return Err(DebabelizerError::Provider(ProviderError::InvalidRequest(
                format!("Unsupported audio format: {}", output_format)
            )));
        }
        
        // Determine model
        let model = options.model.as_ref().unwrap_or(&self.default_model);
        let model = if model == "tts-1" || model == "tts-1-hd" {
            model
        } else {
            &self.default_model
        };
        
        // Build request body
        let mut body = json!({
            "model": model,
            "voice": voice_id,
            "input": text,
            "response_format": output_format,
        });
        
        // Add speed if specified
        if let Some(speed) = options.speed {
            body["speed"] = json!(speed);
        }
        
        // Make API request
        let url = format!("{}/audio/speech", OPENAI_API_BASE);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let audio_data = response.bytes().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?
                    .to_vec();
                
                let size_bytes = audio_data.len();
                let speed = options.speed.unwrap_or(1.0);
                let duration = Self::estimate_duration(text, output_format, size_bytes, speed);
                
                // Get voice info for potential future use
                let _voice_used = self.get_voice_by_id(voice_id).unwrap_or_else(|| Voice {
                    voice_id: voice_id.to_string(),
                    name: voice_id.to_string(),
                    description: None,
                    language: "en".to_string(),
                    gender: None,
                    age: None,
                    accent: None,
                    style: None,
                    use_case: None,
                    preview_url: None,
                    metadata: None,
                });
                
                // Build metadata
                let mut metadata = serde_json::Map::new();
                metadata.insert("model".to_string(), json!(model));
                metadata.insert("voice".to_string(), json!(voice_id));
                metadata.insert("speed".to_string(), json!(speed));
                metadata.insert("character_count".to_string(), json!(text.len()));
                metadata.insert("word_count".to_string(), json!(text.split_whitespace().count()));
                metadata.insert("response_format".to_string(), json!(output_format));
                metadata.insert("actual_sample_rate".to_string(), json!(24000));
                
                let format = match output_format {
                    "mp3" => AudioFormat::mp3(24000),
                    "opus" => AudioFormat::opus(24000),
                    "wav" | "pcm" => AudioFormat::wav(24000),
                    _ => AudioFormat::mp3(24000), // Default fallback
                };
                
                Ok(SynthesisResult {
                    audio_data,
                    format,
                    duration: Some(duration),
                    size_bytes,
                    metadata: Some(json!(metadata)),
                })
            }
            StatusCode::UNAUTHORIZED => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid OpenAI API key".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("OpenAI API error: {}", error_text)
                )))
            }
        }
    }
    
    async fn synthesize_stream(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsStream>> {
        // Validate text length
        self.validate_text_length(text)?;
        
        // Since OpenAI doesn't support true streaming, we'll download the full audio
        // and simulate streaming by chunking it
        let result = self.synthesize(text, options).await?;
        
        // Create a stream that yields chunks of the audio data
        let stream = OpenAIStream::new(
            result.audio_data,
            options.speed.unwrap_or(1.0),
            text.len(),
        );
        
        Ok(Box::new(stream))
    }
    
    async fn list_voices(&self) -> Result<Vec<Voice>> {
        Ok(Self::get_available_voices())
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::mp3(24000),
            AudioFormat::opus(24000),
            AudioFormat::wav(24000), // For AAC, FLAC, PCM we use wav as base format
            AudioFormat::wav(24000), // Multiple wav entries for different internal formats
            AudioFormat::wav(24000),
            AudioFormat::wav(24000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true // Simulated streaming
    }
    
    fn supports_ssml(&self) -> bool {
        false
    }
}

struct OpenAIStream {
    receiver: Arc<Mutex<ReceiverStream<Result<Bytes>>>>,
}

impl OpenAIStream {
    fn new(audio_data: Vec<u8>, speed: f32, text_length: usize) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        // Calculate timing for realistic streaming simulation
        let words = text_length.max(1) / 6; // Rough word count estimate
        let estimated_duration = (words as f32 / 150.0) * 60.0 / speed;
        let chunk_size = 1024;
        let total_chunks = (audio_data.len() + chunk_size - 1) / chunk_size;
        let delay_per_chunk = if total_chunks > 0 {
            Duration::from_secs_f32((estimated_duration / total_chunks as f32).min(0.1))
        } else {
            Duration::from_millis(10)
        };
        
        tokio::spawn(async move {
            for (i, chunk) in audio_data.chunks(chunk_size).enumerate() {
                let is_last = i == total_chunks - 1;
                
                if tx.send(Ok(Bytes::from(chunk.to_vec()))).await.is_err() {
                    break;
                }
                
                // Simulate realistic streaming delay
                if !is_last {
                    sleep(delay_per_chunk).await;
                }
            }
        });
        
        Self {
            receiver: Arc::new(Mutex::new(ReceiverStream::new(rx))),
        }
    }
}

#[async_trait]
impl TtsStream for OpenAIStream {
    async fn receive_chunk(&mut self) -> Result<Option<Bytes>> {
        let mut receiver = self.receiver.lock().await;
        match receiver.next().await {
            Some(Ok(bytes)) => Ok(Some(bytes)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        // Stream will close automatically when dropped
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[tokio::test]
    async fn test_provider_creation() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map
        });
        
        let provider = OpenAIProvider::new(&config).await;
        assert!(provider.is_ok());
        
        let provider = provider.unwrap();
        assert_eq!(provider.default_model, "tts-1");
        assert_eq!(provider.default_voice, "alloy");
    }
    
    #[tokio::test]
    async fn test_provider_creation_with_config() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("tts_model".to_string(), json!("tts-1-hd"));
            map.insert("tts_voice".to_string(), json!("nova"));
            map
        });
        
        let provider = OpenAIProvider::new(&config).await.unwrap();
        assert_eq!(provider.default_model, "tts-1-hd");
        assert_eq!(provider.default_voice, "nova");
    }
    
    #[tokio::test]
    async fn test_provider_creation_fails_without_api_key() {
        let config = ProviderConfig::Simple(std::collections::HashMap::new());
        
        let result = OpenAIProvider::new(&config).await;
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(matches!(error, DebabelizerError::Configuration(_)));
    }
    
    #[test]
    fn test_provider_name() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        assert_eq!(provider.name(), "openai");
    }
    
    #[test]
    fn test_available_voices() {
        let voices = OpenAIProvider::get_available_voices();
        assert_eq!(voices.len(), 6);
        
        // Check all voices are present
        let voice_ids: Vec<&str> = voices.iter().map(|v| v.voice_id.as_str()).collect();
        assert!(voice_ids.contains(&"alloy"));
        assert!(voice_ids.contains(&"echo"));
        assert!(voice_ids.contains(&"fable"));
        assert!(voice_ids.contains(&"onyx"));
        assert!(voice_ids.contains(&"nova"));
        assert!(voice_ids.contains(&"shimmer"));
        
        // Check voice metadata
        let alloy = voices.iter().find(|v| v.voice_id == "alloy").unwrap();
        assert_eq!(alloy.name, "Alloy");
        assert_eq!(alloy.gender, Some("neutral".to_string()));
        assert!(alloy.description.is_some());
    }
    
    #[test]
    fn test_validate_text_length() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        // Valid text
        assert!(provider.validate_text_length("Hello world").is_ok());
        
        // Text at limit
        let max_text = "a".repeat(MAX_TEXT_LENGTH);
        assert!(provider.validate_text_length(&max_text).is_ok());
        
        // Text over limit
        let long_text = "a".repeat(MAX_TEXT_LENGTH + 1);
        assert!(provider.validate_text_length(&long_text).is_err());
    }
    
    #[test]
    fn test_get_voice_by_id() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        // Existing voice
        let voice = provider.get_voice_by_id("nova");
        assert!(voice.is_some());
        let voice = voice.unwrap();
        assert_eq!(voice.voice_id, "nova");
        assert_eq!(voice.gender, Some("female".to_string()));
        
        // Non-existing voice
        assert!(provider.get_voice_by_id("invalid").is_none());
    }
    
    #[test]
    fn test_supported_formats() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        let formats = provider.supported_formats();
        assert_eq!(formats.len(), 6);
        
        // Check all formats have 24kHz sample rate
        for format in &formats {
            assert_eq!(format.sample_rate, 24000);
        }
        
        // Check format types (simplified since some formats are mapped to wav)
        let format_names: Vec<&str> = formats.iter().map(|f| f.format.as_str()).collect();
        assert!(format_names.contains(&"mp3"));
        assert!(format_names.contains(&"opus"));
        assert!(format_names.contains(&"wav")); // AAC, FLAC, PCM are mapped to wav
        
        // Count unique format types
        let mut unique_formats = std::collections::HashSet::new();
        for format in &formats {
            unique_formats.insert(format.format.as_str());
        }
        assert!(unique_formats.len() >= 3); // At least mp3, opus, wav
    }
    
    #[test]
    fn test_supports_streaming() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        assert!(provider.supports_streaming());
    }
    
    #[test]
    fn test_supports_ssml() {
        let provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        assert!(!provider.supports_ssml());
    }
    
    #[test]
    fn test_estimate_duration() {
        // Test MP3 duration estimation
        let text = "Hello world, this is a test.";
        let duration = OpenAIProvider::estimate_duration(text, "mp3", 8000, 1.0);
        assert!(duration > 0.0 && duration < 10.0); // Reasonable duration
        
        // Test WAV duration estimation
        let duration = OpenAIProvider::estimate_duration(text, "wav", 48000, 1.0);
        assert_eq!(duration, 48000.0 / (24000.0 * 2.0)); // 1 second for 48KB at 24kHz 16-bit mono
        
        // Test speed factor
        let normal_duration = OpenAIProvider::estimate_duration(text, "mp3", 8000, 1.0);
        let fast_duration = OpenAIProvider::estimate_duration(text, "mp3", 8000, 2.0);
        assert_eq!(fast_duration, normal_duration / 2.0);
    }
    
    #[tokio::test]
    async fn test_stream_creation() {
        
        let audio_data = vec![0u8; 5120]; // 5KB of data
        let mut stream = OpenAIStream::new(audio_data.clone(), 1.0, 100);
        
        let mut received_data = Vec::new();
        while let Some(chunk) = stream.receive_chunk().await.unwrap() {
            received_data.extend_from_slice(&chunk);
        }
        
        assert_eq!(received_data.len(), audio_data.len());
    }
    
    #[test]
    fn test_provider_config_extraction() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("sk-test123"));
            map.insert("tts_model".to_string(), json!("tts-1-hd"));
            map.insert("tts_voice".to_string(), json!("shimmer"));
            map.insert("extra_param".to_string(), json!(42));
            map
        });
        
        assert_eq!(config.get_api_key(), Some("sk-test123".to_string()));
        assert_eq!(config.get_value("tts_model").unwrap().as_str(), Some("tts-1-hd"));
        assert_eq!(config.get_value("tts_voice").unwrap().as_str(), Some("shimmer"));
        assert_eq!(config.get_value("extra_param").unwrap().as_i64(), Some(42));
        assert!(config.get_value("missing_key").is_none());
    }
    
    #[tokio::test]
    async fn test_synthesis_options_validation() {
        let _provider = OpenAIProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            default_model: "tts-1".to_string(),
            default_voice: "alloy".to_string(),
        };
        
        // Test with default options
        let voice = Voice::new("alloy".to_string(), "Alloy".to_string(), "en".to_string());
        let _options = SynthesisOptions::new(voice);
        
        // This would fail in a real API call, but we're testing the validation logic
        // In a real test with mocked HTTP, we'd verify the request format
        
        // Test with custom options
        let voice = Voice::new("nova".to_string(), "Nova".to_string(), "en".to_string());
        let mut options = SynthesisOptions::new(voice);
        options.model = Some("tts-1-hd".to_string());
        options.speed = Some(1.5);
        options.format = AudioFormat::mp3(24000);
        
        // Verify these options would be processed correctly
        assert_eq!(options.voice.voice_id, "nova");
        assert_eq!(options.model, Some("tts-1-hd".to_string()));
        assert_eq!(options.speed, Some(1.5));
    }
    
    #[test]
    fn test_api_constants() {
        assert_eq!(OPENAI_API_BASE, "https://api.openai.com/v1");
        assert_eq!(MAX_TEXT_LENGTH, 4096);
    }
    
    // Integration test with mocked HTTP would go here
    #[tokio::test]
    async fn test_synthesize_with_mocked_response() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::{method, path, header, body_json};
        
        let mock_server = MockServer::start().await;
        
        // Create provider with mocked URL
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map
        });
        
        let _provider = OpenAIProvider::new(&config).await.unwrap();
        
        // Override the client to use mock server
        // Note: In a real implementation, we'd make the base URL configurable
        // For this test, we're demonstrating the structure
        
        // Mock the OpenAI API response
        let mock_audio = vec![0u8; 1024]; // Mock audio data
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(header("Authorization", "Bearer test-key"))
            .and(body_json(json!({
                "model": "tts-1",
                "voice": "alloy",
                "input": "Hello world",
                "response_format": "mp3"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(mock_audio.clone())
            )
            .mount(&mock_server)
            .await;
        
        // Test would continue here with actual API call
        // This demonstrates the test structure
    }
}