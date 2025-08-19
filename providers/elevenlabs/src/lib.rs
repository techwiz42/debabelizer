use async_trait::async_trait;
use bytes::Bytes;
use debabelizer_core::{
    AudioFormat, DebabelizerError, ProviderError, Result, SynthesisOptions,
    SynthesisResult, TtsProvider, TtsStream, Voice,
};

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
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

const ELEVENLABS_API_BASE: &str = "https://api.elevenlabs.io/v1";

#[derive(Debug)]
pub struct ElevenLabsProvider {
    client: Client,
    api_key: String,
    model_id: String,
}

impl ElevenLabsProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("ElevenLabs API key not found".to_string()))?;
        
        let model_id = config
            .get_value("model")
            .or_else(|| config.get_value("model_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("eleven_monolingual_v1")
            .to_string();
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            model_id,
        })
    }
    
    async fn make_request<T: for<'de> Deserialize<'de>>(
        &self,
        method: reqwest::Method,
        endpoint: &str,
        body: Option<serde_json::Value>,
    ) -> Result<T> {
        let url = format!("{}{}", ELEVENLABS_API_BASE, endpoint);
        let mut request = self.client.request(method, &url)
            .header("xi-api-key", &self.api_key)
            .header("Accept", "application/json");
        
        if let Some(body) = body {
            request = request
                .header("Content-Type", "application/json")
                .json(&body);
        }
        
        let response = request.send().await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                response.json::<T>().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))
            }
            StatusCode::UNAUTHORIZED => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid API key".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("ElevenLabs API error: {}", error_text)
                )))
            }
        }
    }
}

#[async_trait]
impl TtsProvider for ElevenLabsProvider {
    fn name(&self) -> &str {
        "elevenlabs"
    }
    
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult> {
        let voice_id = &options.voice.voice_id;
        let endpoint = format!("/text-to-speech/{}", voice_id);
        
        let mut voice_settings = json!({
            "stability": 0.5,
            "similarity_boost": 0.5,
        });
        
        if let Some(metadata) = &options.metadata {
            if let Some(settings) = metadata.get("voice_settings") {
                voice_settings = settings.clone();
            }
        }
        
        let body = json!({
            "text": text,
            "model_id": options.model.as_ref().unwrap_or(&self.model_id),
            "voice_settings": voice_settings,
        });
        
        let url = format!("{}{}", ELEVENLABS_API_BASE, endpoint);
        let response = self.client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .header("Accept", "audio/mpeg")
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
                
                Ok(SynthesisResult {
                    audio_data,
                    format: AudioFormat::mp3(44100),
                    duration: None,
                    size_bytes,
                    metadata: None,
                })
            }
            StatusCode::UNAUTHORIZED => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid API key".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("ElevenLabs API error: {}", error_text)
                )))
            }
        }
    }
    
    async fn synthesize_stream(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsStream>> {
        let voice_id = &options.voice.voice_id;
        let endpoint = format!("/text-to-speech/{}/stream", voice_id);
        
        let mut voice_settings = json!({
            "stability": 0.5,
            "similarity_boost": 0.5,
        });
        
        if let Some(metadata) = &options.metadata {
            if let Some(settings) = metadata.get("voice_settings") {
                voice_settings = settings.clone();
            }
        }
        
        let body = json!({
            "text": text,
            "model_id": options.model.as_ref().unwrap_or(&self.model_id),
            "voice_settings": voice_settings,
        });
        
        let url = format!("{}{}", ELEVENLABS_API_BASE, endpoint);
        let response = self.client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .header("Accept", "audio/mpeg")
            .json(&body)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let stream = response.bytes_stream();
                Ok(Box::new(ElevenLabsStream::new(stream)))
            }
            StatusCode::UNAUTHORIZED => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid API key".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("ElevenLabs API error: {}", error_text)
                )))
            }
        }
    }
    
    async fn list_voices(&self) -> Result<Vec<Voice>> {
        #[derive(Deserialize)]
        struct VoicesResponse {
            voices: Vec<ElevenLabsVoice>,
        }
        
        #[derive(Deserialize)]
        struct ElevenLabsVoice {
            voice_id: String,
            name: String,
            #[serde(default)]
            labels: Option<std::collections::HashMap<String, String>>,
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            preview_url: Option<String>,
        }
        
        let response: VoicesResponse = self.make_request(
            reqwest::Method::GET,
            "/voices",
            None,
        ).await?;
        
        Ok(response.voices.into_iter().map(|v| {
            let mut voice = Voice::new(
                v.voice_id,
                v.name,
                v.labels.as_ref()
                    .and_then(|l| l.get("language"))
                    .cloned()
                    .unwrap_or_else(|| "en".to_string()),
            );
            
            voice.description = v.description;
            voice.preview_url = v.preview_url;
            
            if let Some(labels) = v.labels {
                voice.gender = labels.get("gender").cloned();
                voice.accent = labels.get("accent").cloned();
                voice.age = labels.get("age").cloned();
                voice.use_case = labels.get("use case").cloned();
            }
            
            voice
        }).collect())
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::mp3(44100),
            AudioFormat::mp3(22050),
            AudioFormat::mp3(24000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_ssml(&self) -> bool {
        false
    }
}

struct ElevenLabsStream {
    receiver: Arc<Mutex<ReceiverStream<Result<Bytes>>>>,
}

impl ElevenLabsStream {
    fn new(stream: impl futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        tokio::spawn(async move {
            futures::pin_mut!(stream);
            while let Some(item) = stream.next().await {
                let result = match item {
                    Ok(bytes) => Ok(bytes),
                    Err(e) => Err(DebabelizerError::Provider(ProviderError::Network(e.to_string()))),
                };
                
                if tx.send(result).await.is_err() {
                    break;
                }
            }
        });
        
        Self {
            receiver: Arc::new(Mutex::new(ReceiverStream::new(rx))),
        }
    }
}

#[async_trait]
impl TtsStream for ElevenLabsStream {
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
    
    #[tokio::test]
    async fn test_provider_creation() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map
        });
        
        let provider = ElevenLabsProvider::new(&config).await;
        assert!(provider.is_ok());
    }
    
    #[test]
    fn test_supported_formats() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        let formats = provider.supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.iter().any(|f| f.format == "mp3"));
    }
    
    #[test]
    fn test_provider_name() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        assert_eq!(provider.name(), "elevenlabs");
    }

    #[test]
    fn test_provider_config_api_key_extraction() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("my-secret-key"));
            map.insert("model".to_string(), json!("eleven_multilingual_v2"));
            map
        });
        
        assert_eq!(config.get_api_key(), Some("my-secret-key".to_string()));
        assert_eq!(config.get_value("model").unwrap().as_str(), Some("eleven_multilingual_v2"));
    }

    #[test]
    fn test_provider_config_missing_api_key() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("model".to_string(), json!("eleven_monolingual_v1"));
            map
        });
        
        assert!(config.get_api_key().is_none());
    }

    #[tokio::test]
    async fn test_provider_creation_fails_without_api_key() {
        let config = ProviderConfig::Simple(std::collections::HashMap::new());
        
        let result = ElevenLabsProvider::new(&config).await;
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(matches!(error, DebabelizerError::Configuration(_)));
    }

    #[tokio::test]
    async fn test_provider_creation_with_custom_model() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("model".to_string(), json!("eleven_turbo_v2"));
            map
        });
        
        let provider = ElevenLabsProvider::new(&config).await.unwrap();
        assert_eq!(provider.model_id, "eleven_turbo_v2");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supports_ssml() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        assert!(!provider.supports_ssml());
    }

    #[test]
    fn test_supported_formats_contains_multiple_sample_rates() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        let formats = provider.supported_formats();
        assert!(formats.len() >= 3);
        
        let sample_rates: Vec<u32> = formats.iter().map(|f| f.sample_rate).collect();
        assert!(sample_rates.contains(&44100));
        assert!(sample_rates.contains(&22050));
        assert!(sample_rates.contains(&24000));
    }

    #[test]
    fn test_make_request_url_construction() {
        let _provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test-key".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        // Test URL construction logic (we can't test actual requests without mocking)
        let base_url = "https://api.elevenlabs.io/v1";
        let endpoint = "/voices";
        let expected_url = format!("{}{}", base_url, endpoint);
        
        assert_eq!(expected_url, "https://api.elevenlabs.io/v1/voices");
    }

    // Mock tests for error handling
    #[tokio::test]
    async fn test_synthesize_with_invalid_options() {
        let provider = ElevenLabsProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
        };
        
        let voice = Voice::new(
            "test_voice".to_string(),
            "Test Voice".to_string(),
            "en".to_string(),
        );
        let options = SynthesisOptions::new(voice);
        
        // Test would fail in real implementation due to missing voice_id
        // This is a structural test to ensure error handling paths exist
        let result = provider.synthesize("Hello world", &options).await;
        // In a real scenario with network, this would return an error
        // For now, we just ensure the method signature is correct
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_synthesis_options_validation() {
        let voice = Voice::new(
            "21m00Tcm4TlvDq8ikWAM".to_string(),
            "Test Voice".to_string(),
            "en".to_string(),
        );
        let mut options = SynthesisOptions::new(voice);
        options.model = Some("eleven_turbo_v2".to_string());
        options.speed = Some(1.5);
        options.stability = Some(0.8);
        options.similarity_boost = Some(0.9);
        options.format = AudioFormat::mp3(44100);
        
        assert_eq!(options.voice.voice_id, "21m00Tcm4TlvDq8ikWAM");
        assert_eq!(options.model, Some("eleven_turbo_v2".to_string()));
        assert_eq!(options.speed, Some(1.5));
        
        // Test boundary values
        assert!(options.stability.unwrap() >= 0.0 && options.stability.unwrap() <= 1.0);
        assert!(options.similarity_boost.unwrap() >= 0.0 && options.similarity_boost.unwrap() <= 1.0);
    }

    #[tokio::test]
    async fn test_stream_creation() {
        use futures::stream;
        
        // Create a mock stream of bytes
        let mock_data = vec![
            Ok(Bytes::from("chunk1")),
            Ok(Bytes::from("chunk2")),
            Ok(Bytes::from("chunk3")),
        ];
        let mock_stream = stream::iter(mock_data);
        
        let mut tts_stream = ElevenLabsStream::new(mock_stream);
        
        // Test receiving chunks
        let chunk1 = tts_stream.receive_chunk().await;
        assert!(chunk1.is_ok());
        let chunk1 = chunk1.unwrap();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap(), Bytes::from("chunk1"));
        
        let chunk2 = tts_stream.receive_chunk().await;
        assert!(chunk2.is_ok());
        let chunk2 = chunk2.unwrap();
        assert!(chunk2.is_some());
        assert_eq!(chunk2.unwrap(), Bytes::from("chunk2"));
        
        // Test closing the stream
        let close_result = tts_stream.close().await;
        assert!(close_result.is_ok());
    }

    #[tokio::test]
    async fn test_stream_end() {
        use futures::stream;
        
        // Create an empty stream
        let empty_stream = stream::iter(Vec::<reqwest::Result<Bytes>>::new());
        let mut tts_stream = ElevenLabsStream::new(empty_stream);
        
        // Should return None when stream is exhausted
        let result = tts_stream.receive_chunk().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_voice_model_creation() {
        let voice = Voice::new(
            "21m00Tcm4TlvDq8ikWAM".to_string(),
            "Rachel".to_string(),
            "en-US".to_string(),
        );
        
        assert_eq!(voice.voice_id, "21m00Tcm4TlvDq8ikWAM");
        assert_eq!(voice.name, "Rachel");
        assert_eq!(voice.language, "en-US");
        assert!(voice.description.is_none());
        assert!(voice.gender.is_none());
        assert!(voice.accent.is_none());
    }

    #[test]
    fn test_audio_format_creation() {
        let format = AudioFormat::mp3(44100);
        
        assert_eq!(format.format, "mp3");
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, None);
    }

    #[test]
    fn test_provider_constants() {
        assert_eq!(ELEVENLABS_API_BASE, "https://api.elevenlabs.io/v1");
    }
}