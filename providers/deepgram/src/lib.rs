use async_trait::async_trait;
use chrono::Utc;
use debabelizer_core::{
    AudioData, AudioFormat, DebabelizerError, Model, ProviderError, Result, SttProvider,
    SttStream, StreamConfig, StreamingResult, TranscriptionResult, WordTiming,
};
use futures::{SinkExt, StreamExt};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use tracing::{debug, error, info};
use url::Url;
use uuid::Uuid;

const DEEPGRAM_API_BASE: &str = "https://api.deepgram.com/v1";
const DEEPGRAM_WS_BASE: &str = "wss://api.deepgram.com/v1/listen";

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
pub struct DeepgramProvider {
    client: Client,
    api_key: String,
    model: String,
    default_language: String,
}

impl DeepgramProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Deepgram API key not found".to_string()))?;
        
        let model = config
            .get_value("model")
            .and_then(|v| v.as_str())
            .unwrap_or("nova-2")
            .to_string();
        
        let default_language = config
            .get_value("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US")
            .to_string();
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            model,
            default_language,
        })
    }
    
    fn get_supported_languages() -> Vec<String> {
        vec![
            "en".to_string(), "en-US".to_string(), "en-GB".to_string(), "en-AU".to_string(),
            "en-IN".to_string(), "en-NZ".to_string(), "en-CA".to_string(),
            "es".to_string(), "es-ES".to_string(), "es-419".to_string(),
            "fr".to_string(), "fr-CA".to_string(),
            "de".to_string(), "it".to_string(), "pt".to_string(), "pt-BR".to_string(),
            "ru".to_string(), "hi".to_string(), "ja".to_string(), "ko".to_string(),
            "zh".to_string(), "zh-CN".to_string(), "zh-TW".to_string(),
            "nl".to_string(), "sv".to_string(), "da".to_string(), "no".to_string(),
            "fi".to_string(), "pl".to_string(), "tr".to_string(), "ar".to_string(),
            "th".to_string(), "vi".to_string(), "uk".to_string(), "cs".to_string(),
            "sk".to_string(), "hu".to_string(), "ro".to_string(), "bg".to_string(),
            "hr".to_string(), "sl".to_string(), "et".to_string(), "lv".to_string(),
            "lt".to_string(), "mt".to_string(), "ga".to_string(), "cy".to_string(),
            "is".to_string(),
        ]
    }
    
    fn normalize_language(language: &str) -> Result<String> {
        let supported = Self::get_supported_languages();
        
        // Auto-detection
        if language.to_lowercase() == "auto" {
            return Ok("auto".to_string());
        }
        
        // Try short code mapping first for common ambiguous cases
        let language_map: HashMap<&str, &str> = [
            ("en", "en-US"), ("es", "es"), ("fr", "fr"), ("de", "de"),
            ("zh", "zh"), ("ja", "ja"), ("ko", "ko"), ("pt", "pt"),
            ("ru", "ru"), ("it", "it"), ("nl", "nl"), ("pl", "pl"),
            ("tr", "tr"), ("ar", "ar"), ("hi", "hi"),
        ].iter().cloned().collect();
        
        let short_code = language.split("-").next().unwrap_or(language).to_lowercase();
        if let Some(mapped) = language_map.get(short_code.as_str()) {
            return Ok(mapped.to_string());
        }
        
        // Direct match for specific locales
        if supported.contains(&language.to_string()) {
            return Ok(language.to_string());
        }
        
        // Check case-insensitive
        for lang in &supported {
            if lang.to_lowercase() == language.to_lowercase() {
                return Ok(lang.clone());
            }
        }
        
        Err(DebabelizerError::Provider(ProviderError::UnsupportedFeature(
            format!("Language '{}' is not supported by Deepgram", language)
        )))
    }
    
    async fn make_prerecorded_request(&self, audio_data: &[u8], options: &PrerecordedOptions) -> Result<DeepgramResponse> {
        let url = format!("{}/listen", DEEPGRAM_API_BASE);
        
        // Build query parameters with owned strings for boolean values
        let punctuate_str = options.punctuate.to_string();
        let smart_format_str = options.smart_format.to_string();
        let utterances_str = options.utterances.to_string();
        let diarize_str = options.diarize.to_string();
        
        let mut params = vec![
            ("model", options.model.as_str()),
            ("punctuate", punctuate_str.as_str()),
            ("smart_format", smart_format_str.as_str()),
            ("utterances", utterances_str.as_str()),
            ("diarize", diarize_str.as_str()),
        ];
        
        if let Some(ref language) = options.language {
            if language != "auto" {
                params.push(("language", language));
            } else {
                params.push(("detect_language", "true"));
            }
        }
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "audio/wav") // Deepgram auto-detects format
            .query(&params)
            .body(audio_data.to_vec())
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let deepgram_response: DeepgramResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                Ok(deepgram_response)
            }
            StatusCode::UNAUTHORIZED => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Deepgram API key".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Deepgram API error: {}", error_text)
                )))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PrerecordedOptions {
    model: String,
    language: Option<String>,
    punctuate: bool,
    smart_format: bool,
    utterances: bool,
    diarize: bool,
}

impl Default for PrerecordedOptions {
    fn default() -> Self {
        Self {
            model: "nova-2".to_string(),
            language: Some("en-US".to_string()),
            punctuate: true,
            smart_format: true,
            utterances: false,
            diarize: false,
        }
    }
}

#[async_trait]
impl SttProvider for DeepgramProvider {
    fn name(&self) -> &str {
        "deepgram"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        let options = PrerecordedOptions {
            model: self.model.clone(),
            language: Some(self.default_language.clone()),
            ..Default::default()
        };
        
        let response = self.make_prerecorded_request(&audio.data, &options).await?;
        
        // Parse the response
        if response.results.channels.is_empty() {
            return Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                "No channels in Deepgram response".to_string()
            )));
        }
        
        let channel = &response.results.channels[0];
        if channel.alternatives.is_empty() {
            return Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                "No alternatives in Deepgram response".to_string()
            )));
        }
        
        let best_alt = &channel.alternatives[0];
        
        // Extract words with timestamps
        let words = if let Some(ref word_list) = best_alt.words {
            word_list.iter().map(|w| WordTiming {
                word: w.word.clone(),
                start: w.start,
                end: w.end,
                confidence: w.confidence,
            }).collect()
        } else {
            Vec::new()
        };
        
        // Get detected language
        let detected_language = response.metadata.as_ref()
            .and_then(|m| m.detected_language.clone())
            .or_else(|| Some(self.default_language.clone()));
        
        // Get duration
        let duration = response.metadata.as_ref()
            .and_then(|m| m.duration);
        
        let mut metadata = serde_json::Map::new();
        metadata.insert("provider".to_string(), json!("deepgram"));
        metadata.insert("model".to_string(), json!(self.model));
        if let Some(ref meta) = response.metadata {
            if let Some(model_info) = &meta.model_info {
                metadata.insert("model_info".to_string(), json!(model_info));
            }
        }
        
        Ok(TranscriptionResult {
            text: best_alt.transcript.clone(),
            confidence: best_alt.confidence,
            language_detected: detected_language,
            duration,
            words: if words.is_empty() { None } else { Some(words) },
            metadata: Some(json!(metadata)),
        })
    }
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let stream = DeepgramStream::new(
            self.api_key.clone(),
            self.model.clone(),
            config,
        ).await?;
        Ok(Box::new(stream))
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model {
                id: "nova-2".to_string(),
                name: "Nova-2".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "streaming".to_string(),
                    "timestamps".to_string(),
                    "language-detection".to_string(),
                    "diarization".to_string(),
                    "punctuation".to_string(),
                    "smart-formatting".to_string(),
                ],
            },
            Model {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "streaming".to_string(),
                    "timestamps".to_string(),
                    "language-detection".to_string(),
                    "diarization".to_string(),
                ],
            },
            Model {
                id: "enhanced".to_string(),
                name: "Enhanced".to_string(),
                languages: vec!["en-US".to_string()],
                capabilities: vec![
                    "streaming".to_string(),
                    "timestamps".to_string(),
                    "high-accuracy".to_string(),
                ],
            },
            Model {
                id: "base".to_string(),
                name: "Base".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "streaming".to_string(),
                    "timestamps".to_string(),
                    "cost-effective".to_string(),
                ],
            },
        ])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(16000),
            AudioFormat::wav(44100),
            AudioFormat::wav(48000),
            AudioFormat::mp3(16000),
            AudioFormat::mp3(44100),
            AudioFormat::opus(16000),
            AudioFormat::opus(48000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

struct DeepgramStream {
    websocket: Arc<Mutex<WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>>>,
    session_id: Uuid,
    _model: String,
}

impl DeepgramStream {
    async fn new(
        api_key: String,
        model: String,
        config: StreamConfig,
    ) -> Result<Self> {
        // Build WebSocket URL with parameters
        let mut url = Url::parse(DEEPGRAM_WS_BASE)
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        url.query_pairs_mut()
            .append_pair("model", &model)
            .append_pair("punctuate", &config.punctuate.to_string())
            .append_pair("smart_format", "true")
            .append_pair("interim_results", &config.interim_results.to_string())
            .append_pair("utterance_end_ms", "1000")
            .append_pair("vad_events", "true")
            .append_pair("encoding", "linear16")
            .append_pair("sample_rate", &config.format.sample_rate.to_string())
            .append_pair("channels", &config.format.channels.to_string());
        
        // Add language if specified
        if let Some(ref language) = config.language {
            if language == "auto" {
                url.query_pairs_mut().append_pair("detect_language", "true");
            } else {
                let normalized_lang = DeepgramProvider::normalize_language(language)?;
                url.query_pairs_mut().append_pair("language", &normalized_lang);
            }
        }
        
        // Build request with authorization header
        let mut request_builder = tokio_tungstenite::tungstenite::handshake::client::Request::builder();
        request_builder = request_builder
            .uri(url.as_str())
            .header("Authorization", format!("Token {}", api_key));
        
        let request = request_builder
            .body(())
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        info!("Connected to Deepgram WebSocket for session {}", config.session_id);
        
        Ok(Self {
            websocket: Arc::new(Mutex::new(ws_stream)),
            session_id: config.session_id,
            _model: model,
        })
    }
}

#[async_trait]
impl SttStream for DeepgramStream {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()> {
        let mut ws = self.websocket.lock().await;
        ws.send(Message::Binary(chunk.to_vec()))
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))
    }
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>> {
        let mut ws = self.websocket.lock().await;
        
        match ws.next().await {
            Some(Ok(Message::Text(text))) => {
                debug!("Received Deepgram message: {}", text);
                
                let response: DeepgramStreamingResponse = serde_json::from_str(&text)
                    .map_err(|e| {
                        error!("Failed to parse Deepgram response: {}", e);
                        DebabelizerError::Serialization(e)
                    })?;
                
                // Handle different message types
                match response {
                    DeepgramStreamingResponse::Results(results) => {
                        if let Some(channel) = results.channel.alternatives.first() {
                            let words = if let Some(ref word_list) = channel.words {
                                Some(word_list.iter().map(|w| WordTiming {
                                    word: w.word.clone(),
                                    start: w.start,
                                    end: w.end,
                                    confidence: w.confidence,
                                }).collect())
                            } else {
                                None
                            };
                            
                            let mut metadata = serde_json::Map::new();
                            metadata.insert("provider".to_string(), json!("deepgram"));
                            if let Some(detected_lang) = &results.metadata.as_ref().and_then(|m| m.detected_language.clone()) {
                                metadata.insert("detected_language".to_string(), json!(detected_lang));
                            }
                            
                            return Ok(Some(StreamingResult {
                                session_id: self.session_id,
                                is_final: results.is_final,
                                text: channel.transcript.clone(),
                                confidence: channel.confidence,
                                timestamp: Utc::now(),
                                words,
                                metadata: Some(json!(metadata)),
                            }));
                        }
                    }
                    DeepgramStreamingResponse::Metadata(_) => {
                        debug!("Received Deepgram metadata message");
                        // Metadata messages don't contain transcription results
                        return Ok(None);
                    }
                    DeepgramStreamingResponse::Error(error) => {
                        error!("Deepgram streaming error: {:?}", error);
                        return Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                            format!("Deepgram streaming error: {:?}", error)
                        )));
                    }
                }
                
                Ok(None)
            }
            Some(Ok(Message::Close(_))) => {
                info!("Deepgram WebSocket connection closed");
                Ok(None)
            }
            Some(Err(e)) => {
                error!("WebSocket error: {}", e);
                Err(DebabelizerError::Provider(ProviderError::Network(e.to_string())))
            }
            None => {
                info!("Deepgram WebSocket stream ended");
                Ok(None)
            }
            _ => Ok(None), // Ignore other message types
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        let mut ws = self.websocket.lock().await;
        ws.close(None)
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))
    }
    
    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// Deepgram API Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramResponse {
    metadata: Option<DeepgramMetadata>,
    results: DeepgramResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramMetadata {
    #[serde(rename = "transaction_key")]
    transaction_key: Option<String>,
    #[serde(rename = "request_id")]
    request_id: Option<String>,
    #[serde(rename = "sha256")]
    sha256: Option<String>,
    #[serde(rename = "created")]
    created: Option<String>,
    #[serde(rename = "duration")]
    duration: Option<f32>,
    #[serde(rename = "channels")]
    channels: Option<u32>,
    #[serde(rename = "models")]
    models: Option<Vec<String>>,
    #[serde(rename = "model_info")]
    model_info: Option<serde_json::Value>,
    #[serde(rename = "detected_language")]
    detected_language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramResults {
    channels: Vec<DeepgramChannel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramChannel {
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramAlternative {
    transcript: String,
    confidence: f32,
    words: Option<Vec<DeepgramWord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramWord {
    word: String,
    start: f32,
    end: f32,
    confidence: f32,
    #[serde(rename = "punctuated_word")]
    punctuated_word: Option<String>,
}

// Streaming response types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum DeepgramStreamingResponse {
    Results(DeepgramStreamingResults),
    Metadata(DeepgramStreamingMetadata),
    Error(DeepgramStreamingError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramStreamingResults {
    #[serde(rename = "type")]
    message_type: String,
    channel_index: Vec<u32>,
    duration: f32,
    start: f32,
    is_final: bool,
    channel: DeepgramStreamingChannel,
    metadata: Option<DeepgramStreamingResultMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramStreamingChannel {
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramStreamingResultMetadata {
    #[serde(rename = "request_id")]
    request_id: Option<String>,
    #[serde(rename = "model_info")]
    model_info: Option<serde_json::Value>,
    #[serde(rename = "detected_language")]
    detected_language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramStreamingMetadata {
    #[serde(rename = "type")]
    message_type: String,
    #[serde(rename = "transaction_key")]
    transaction_key: String,
    #[serde(rename = "request_id")]
    request_id: String,
    #[serde(rename = "sha256")]
    sha256: String,
    #[serde(rename = "created")]
    created: String,
    #[serde(rename = "duration")]
    duration: Option<f32>,
    #[serde(rename = "channels")]
    channels: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepgramStreamingError {
    #[serde(rename = "type")]
    message_type: String,
    description: String,
    message: String,
    variant: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    
    fn create_test_config() -> ProviderConfig {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key"));
        config.insert("model".to_string(), json!("nova-2"));
        config.insert("language".to_string(), json!("en-US"));
        ProviderConfig::Simple(config)
    }
    
    #[tokio::test]
    async fn test_provider_creation() {
        let config = create_test_config();
        let provider = DeepgramProvider::new(&config).await;
        assert!(provider.is_ok());
        
        let provider = provider.unwrap();
        assert_eq!(provider.name(), "deepgram");
        assert_eq!(provider.model, "nova-2");
        assert_eq!(provider.default_language, "en-US");
    }
    
    #[tokio::test]
    async fn test_provider_creation_with_defaults() {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-key"));
        let config = ProviderConfig::Simple(config);
        
        let provider = DeepgramProvider::new(&config).await.unwrap();
        assert_eq!(provider.model, "nova-2");
        assert_eq!(provider.default_language, "en-US");
    }
    
    #[tokio::test]
    async fn test_provider_creation_fails_without_api_key() {
        let config = ProviderConfig::Simple(HashMap::new());
        let result = DeepgramProvider::new(&config).await;
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(matches!(error, DebabelizerError::Configuration(_)));
    }
    
    #[test]
    fn test_supported_languages() {
        let languages = DeepgramProvider::get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"es".to_string()));
        assert!(languages.contains(&"fr".to_string()));
        assert!(languages.contains(&"de".to_string()));
        assert!(languages.contains(&"zh".to_string()));
        assert!(languages.contains(&"ja".to_string()));
    }
    
    #[test]
    fn test_language_normalization() {
        // Test direct matches
        assert_eq!(DeepgramProvider::normalize_language("en-US").unwrap(), "en-US");
        assert_eq!(DeepgramProvider::normalize_language("es").unwrap(), "es");
        
        // Test auto-detection
        assert_eq!(DeepgramProvider::normalize_language("auto").unwrap(), "auto");
        
        // Test short code mapping
        assert_eq!(DeepgramProvider::normalize_language("en").unwrap(), "en-US");
        assert_eq!(DeepgramProvider::normalize_language("fr").unwrap(), "fr");
        
        // Test case insensitive
        assert_eq!(DeepgramProvider::normalize_language("EN-US").unwrap(), "en-US");
        
        // Test unsupported language
        assert!(DeepgramProvider::normalize_language("xyz").is_err());
    }
    
    #[tokio::test]
    async fn test_list_models() {
        let provider = DeepgramProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model: "nova-2".to_string(),
            default_language: "en-US".to_string(),
        };
        
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        
        // Check for Nova-2 model
        let nova2 = models.iter().find(|m| m.id == "nova-2");
        assert!(nova2.is_some());
        
        let nova2 = nova2.unwrap();
        assert_eq!(nova2.name, "Nova-2");
        assert!(nova2.capabilities.contains(&"streaming".to_string()));
        assert!(nova2.capabilities.contains(&"language-detection".to_string()));
        assert!(!nova2.languages.is_empty());
    }
    
    #[test]
    fn test_supported_formats() {
        let provider = DeepgramProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model: "nova-2".to_string(),
            default_language: "en-US".to_string(),
        };
        
        let formats = provider.supported_formats();
        assert!(!formats.is_empty());
        
        // Check for common formats
        assert!(formats.iter().any(|f| f.format == "wav" && f.sample_rate == 16000));
        assert!(formats.iter().any(|f| f.format == "mp3"));
        assert!(formats.iter().any(|f| f.format == "opus"));
    }
    
    #[test]
    fn test_supports_streaming() {
        let provider = DeepgramProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model: "nova-2".to_string(),
            default_language: "en-US".to_string(),
        };
        
        assert!(provider.supports_streaming());
    }
    
    #[test]
    fn test_provider_config_extraction() {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key-123"));
        config.insert("model".to_string(), json!("nova"));
        config.insert("language".to_string(), json!("fr"));
        config.insert("custom_param".to_string(), json!("custom_value"));
        
        let provider_config = ProviderConfig::Simple(config);
        
        assert_eq!(provider_config.get_api_key().unwrap(), "test-api-key-123");
        assert_eq!(provider_config.get_value("model").unwrap().as_str().unwrap(), "nova");
        assert_eq!(provider_config.get_value("language").unwrap().as_str().unwrap(), "fr");
        assert_eq!(provider_config.get_value("custom_param").unwrap().as_str().unwrap(), "custom_value");
        assert!(provider_config.get_value("missing_param").is_none());
    }
    
    #[test]
    fn test_deepgram_response_parsing() {
        let response_json = r#"
        {
            "metadata": {
                "transaction_key": "deprecated",
                "request_id": "test-request-id",
                "duration": 5.2,
                "channels": 1,
                "detected_language": "en"
            },
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello, how are you today?",
                                "confidence": 0.95,
                                "words": [
                                    {
                                        "word": "hello",
                                        "start": 0.0,
                                        "end": 0.5,
                                        "confidence": 0.98,
                                        "punctuated_word": "Hello,"
                                    },
                                    {
                                        "word": "how",
                                        "start": 0.6,
                                        "end": 0.8,
                                        "confidence": 0.96,
                                        "punctuated_word": "how"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        "#;
        
        let response: DeepgramResponse = serde_json::from_str(response_json).unwrap();
        
        assert!(response.metadata.is_some());
        let metadata = response.metadata.unwrap();
        assert_eq!(metadata.duration, Some(5.2));
        assert_eq!(metadata.channels, Some(1));
        assert_eq!(metadata.detected_language, Some("en".to_string()));
        
        assert_eq!(response.results.channels.len(), 1);
        let channel = &response.results.channels[0];
        assert_eq!(channel.alternatives.len(), 1);
        
        let alternative = &channel.alternatives[0];
        assert_eq!(alternative.transcript, "Hello, how are you today?");
        assert_eq!(alternative.confidence, 0.95);
        
        let words = alternative.words.as_ref().unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[0].end, 0.5);
        assert_eq!(words[0].confidence, 0.98);
        assert_eq!(words[0].punctuated_word, Some("Hello,".to_string()));
    }
    
    #[test]
    fn test_streaming_response_parsing() {
        let response_json = r#"
        {
            "type": "Results",
            "channel_index": [0],
            "duration": 2.1,
            "start": 0.0,
            "is_final": true,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "Testing streaming response",
                        "confidence": 0.92,
                        "words": [
                            {
                                "word": "testing",
                                "start": 0.0,
                                "end": 0.6,
                                "confidence": 0.94
                            },
                            {
                                "word": "streaming",
                                "start": 0.7,
                                "end": 1.3,
                                "confidence": 0.91
                            }
                        ]
                    }
                ]
            },
            "metadata": {
                "request_id": "test-streaming-request",
                "detected_language": "en"
            }
        }
        "#;
        
        let response: DeepgramStreamingResponse = serde_json::from_str(response_json).unwrap();
        
        match response {
            DeepgramStreamingResponse::Results(results) => {
                assert_eq!(results.message_type, "Results");
                assert_eq!(results.duration, 2.1);
                assert_eq!(results.start, 0.0);
                assert!(results.is_final);
                
                let alternative = &results.channel.alternatives[0];
                assert_eq!(alternative.transcript, "Testing streaming response");
                assert_eq!(alternative.confidence, 0.92);
                
                let words = alternative.words.as_ref().unwrap();
                assert_eq!(words.len(), 2);
                assert_eq!(words[0].word, "testing");
                assert_eq!(words[1].word, "streaming");
                
                let metadata = results.metadata.unwrap();
                assert_eq!(metadata.request_id, Some("test-streaming-request".to_string()));
                assert_eq!(metadata.detected_language, Some("en".to_string()));
            }
            _ => panic!("Expected Results variant"),
        }
    }
    
    #[test]
    fn test_streaming_metadata_parsing() {
        let metadata_json = r#"
        {
            "type": "Metadata",
            "transaction_key": "deprecated",
            "request_id": "test-metadata-request",
            "sha256": "test-sha256",
            "created": "2024-01-01T12:00:00Z",
            "duration": 10.5,
            "channels": 2
        }
        "#;
        
        let response: DeepgramStreamingResponse = serde_json::from_str(metadata_json).unwrap();
        
        match response {
            DeepgramStreamingResponse::Metadata(metadata) => {
                assert_eq!(metadata.message_type, "Metadata");
                assert_eq!(metadata.request_id, "test-metadata-request");
                assert_eq!(metadata.duration, Some(10.5));
                assert_eq!(metadata.channels, 2);
            }
            _ => panic!("Expected Metadata variant"),
        }
    }
    
    #[test]
    fn test_streaming_error_parsing() {
        let error_json = r#"
        {
            "type": "error",
            "description": "Invalid audio format",
            "message": "The provided audio format is not supported",
            "variant": "audio_format_error"
        }
        "#;
        
        let response: DeepgramStreamingResponse = serde_json::from_str(error_json).unwrap();
        
        match response {
            DeepgramStreamingResponse::Error(error) => {
                assert_eq!(error.message_type, "error");
                assert_eq!(error.description, "Invalid audio format");
                assert_eq!(error.message, "The provided audio format is not supported");
                assert_eq!(error.variant, Some("audio_format_error".to_string()));
            }
            _ => panic!("Expected Error variant"),
        }
    }
    
    #[test]
    fn test_prerecorded_options_default() {
        let options = PrerecordedOptions::default();
        assert_eq!(options.model, "nova-2");
        assert_eq!(options.language, Some("en-US".to_string()));
        assert!(options.punctuate);
        assert!(options.smart_format);
        assert!(!options.utterances);
        assert!(!options.diarize);
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(DEEPGRAM_API_BASE, "https://api.deepgram.com/v1");
        assert_eq!(DEEPGRAM_WS_BASE, "wss://api.deepgram.com/v1/listen");
    }
    
    #[tokio::test]
    async fn test_stream_config_validation() {
        let config = StreamConfig {
            session_id: Uuid::new_v4(),
            language: Some("en-US".to_string()),
            model: Some("nova-2".to_string()),
            format: AudioFormat::wav(16000),
            interim_results: true,
            punctuate: true,
            profanity_filter: false,
            diarization: false,
            metadata: None,
            enable_word_time_offsets: true,
            enable_automatic_punctuation: true,
            enable_language_identification: false,
        };
        
        // Validate language normalization works with config
        let normalized = DeepgramProvider::normalize_language(
            config.language.as_ref().unwrap()
        ).unwrap();
        assert_eq!(normalized, "en-US");
        
        // Validate format is supported
        let provider = DeepgramProvider {
            client: Client::new(),
            api_key: "test".to_string(),
            model: "nova-2".to_string(),
            default_language: "en-US".to_string(),
        };
        
        let supported_formats = provider.supported_formats();
        assert!(supported_formats.iter().any(|f| {
            f.format == config.format.format && f.sample_rate == config.format.sample_rate
        }));
    }
}