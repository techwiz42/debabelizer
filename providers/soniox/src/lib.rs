use async_trait::async_trait;
pub use debabelizer_core::*;

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
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use url::Url;
use uuid::Uuid;

const SONIOX_WS_URL: &str = "wss://api.soniox.com/transcribe-websocket";

#[derive(Debug)]
pub struct SonioxProvider {
    api_key: String,
    model: String,
    auto_detect_language: bool,
}

impl SonioxProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Soniox API key not found".to_string()))?;
        
        let model = config
            .get_value("model")
            .and_then(|v| v.as_str())
            .unwrap_or("en")
            .to_string();
        
        let auto_detect_language = config
            .get_value("auto_detect_language")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        
        Ok(Self { 
            api_key, 
            model,
            auto_detect_language,
        })
    }
}

#[async_trait]
impl SttProvider for SonioxProvider {
    fn name(&self) -> &str {
        "soniox"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        // For batch transcription, we'll use streaming and collect results
        let config = StreamConfig {
            session_id: Uuid::new_v4(),
            format: audio.format.clone(),
            interim_results: false,
            language: if self.auto_detect_language { None } else { Some(self.model.clone()) },
            ..Default::default()
        };
        
        let mut stream = self.transcribe_stream(config).await?;
        
        // Send all audio at once
        stream.send_audio(&audio.data).await?;
        stream.close().await?;
        
        // Collect all results
        let mut full_text = String::new();
        let mut words = Vec::new();
        let mut confidence_sum = 0.0;
        let mut confidence_count = 0;
        let mut detected_language = None;
        
        while let Some(result) = stream.receive_transcript().await? {
            if result.is_final {
                full_text.push_str(&result.text);
                full_text.push(' ');
                
                if let Some(result_words) = result.words {
                    words.extend(result_words);
                }
                
                confidence_sum += result.confidence;
                confidence_count += 1;
                
                // Extract language from metadata if available
                if detected_language.is_none() {
                    if let Some(metadata) = &result.metadata {
                        if let Some(lang) = metadata.get("language").and_then(|v| v.as_str()) {
                            detected_language = Some(lang.to_string());
                        }
                    }
                }
            }
        }
        
        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            confidence: if confidence_count > 0 {
                confidence_sum / confidence_count as f32
            } else {
                0.0
            },
            language_detected: detected_language.or_else(|| {
                if !self.auto_detect_language {
                    Some(self.model.clone())
                } else {
                    None
                }
            }),
            duration: None,
            words: if words.is_empty() { None } else { Some(words) },
            metadata: None,
        })
    }
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let stream = SonioxStream::new(
            self.api_key.clone(), 
            self.model.clone(), 
            self.auto_detect_language,
            config
        ).await?;
        Ok(Box::new(stream))
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model {
                id: "auto".to_string(),
                name: "Auto-detect Language".to_string(),
                languages: vec!["multi".to_string()],
                capabilities: vec![
                    "streaming".to_string(), 
                    "timestamps".to_string(),
                    "language-detection".to_string()
                ],
            },
            Model {
                id: "en".to_string(),
                name: "English".to_string(),
                languages: vec!["en".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "es".to_string(),
                name: "Spanish".to_string(),
                languages: vec!["es".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "fr".to_string(),
                name: "French".to_string(),
                languages: vec!["fr".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "de".to_string(),
                name: "German".to_string(),
                languages: vec!["de".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "it".to_string(),
                name: "Italian".to_string(),
                languages: vec!["it".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "pt".to_string(),
                name: "Portuguese".to_string(),
                languages: vec!["pt".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "nl".to_string(),
                name: "Dutch".to_string(),
                languages: vec!["nl".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ru".to_string(),
                name: "Russian".to_string(),
                languages: vec!["ru".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "zh".to_string(),
                name: "Chinese".to_string(),
                languages: vec!["zh".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ja".to_string(),
                name: "Japanese".to_string(),
                languages: vec!["ja".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "ko".to_string(),
                name: "Korean".to_string(),
                languages: vec!["ko".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
            Model {
                id: "hi".to_string(),
                name: "Hindi".to_string(),
                languages: vec!["hi".to_string()],
                capabilities: vec!["streaming".to_string(), "timestamps".to_string()],
            },
        ])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(16000),
            AudioFormat::wav(8000),
            AudioFormat::wav(48000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

struct SonioxStream {
    websocket: Arc<Mutex<WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>>>,
    session_id: Uuid,
}

impl SonioxStream {
    async fn new(
        api_key: String, 
        model: String, 
        auto_detect_language: bool,
        config: StreamConfig
    ) -> Result<Self> {
        let url = Url::parse(SONIOX_WS_URL)
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        let (ws_stream, _) = connect_async(url.as_str())
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        let websocket = Arc::new(Mutex::new(ws_stream));
        
        // Send initial configuration
        let mut init_message = json!({
            "api_key": api_key,
            "include_nonfinal": config.interim_results,
            "punctuate": config.punctuate,
            "diarization": config.diarization,
        });
        
        // Handle language configuration
        if let Some(lang) = config.language {
            // Explicit language requested
            init_message["model"] = json!(lang);
        } else if auto_detect_language {
            // Auto-detect language
            init_message["model"] = json!("auto");
            init_message["detect_language"] = json!(true);
        } else {
            // Use default model
            init_message["model"] = json!(model);
        }
        
        // Add any additional metadata from config
        if let Some(metadata) = config.metadata {
            if let Some(obj) = metadata.as_object() {
                for (key, value) in obj {
                    init_message[key] = value.clone();
                }
            }
        }
        
        websocket
            .lock()
            .await
            .send(Message::Text(init_message.to_string()))
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        Ok(Self {
            websocket,
            session_id: config.session_id,
        })
    }
}

#[async_trait]
impl SttStream for SonioxStream {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()> {
        self.websocket
            .lock()
            .await
            .send(Message::Binary(chunk.to_vec()))
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))
    }
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>> {
        let mut ws = self.websocket.lock().await;
        
        match ws.next().await {
            Some(Ok(Message::Text(text))) => {
                let response: SonioxResponse = serde_json::from_str(&text)
                    .map_err(|e| DebabelizerError::Serialization(e))?;
                
                if let Some(result) = response.result {
                    let words = result.words.map(|words| {
                        words
                            .into_iter()
                            .map(|w| WordTiming {
                                word: w.text,
                                start: w.start_time,
                                end: w.start_time + w.duration,
                                confidence: w.confidence.unwrap_or(1.0),
                            })
                            .collect()
                    });
                    
                    // Build metadata including detected language
                    let mut metadata = serde_json::Map::new();
                    if let Some(lang) = result.language {
                        metadata.insert("language".to_string(), json!(lang));
                    }
                    
                    Ok(Some(StreamingResult {
                        session_id: self.session_id,
                        is_final: result.is_final,
                        text: result.transcript,
                        confidence: result.confidence.unwrap_or(1.0),
                        timestamp: chrono::Utc::now(),
                        words,
                        metadata: if metadata.is_empty() { None } else { Some(json!(metadata)) },
                    }))
                } else {
                    Ok(None)
                }
            }
            Some(Ok(Message::Close(_))) => Ok(None),
            Some(Err(e)) => Err(DebabelizerError::Provider(ProviderError::Network(
                e.to_string(),
            ))),
            None => Ok(None),
            _ => Ok(None),
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        self.websocket
            .lock()
            .await
            .close(None)
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))
    }
    
    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

#[derive(Debug, Deserialize)]
struct SonioxResponse {
    result: Option<SonioxResult>,
}

#[derive(Debug, Deserialize)]
struct SonioxResult {
    transcript: String,
    is_final: bool,
    confidence: Option<f32>,
    words: Option<Vec<SonioxWord>>,
    language: Option<String>,  // Language detected when using auto-detect
}

#[derive(Debug, Deserialize)]
struct SonioxWord {
    text: String,
    start_time: f32,
    duration: f32,
    confidence: Option<f32>,
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
        
        let provider = SonioxProvider::new(&config).await;
        assert!(provider.is_ok());
    }
    
    #[tokio::test]
    async fn test_auto_detect_language_config() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("auto_detect_language".to_string(), json!(true));
            map
        });
        
        let provider = SonioxProvider::new(&config).await.unwrap();
        assert!(provider.auto_detect_language);
    }
    
    #[test]
    fn test_supported_formats() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let formats = provider.supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.iter().any(|f| f.format == "wav"));
    }
    
    #[tokio::test]
    async fn test_list_models_includes_auto() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: true,
        };
        
        let models = provider.list_models().await.unwrap();
        assert!(models.iter().any(|m| m.id == "auto"));
        assert!(models.iter().any(|m| m.capabilities.contains(&"language-detection".to_string())));
    }

    #[test]
    fn test_provider_config_api_key_extraction() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("my-secret-key"));
            map.insert("model".to_string(), json!("es"));
            map.insert("auto_detect_language".to_string(), json!(false));
            map
        });
        
        assert_eq!(config.get_api_key(), Some("my-secret-key".to_string()));
        assert_eq!(config.get_value("model").unwrap().as_str(), Some("es"));
        assert_eq!(config.get_value("auto_detect_language").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn test_provider_config_missing_api_key() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("model".to_string(), json!("en"));
            map
        });
        
        assert!(config.get_api_key().is_none());
    }

    #[tokio::test]
    async fn test_provider_creation_fails_without_api_key() {
        let config = ProviderConfig::Simple(std::collections::HashMap::new());
        
        let result = SonioxProvider::new(&config).await;
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(matches!(error, DebabelizerError::Configuration(_)));
    }

    #[tokio::test]
    async fn test_provider_creation_with_custom_model() {
        let config = ProviderConfig::Simple({
            let mut map = std::collections::HashMap::new();
            map.insert("api_key".to_string(), json!("test-key"));
            map.insert("model".to_string(), json!("fr"));
            map
        });
        
        let provider = SonioxProvider::new(&config).await.unwrap();
        assert_eq!(provider.model, "fr");
    }

    #[test]
    fn test_provider_name() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        assert_eq!(provider.name(), "soniox");
    }

    #[test]
    fn test_supports_streaming() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_supported_formats_contains_multiple_sample_rates() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let formats = provider.supported_formats();
        assert!(formats.len() >= 2);
        
        let sample_rates: Vec<u32> = formats.iter().map(|f| f.sample_rate).collect();
        assert!(sample_rates.contains(&16000));
        assert!(sample_rates.contains(&48000));
    }

    #[tokio::test]
    async fn test_list_models_without_auto_detect() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        let models = provider.list_models().await.unwrap();
        assert!(models.iter().any(|m| m.id == "en"));
        assert!(models.iter().any(|m| m.id == "es"));
        assert!(models.iter().any(|m| m.id == "fr"));
        assert!(models.iter().any(|m| m.id == "de"));
        assert!(models.iter().any(|m| m.id == "hi"));
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        
        assert!(config.language.is_none());
        assert!(config.model.is_none());
        assert!(config.interim_results); // Default is true
        assert!(config.punctuate); // Default is true
        assert!(!config.profanity_filter); // Default is false
    }

    #[test]
    fn test_stream_config_custom() {
        let config = StreamConfig {
            session_id: uuid::Uuid::new_v4(),
            language: Some("en-US".to_string()),
            model: Some("en".to_string()),
            format: AudioFormat::wav(16000),
            interim_results: true,
            punctuate: true,
            profanity_filter: false,
            diarization: true,
            metadata: None,
            enable_word_time_offsets: false,
            enable_automatic_punctuation: false,
        };
        
        assert_eq!(config.language, Some("en-US".to_string()));
        assert_eq!(config.model, Some("en".to_string()));
        assert!(config.interim_results);
        assert!(config.punctuate);
        assert!(config.diarization);
    }

    #[test]
    fn test_soniox_response_parsing() {
        let json_response = r#"{
            "result": {
                "transcript": "Hello world",
                "is_final": true,
                "confidence": 0.95,
                "language": "en",
                "words": [
                    {
                        "text": "Hello",
                        "start_time": 0.0,
                        "duration": 0.5,
                        "confidence": 0.98
                    },
                    {
                        "text": "world",
                        "start_time": 0.6,
                        "duration": 0.4,
                        "confidence": 0.92
                    }
                ]
            }
        }"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        let result = response.result.unwrap();
        
        assert_eq!(result.transcript, "Hello world");
        assert!(result.is_final);
        assert_eq!(result.confidence, Some(0.95));
        assert_eq!(result.language, Some("en".to_string()));
        
        let words = result.words.unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].start_time, 0.0);
        assert_eq!(words[0].duration, 0.5);
        assert_eq!(words[0].confidence, Some(0.98));
    }

    #[test]
    fn test_soniox_response_minimal() {
        let json_response = r#"{
            "result": {
                "transcript": "Test",
                "is_final": false
            }
        }"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        let result = response.result.unwrap();
        
        assert_eq!(result.transcript, "Test");
        assert!(!result.is_final);
        assert!(result.confidence.is_none());
        assert!(result.language.is_none());
        assert!(result.words.is_none());
    }

    #[test]
    fn test_soniox_response_empty() {
        let json_response = r#"{}"#;
        
        let response: SonioxResponse = serde_json::from_str(json_response).unwrap();
        assert!(response.result.is_none());
    }

    #[test]
    fn test_websocket_url_construction() {
        let _provider = SonioxProvider {
            api_key: "test-key".to_string(),
            model: "en".to_string(),
            auto_detect_language: false,
        };
        
        // Test URL construction logic
        let base_url = "wss://api.soniox.com/transcribe-websocket";
        assert_eq!(SONIOX_WS_URL, base_url);
    }

    #[test]
    fn test_word_timing_conversion() {
        let soniox_word = SonioxWord {
            text: "hello".to_string(),
            start_time: 1.0,
            duration: 0.5,
            confidence: Some(0.95),
        };
        
        let word_timing = WordTiming {
            word: soniox_word.text.clone(),
            start: soniox_word.start_time,
            end: soniox_word.start_time + soniox_word.duration,
            confidence: soniox_word.confidence.unwrap_or(1.0),
        };
        
        assert_eq!(word_timing.word, "hello");
        assert_eq!(word_timing.start, 1.0);
        assert_eq!(word_timing.end, 1.5);
        assert_eq!(word_timing.confidence, 0.95);
    }

    #[test]
    fn test_auto_detect_language_provider() {
        let provider = SonioxProvider {
            api_key: "test".to_string(),
            model: "auto".to_string(),
            auto_detect_language: true,
        };
        
        assert_eq!(provider.model, "auto");
        assert!(provider.auto_detect_language);
    }

    #[test]
    fn test_provider_constants() {
        assert_eq!(SONIOX_WS_URL, "wss://api.soniox.com/transcribe-websocket");
    }

    // Test streaming result creation
    #[test]
    fn test_streaming_result_creation() {
        let session_id = uuid::Uuid::new_v4();
        let words = vec![
            WordTiming {
                word: "hello".to_string(),
                start: 0.0,
                end: 0.5,
                confidence: 0.98,
            },
            WordTiming {
                word: "world".to_string(),
                start: 0.6,
                end: 1.0,
                confidence: 0.92,
            },
        ];
        
        let mut result = StreamingResult::new(session_id, "hello world".to_string(), true, 0.95);
        result.words = Some(words.clone());
        
        assert_eq!(result.session_id, session_id);
        assert!(result.is_final);
        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.words.unwrap().len(), 2);
    }

    #[test]
    fn test_model_info_creation() {
        let model = Model {
            id: "soniox-en".to_string(),
            name: "Soniox English Model".to_string(),
            languages: vec!["en".to_string()],
            capabilities: vec!["transcription".to_string(), "streaming".to_string()],
        };
        
        assert_eq!(model.id, "soniox-en");
        assert_eq!(model.name, "Soniox English Model");
        assert_eq!(model.languages.len(), 1);
        assert_eq!(model.capabilities.len(), 2);
    }
}