use async_trait::async_trait;
use chrono::Utc;
use debabelizer_core::{
    AudioData, AudioFormat, DebabelizerError, Model, ProviderError, Result, SttProvider,
    SttStream, StreamConfig, StreamingResult, TranscriptionResult, WordTiming,
    TtsProvider, TtsStream, Voice, SynthesisResult, SynthesisOptions,
};
use futures::{SinkExt, StreamExt};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use url::Url;
use uuid::Uuid;
use bytes::Bytes;

const AZURE_STT_REST_API_BASE: &str = "https://{region}.stt.speech.microsoft.com/speech/recognition";
const AZURE_STT_WS_BASE: &str = "wss://{region}.stt.speech.microsoft.com/speech/recognition";
const AZURE_TTS_API_BASE: &str = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1";

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
    
    pub fn get_region(&self) -> Option<String> {
        match self {
            Self::Simple(map) => map
                .get("region")
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

pub struct AzureSttProvider {
    client: Client,
    api_key: String,
    region: String,
    language: String,
    endpoint_id: Option<String>,
    #[allow(dead_code)]
    enable_dictation: bool,
    profanity_filter: bool,
    enable_speaker_identification: bool,
    max_speakers: u32,
}

impl AzureSttProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Azure API key not found".to_string()))?;
        
        let region = config
            .get_region()
            .ok_or_else(|| DebabelizerError::Configuration("Azure region not found".to_string()))?;
            
        let language = config
            .get_value("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US")
            .to_string();
            
        let endpoint_id = config
            .get_value("endpoint_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
            
        let enable_dictation = config
            .get_value("enable_dictation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            
        let profanity_filter = config
            .get_value("profanity_filter")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
            
        let enable_speaker_identification = config
            .get_value("enable_speaker_identification")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            
        let max_speakers = config
            .get_value("max_speakers")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32;
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            region,
            language,
            endpoint_id,
            enable_dictation,
            profanity_filter,
            enable_speaker_identification,
            max_speakers,
        })
    }
    
    fn get_supported_languages() -> Vec<String> {
        vec![
            "ar-SA".to_string(), "bg-BG".to_string(), "ca-ES".to_string(), "cs-CZ".to_string(),
            "da-DK".to_string(), "de-DE".to_string(), "el-GR".to_string(), "en-AU".to_string(),
            "en-CA".to_string(), "en-GB".to_string(), "en-IN".to_string(), "en-NZ".to_string(),
            "en-US".to_string(), "es-ES".to_string(), "es-MX".to_string(), "et-EE".to_string(),
            "fi-FI".to_string(), "fr-CA".to_string(), "fr-FR".to_string(), "gu-IN".to_string(),
            "he-IL".to_string(), "hi-IN".to_string(), "hr-HR".to_string(), "hu-HU".to_string(),
            "it-IT".to_string(), "ja-JP".to_string(), "ko-KR".to_string(), "lt-LT".to_string(),
            "lv-LV".to_string(), "ms-MY".to_string(), "nb-NO".to_string(), "nl-NL".to_string(),
            "pl-PL".to_string(), "pt-BR".to_string(), "pt-PT".to_string(), "ro-RO".to_string(),
            "ru-RU".to_string(), "sk-SK".to_string(), "sl-SI".to_string(), "sv-SE".to_string(),
            "ta-IN".to_string(), "te-IN".to_string(), "th-TH".to_string(), "tr-TR".to_string(),
            "uk-UA".to_string(), "vi-VN".to_string(), "zh-CN".to_string(), "zh-HK".to_string(),
            "zh-TW".to_string(), "af-ZA".to_string(), "am-ET".to_string(), "az-AZ".to_string(),
            "bn-IN".to_string(), "bs-BA".to_string(), "cy-GB".to_string(), "eu-ES".to_string(),
            "fa-IR".to_string(), "ga-IE".to_string(), "gl-ES".to_string(), "id-ID".to_string(),
            "is-IS".to_string(), "jv-ID".to_string(), "ka-GE".to_string(), "kk-KZ".to_string(),
            "km-KH".to_string(), "kn-IN".to_string(), "lo-LA".to_string(), "mk-MK".to_string(),
            "ml-IN".to_string(), "mn-MN".to_string(), "mr-IN".to_string(), "mt-MT".to_string(),
            "my-MM".to_string(), "ne-NP".to_string(), "pa-IN".to_string(), "ps-AF".to_string(),
            "si-LK".to_string(), "so-SO".to_string(), "sq-AL".to_string(), "su-ID".to_string(),
            "sw-KE".to_string(), "sw-TZ".to_string(), "ur-PK".to_string(), "uz-UZ".to_string(),
            "zu-ZA".to_string(),
        ]
    }
    
    fn normalize_language(language: &str) -> Result<String> {
        let supported = Self::get_supported_languages();
        
        // Language mapping for common short codes
        let language_map: HashMap<&str, &str> = [
            ("en", "en-US"), ("es", "es-ES"), ("fr", "fr-FR"), ("de", "de-DE"),
            ("it", "it-IT"), ("pt", "pt-BR"), ("ru", "ru-RU"), ("ja", "ja-JP"),
            ("ko", "ko-KR"), ("zh", "zh-CN"), ("ar", "ar-SA"), ("hi", "hi-IN"),
            ("nl", "nl-NL"), ("pl", "pl-PL"), ("tr", "tr-TR"), ("sv", "sv-SE"),
            ("da", "da-DK"), ("no", "nb-NO"), ("fi", "fi-FI"), ("cs", "cs-CZ"),
            ("hu", "hu-HU"), ("el", "el-GR"), ("he", "he-IL"), ("th", "th-TH"),
            ("vi", "vi-VN"), ("id", "id-ID"), ("ms", "ms-MY"), ("ro", "ro-RO"),
            ("uk", "uk-UA"), ("bg", "bg-BG"), ("hr", "hr-HR"), ("sk", "sk-SK"),
            ("sl", "sl-SI"), ("et", "et-EE"), ("lv", "lv-LV"), ("lt", "lt-LT"),
            ("ca", "ca-ES"), ("eu", "eu-ES"), ("gl", "gl-ES"), ("af", "af-ZA"),
            ("sq", "sq-AL"), ("am", "am-ET"), ("hy", "hy-AM"), ("az", "az-AZ"),
            ("bn", "bn-IN"), ("bs", "bs-BA"), ("my", "my-MM"), ("cy", "cy-GB"),
            ("gu", "gu-IN"), ("is", "is-IS"), ("jv", "jv-ID"), ("kn", "kn-IN"),
            ("km", "km-KH"), ("lo", "lo-LA"), ("mk", "mk-MK"), ("ml", "ml-IN"),
            ("mn", "mn-MN"), ("mr", "mr-IN"), ("ne", "ne-NP"), ("ps", "ps-AF"),
            ("si", "si-LK"), ("su", "su-ID"), ("sw", "sw-KE"), ("ta", "ta-IN"),
            ("te", "te-IN"), ("ur", "ur-PK"), ("uz", "uz-UZ"), ("zu", "zu-ZA"),
        ].iter().cloned().collect();
        
        let short_code = language.split("-").next().unwrap_or(language).to_lowercase();
        if let Some(mapped) = language_map.get(short_code.as_str()) {
            return Ok(mapped.to_string());
        }
        
        // Direct match
        if supported.contains(&language.to_string()) {
            return Ok(language.to_string());
        }
        
        // Case-insensitive match
        for lang in &supported {
            if lang.to_lowercase() == language.to_lowercase() {
                return Ok(lang.clone());
            }
        }
        
        Err(DebabelizerError::Provider(ProviderError::UnsupportedFeature(
            format!("Language '{}' is not supported by Azure Speech", language)
        )))
    }
    
    async fn make_recognize_request(&self, audio_data: &[u8], language: &str) -> Result<AzureSpeechResponse> {
        let url = AZURE_STT_REST_API_BASE
            .replace("{region}", &self.region)
            + "/conversation/cognitiveservices/v1";
        
        // Build query parameters
        let mut query_params = vec![
            ("language".to_string(), language.to_string()),
            ("format".to_string(), "detailed".to_string()),
        ];
        
        if let Some(ref endpoint_id) = self.endpoint_id {
            query_params.push(("cid".to_string(), endpoint_id.clone()));
        }
        
        if !self.profanity_filter {
            query_params.push(("profanity".to_string(), "raw".to_string()));
        }
        
        if self.enable_speaker_identification {
            query_params.push(("diarization".to_string(), "true".to_string()));
            query_params.push(("maxSpeakers".to_string(), self.max_speakers.to_string()));
        }
        
        let response = self.client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .header("Content-Type", "audio/wav")
            .query(&query_params)
            .body(audio_data.to_vec())
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let azure_response: AzureSpeechResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                Ok(azure_response)
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Azure Speech API key or insufficient permissions".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Azure Speech API error: {}", error_text)
                )))
            }
        }
    }
}

#[async_trait]
impl SttProvider for AzureSttProvider {
    fn name(&self) -> &str {
        "azure"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        let language_code = Self::normalize_language(&self.language)?;
        
        let response = self.make_recognize_request(&audio.data, &language_code).await?;
        
        // Parse response
        if response.recognition_status != "Success" {
            return Ok(TranscriptionResult {
                text: String::new(),
                confidence: 0.0,
                language_detected: Some(language_code),
                duration: None,
                words: None,
                metadata: Some(json!({"provider": "azure", "status": response.recognition_status})),
            });
        }
        
        let mut words = Vec::new();
        let mut total_confidence = 0.0;
        let mut confidence_count = 0;
        
        // Extract word-level details
        if let Some(ref nbest) = response.nbest {
            if let Some(first_result) = nbest.first() {
                if let Some(ref word_details) = first_result.words {
                    for word in word_details {
                        words.push(WordTiming {
                            word: word.word.clone(),
                            start: (word.offset.unwrap_or(0.0) / 10_000_000.0) as f32, // Convert from 100ns ticks to seconds
                            end: ((word.offset.unwrap_or(0.0) + word.duration.unwrap_or(0.0)) / 10_000_000.0) as f32,
                            confidence: word.confidence.unwrap_or(0.0),
                        });
                        
                        if let Some(conf) = word.confidence {
                            total_confidence += conf;
                            confidence_count += 1;
                        }
                    }
                }
                
                let avg_confidence = if confidence_count > 0 {
                    total_confidence / confidence_count as f32
                } else {
                    first_result.confidence.unwrap_or(0.0)
                };
                
                let duration = if let Some(last_word) = words.last() {
                    Some(last_word.end)
                } else {
                    None
                };
                
                let mut metadata = serde_json::Map::new();
                metadata.insert("provider".to_string(), json!("azure"));
                metadata.insert("region".to_string(), json!(self.region));
                metadata.insert("recognition_status".to_string(), json!(response.recognition_status));
                
                return Ok(TranscriptionResult {
                    text: first_result.display.clone(),
                    confidence: avg_confidence,
                    language_detected: Some(language_code),
                    duration,
                    words: if words.is_empty() { None } else { Some(words) },
                    metadata: Some(json!(metadata)),
                });
            }
        }
        
        // Fallback to simple display text
        let mut metadata = serde_json::Map::new();
        metadata.insert("provider".to_string(), json!("azure"));
        metadata.insert("region".to_string(), json!(self.region));
        
        Ok(TranscriptionResult {
            text: response.display_text.unwrap_or_default(),
            confidence: 0.0,
            language_detected: Some(language_code),
            duration: None,
            words: None,
            metadata: Some(json!(metadata)),
        })
    }
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let stream = AzureSttStream::new(
            self.api_key.clone(),
            self.region.clone(),
            self.language.clone(),
            config,
            self.profanity_filter,
            self.enable_speaker_identification,
        ).await?;
        Ok(Box::new(stream))
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model {
                id: "default".to_string(),
                name: "Default Speech Model".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "transcription".to_string(),
                    "streaming".to_string(),
                    "word-timestamps".to_string(),
                    "speaker-identification".to_string(),
                    "language-detection".to_string(),
                ],
            },
        ])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(8000),
            AudioFormat::wav(16000),
            AudioFormat::wav(22050),
            AudioFormat::wav(44100),
            AudioFormat::wav(48000),
            AudioFormat::opus(48000),
            AudioFormat::mp3(16000),
            AudioFormat::mp3(44100),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

struct AzureSttStream {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    region: String,
    #[allow(dead_code)]
    language: String,
    session_id: Uuid,
    #[allow(dead_code)]
    profanity_filter: bool,
    #[allow(dead_code)]
    enable_speaker_identification: bool,
    ws_sender: Option<futures::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, Message>>,
    ws_receiver: Option<futures::stream::SplitStream<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>>>,
    #[allow(dead_code)]
    connection_id: String,
    #[allow(dead_code)]
    request_id: String,
}

impl AzureSttStream {
    async fn new(
        api_key: String,
        region: String,
        language: String,
        config: StreamConfig,
        profanity_filter: bool,
        enable_speaker_identification: bool,
    ) -> Result<Self> {
        let connection_id = Uuid::new_v4().to_string();
        let request_id = Uuid::new_v4().to_string();
        
        // Build WebSocket URL
        let ws_url = AZURE_STT_WS_BASE
            .replace("{region}", &region)
            + "/conversation/cognitiveservices/v1";
        
        let mut query_params = vec![
            ("Ocp-Apim-Subscription-Key".to_string(), api_key.clone()),
            ("language".to_string(), language.clone()),
            ("format".to_string(), "detailed".to_string()),
            ("X-ConnectionId".to_string(), connection_id.clone()),
        ];
        
        if !profanity_filter {
            query_params.push(("profanity".to_string(), "raw".to_string()));
        }
        
        if enable_speaker_identification {
            query_params.push(("diarization".to_string(), "true".to_string()));
        }
        
        let url = Url::parse_with_params(&ws_url, &query_params)
            .map_err(|e| DebabelizerError::Provider(ProviderError::ProviderSpecific(
                format!("Invalid WebSocket URL: {}", e)
            )))?;
        
        info!("Connecting to Azure Speech WebSocket: {}", url);
        
        let (ws_stream, _) = connect_async(url.as_str()).await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(
                format!("Failed to connect to Azure Speech WebSocket: {}", e)
            )))?;
        
        let (ws_sender, ws_receiver) = ws_stream.split();
        
        info!("Created Azure Speech streaming session {}", config.session_id);
        
        Ok(Self {
            api_key,
            region,
            language,
            session_id: config.session_id,
            profanity_filter,
            enable_speaker_identification,
            ws_sender: Some(ws_sender),
            ws_receiver: Some(ws_receiver),
            connection_id,
            request_id,
        })
    }
}

#[async_trait]
impl SttStream for AzureSttStream {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()> {
        if let Some(ref mut sender) = self.ws_sender {
            let audio_message = Message::Binary(chunk.to_vec());
            sender.send(audio_message).await
                .map_err(|e| DebabelizerError::Provider(ProviderError::Network(
                    format!("Failed to send audio to Azure WebSocket: {}", e)
                )))?;
        }
        Ok(())
    }
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>> {
        if let Some(ref mut receiver) = self.ws_receiver {
            match receiver.next().await {
                Some(Ok(message)) => {
                    match message {
                        Message::Text(text) => {
                            debug!("Received Azure WebSocket message: {}", text);
                            
                            // Parse Azure WebSocket response
                            if let Ok(response) = serde_json::from_str::<AzureWebSocketResponse>(&text) {
                                if response.path == "speech.hypothesis" || response.path == "speech.phrase" {
                                    if let Some(text_content) = response.text {
                                        let is_final = response.path == "speech.phrase";
                                        
                                        let mut metadata = serde_json::Map::new();
                                        metadata.insert("provider".to_string(), json!("azure"));
                                        metadata.insert("path".to_string(), json!(response.path));
                                        metadata.insert("connection_id".to_string(), json!(self.connection_id));
                                        
                                        return Ok(Some(StreamingResult {
                                            session_id: self.session_id,
                                            is_final,
                                            text: text_content,
                                            confidence: response.confidence.unwrap_or(0.0),
                                            timestamp: Utc::now(),
                                            words: None, // Azure WebSocket doesn't provide word-level details in real-time
                                            metadata: Some(json!(metadata)),
                                        }));
                                    }
                                }
                            }
                        }
                        Message::Binary(_) => {
                            // Azure might send binary responses, but we focus on text for transcription
                            debug!("Received binary message from Azure WebSocket");
                        }
                        Message::Close(_) => {
                            warn!("Azure WebSocket connection closed");
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
                Some(Err(e)) => {
                    error!("Azure WebSocket error: {}", e);
                    return Err(DebabelizerError::Provider(ProviderError::Network(
                        format!("Azure WebSocket error: {}", e)
                    )));
                }
                None => {
                    debug!("Azure WebSocket stream ended");
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }
    
    async fn close(&mut self) -> Result<()> {
        if let Some(mut sender) = self.ws_sender.take() {
            let _ = sender.close().await;
        }
        self.ws_receiver = None;
        Ok(())
    }
    
    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// Azure Speech API Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureSpeechResponse {
    #[serde(rename = "RecognitionStatus")]
    recognition_status: String,
    #[serde(rename = "DisplayText")]
    display_text: Option<String>,
    #[serde(rename = "Offset")]
    offset: Option<u64>,
    #[serde(rename = "Duration")]
    duration: Option<u64>,
    #[serde(rename = "NBest")]
    nbest: Option<Vec<AzureNBestResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureNBestResult {
    #[serde(rename = "Confidence")]
    confidence: Option<f32>,
    #[serde(rename = "Lexical")]
    lexical: String,
    #[serde(rename = "ITN")]
    itn: String,
    #[serde(rename = "MaskedITN")]
    masked_itn: String,
    #[serde(rename = "Display")]
    display: String,
    #[serde(rename = "Words")]
    words: Option<Vec<AzureWordDetail>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureWordDetail {
    #[serde(rename = "Word")]
    word: String,
    #[serde(rename = "Offset")]
    offset: Option<f64>,
    #[serde(rename = "Duration")]
    duration: Option<f64>,
    #[serde(rename = "Confidence")]
    confidence: Option<f32>,
}

// Azure WebSocket Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureWebSocketResponse {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Text")]
    text: Option<String>,
    #[serde(rename = "Confidence")]
    confidence: Option<f32>,
    #[serde(rename = "Properties")]
    properties: Option<serde_json::Value>,
}

// Azure Cognitive Services Text-to-Speech Provider
pub struct AzureTtsProvider {
    client: Client,
    api_key: String,
    region: String,
    default_voice: String,
    speaking_rate: String,
    pitch: String,
    volume: String,
}

impl AzureTtsProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Azure API key not found".to_string()))?;
        
        let region = config
            .get_region()
            .ok_or_else(|| DebabelizerError::Configuration("Azure region not found".to_string()))?;
            
        let default_voice = config
            .get_value("voice")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US-JennyNeural")
            .to_string();
            
        let speaking_rate = config
            .get_value("speaking_rate")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string();
            
        let pitch = config
            .get_value("pitch")
            .and_then(|v| v.as_str())
            .unwrap_or("+0Hz")
            .to_string();
            
        let volume = config
            .get_value("volume")
            .and_then(|v| v.as_str())
            .unwrap_or("100")
            .to_string();
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            region,
            default_voice,
            speaking_rate,
            pitch,
            volume,
        })
    }
    
    fn get_supported_languages() -> Vec<String> {
        vec![
            "ar-SA".to_string(), "bg-BG".to_string(), "ca-ES".to_string(), "cs-CZ".to_string(),
            "da-DK".to_string(), "de-DE".to_string(), "el-GR".to_string(), "en-AU".to_string(),
            "en-CA".to_string(), "en-GB".to_string(), "en-IN".to_string(), "en-NZ".to_string(),
            "en-US".to_string(), "es-ES".to_string(), "es-MX".to_string(), "et-EE".to_string(),
            "fi-FI".to_string(), "fr-CA".to_string(), "fr-FR".to_string(), "gu-IN".to_string(),
            "he-IL".to_string(), "hi-IN".to_string(), "hr-HR".to_string(), "hu-HU".to_string(),
            "it-IT".to_string(), "ja-JP".to_string(), "ko-KR".to_string(), "lt-LT".to_string(),
            "lv-LV".to_string(), "ms-MY".to_string(), "nb-NO".to_string(), "nl-NL".to_string(),
            "pl-PL".to_string(), "pt-BR".to_string(), "pt-PT".to_string(), "ro-RO".to_string(),
            "ru-RU".to_string(), "sk-SK".to_string(), "sl-SI".to_string(), "sv-SE".to_string(),
            "ta-IN".to_string(), "te-IN".to_string(), "th-TH".to_string(), "tr-TR".to_string(),
            "uk-UA".to_string(), "vi-VN".to_string(), "zh-CN".to_string(), "zh-HK".to_string(),
            "zh-TW".to_string(), "af-ZA".to_string(), "am-ET".to_string(), "az-AZ".to_string(),
            "bn-IN".to_string(), "bs-BA".to_string(), "cy-GB".to_string(), "eu-ES".to_string(),
            "fa-IR".to_string(), "ga-IE".to_string(), "gl-ES".to_string(), "id-ID".to_string(),
            "is-IS".to_string(), "jv-ID".to_string(), "ka-GE".to_string(), "kk-KZ".to_string(),
            "km-KH".to_string(), "kn-IN".to_string(), "lo-LA".to_string(), "mk-MK".to_string(),
            "ml-IN".to_string(), "mn-MN".to_string(), "mr-IN".to_string(), "mt-MT".to_string(),
            "my-MM".to_string(), "ne-NP".to_string(), "pa-IN".to_string(), "ps-AF".to_string(),
            "si-LK".to_string(), "so-SO".to_string(), "sq-AL".to_string(), "su-ID".to_string(),
            "sw-KE".to_string(), "sw-TZ".to_string(), "ur-PK".to_string(), "uz-UZ".to_string(),
            "zu-ZA".to_string(),
        ]
    }
    
    fn get_default_voices() -> HashMap<&'static str, &'static str> {
        [
            ("en-US", "en-US-JennyNeural"),
            ("en-GB", "en-GB-SoniaNeural"),
            ("es-ES", "es-ES-ElviraNeural"),
            ("es-MX", "es-MX-DaliaNeural"),
            ("fr-FR", "fr-FR-DeniseNeural"),
            ("de-DE", "de-DE-KatjaNeural"),
            ("it-IT", "it-IT-ElsaNeural"),
            ("pt-BR", "pt-BR-FranciscaNeural"),
            ("ru-RU", "ru-RU-SvetlanaNeural"),
            ("ja-JP", "ja-JP-NanamiNeural"),
            ("ko-KR", "ko-KR-SunHiNeural"),
            ("zh-CN", "zh-CN-XiaoxiaoNeural"),
            ("ar-SA", "ar-SA-ZariyahNeural"),
            ("hi-IN", "hi-IN-SwaraNeural"),
            ("nl-NL", "nl-NL-ColetteNeural"),
            ("pl-PL", "pl-PL-ZofiaNeural"),
            ("tr-TR", "tr-TR-EmelNeural"),
            ("sv-SE", "sv-SE-SofieNeural"),
            ("da-DK", "da-DK-ChristelNeural"),
            ("nb-NO", "nb-NO-PernilleNeural"),
            ("fi-FI", "fi-FI-NooraNeural"),
        ].iter().cloned().collect()
    }
    
    fn normalize_language(language: &str) -> String {
        // Language mapping for short codes
        let language_map: HashMap<&str, &str> = [
            ("en", "en-US"), ("es", "es-ES"), ("fr", "fr-FR"), ("de", "de-DE"),
            ("it", "it-IT"), ("pt", "pt-BR"), ("ru", "ru-RU"), ("ja", "ja-JP"),
            ("ko", "ko-KR"), ("zh", "zh-CN"), ("ar", "ar-SA"), ("hi", "hi-IN"),
            ("nl", "nl-NL"), ("pl", "pl-PL"), ("tr", "tr-TR"), ("sv", "sv-SE"),
            ("da", "da-DK"), ("no", "nb-NO"), ("fi", "fi-FI"), ("cs", "cs-CZ"),
            ("hu", "hu-HU"), ("el", "el-GR"), ("he", "he-IL"), ("th", "th-TH"),
            ("vi", "vi-VN"), ("id", "id-ID"), ("ms", "ms-MY"), ("ro", "ro-RO"),
            ("uk", "uk-UA"), ("bg", "bg-BG"), ("hr", "hr-HR"), ("sk", "sk-SK"),
            ("sl", "sl-SI"), ("et", "et-EE"), ("lv", "lv-LV"), ("lt", "lt-LT"),
            ("ca", "ca-ES"), ("eu", "eu-ES"), ("gl", "gl-ES"), ("af", "af-ZA"),
            ("sq", "sq-AL"), ("am", "am-ET"), ("hy", "hy-AM"), ("az", "az-AZ"),
            ("bn", "bn-IN"), ("bs", "bs-BA"), ("my", "my-MM"), ("cy", "cy-GB"),
            ("gu", "gu-IN"), ("is", "is-IS"), ("jv", "jv-ID"), ("kn", "kn-IN"),
            ("km", "km-KH"), ("lo", "lo-LA"), ("mk", "mk-MK"), ("ml", "ml-IN"),
            ("mn", "mn-MN"), ("mr", "mr-IN"), ("ne", "ne-NP"), ("ps", "ps-AF"),
            ("si", "si-LK"), ("su", "su-ID"), ("sw", "sw-KE"), ("ta", "ta-IN"),
            ("te", "te-IN"), ("ur", "ur-PK"), ("uz", "uz-UZ"), ("zu", "zu-ZA"),
        ].iter().cloned().collect();
        
        let short_code = language.split("-").next().unwrap_or(language).to_lowercase();
        if let Some(mapped) = language_map.get(short_code.as_str()) {
            return mapped.to_string();
        }
        
        // Direct match or fallback to en-US
        if Self::get_supported_languages().contains(&language.to_string()) {
            language.to_string()
        } else {
            "en-US".to_string()
        }
    }
    
    fn get_voice_for_language(&self, language: &str) -> String {
        let default_voices = Self::get_default_voices();
        let normalized_lang = Self::normalize_language(language);
        
        if let Some(voice) = default_voices.get(normalized_lang.as_str()) {
            voice.to_string()
        } else {
            self.default_voice.clone()
        }
    }
    
    fn map_audio_format_to_azure(&self, format: &AudioFormat) -> String {
        match format.format.as_str() {
            "wav" => match format.sample_rate {
                8000 => "riff-8khz-16bit-mono-pcm".to_string(),
                16000 => "riff-16khz-16bit-mono-pcm".to_string(),
                22050 => "riff-22050hz-16bit-mono-pcm".to_string(),
                24000 => "riff-24khz-16bit-mono-pcm".to_string(),
                48000 => "riff-48khz-16bit-mono-pcm".to_string(),
                _ => "riff-16khz-16bit-mono-pcm".to_string(),
            },
            "mp3" => match format.sample_rate {
                16000 => "audio-16khz-32kbitrate-mono-mp3".to_string(),
                24000 => "audio-24khz-48kbitrate-mono-mp3".to_string(),
                48000 => "audio-48khz-96kbitrate-mono-mp3".to_string(),
                _ => "audio-16khz-32kbitrate-mono-mp3".to_string(),
            },
            "opus" => "ogg-16khz-16bit-mono-opus".to_string(),
            "ogg" => "ogg-16khz-16bit-mono-opus".to_string(),
            "webm" => "webm-16khz-16bit-mono-opus".to_string(),
            _ => "riff-16khz-16bit-mono-pcm".to_string(), // Default to WAV
        }
    }
    
    fn build_ssml(&self, text: &str, options: &SynthesisOptions) -> String {
        let voice_name = if options.voice.voice_id.contains("Neural") {
            options.voice.voice_id.clone()
        } else {
            self.get_voice_for_language(&options.voice.language)
        };
        
        let speaking_rate = options.speed.map(|s| s.to_string()).unwrap_or_else(|| self.speaking_rate.clone());
        let pitch = options.pitch.map(|p| format!("{}Hz", p)).unwrap_or_else(|| self.pitch.clone());
        let volume = options.volume_gain_db.map(|v| format!("{}%", (v * 10.0 + 100.0).max(0.0).min(200.0))).unwrap_or_else(|| self.volume.clone());
        
        format!(
            r#"<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{}">
                <voice name="{}">
                    <prosody rate="{}" pitch="{}" volume="{}">
                        {}
                    </prosody>
                </voice>
            </speak>"#,
            options.voice.language,
            voice_name,
            speaking_rate,
            pitch,
            volume,
            text
        )
    }
    
    async fn make_synthesis_request(&self, text: &str, options: &SynthesisOptions) -> Result<Vec<u8>> {
        let url = AZURE_TTS_API_BASE.replace("{region}", &self.region);
        let ssml = self.build_ssml(text, options);
        let output_format = self.map_audio_format_to_azure(&options.format);
        
        let response = self.client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .header("Content-Type", "application/ssml+xml")
            .header("X-Microsoft-OutputFormat", output_format)
            .header("User-Agent", "debabelizer-rust")
            .body(ssml)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let audio_data = response.bytes().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                Ok(audio_data.to_vec())
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Azure Speech API key or insufficient permissions".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Azure TTS API error: {}", error_text)
                )))
            }
        }
    }
    
    async fn fetch_available_voices(&self) -> Result<Vec<Voice>> {
        let url = format!("https://{}.tts.speech.microsoft.com/cognitiveservices/voices/list", self.region);
        
        let response = self.client
            .get(&url)
            .header("Ocp-Apim-Subscription-Key", &self.api_key)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let azure_voices: Vec<AzureTtsVoice> = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                
                let voices = azure_voices.into_iter().map(|voice| {
                    Voice {
                        voice_id: voice.short_name.clone(),
                        name: voice.display_name.clone(),
                        description: Some(voice.local_name.clone()),
                        language: voice.locale.clone(),
                        gender: Some(voice.gender.clone()),
                        age: None,
                        accent: None,
                        style: None,
                        use_case: Some(voice.voice_type.clone()),
                        preview_url: None,
                        metadata: Some(json!({
                            "sampleRateHertz": voice.sample_rate_hertz,
                            "voiceType": voice.voice_type,
                            "status": voice.status,
                            "wordsPerMinute": voice.words_per_minute,
                            "styleDegree": voice.style_list.map(|s| s.len()).unwrap_or(0)
                        })),
                    }
                }).collect();
                
                Ok(voices)
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Azure Speech API key or insufficient permissions".to_string()
                )))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Azure TTS voices API error: {}", error_text)
                )))
            }
        }
    }
}

#[async_trait]
impl TtsProvider for AzureTtsProvider {
    fn name(&self) -> &str {
        "azure"
    }
    
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult> {
        let audio_data = self.make_synthesis_request(text, options).await?;
        let size_bytes = audio_data.len();
        
        let mut metadata = serde_json::Map::new();
        metadata.insert("provider".to_string(), json!("azure"));
        metadata.insert("region".to_string(), json!(self.region));
        metadata.insert("voice_name".to_string(), json!(options.voice.voice_id));
        metadata.insert("speaking_rate".to_string(), json!(self.speaking_rate));
        metadata.insert("pitch".to_string(), json!(self.pitch));
        metadata.insert("volume".to_string(), json!(self.volume));
        
        Ok(SynthesisResult {
            audio_data,
            format: options.format.clone(),
            duration: None, // Azure doesn't provide duration in response
            size_bytes,
            metadata: Some(json!(metadata)),
        })
    }
    
    async fn synthesize_stream(&self, text: &str, options: &SynthesisOptions) -> Result<Box<dyn TtsStream>> {
        // Azure TTS doesn't support real-time streaming, so we'll simulate it
        let result = self.synthesize(text, options).await?;
        let stream = AzureTtsStream::new(result.audio_data);
        Ok(Box::new(stream))
    }
    
    async fn list_voices(&self) -> Result<Vec<Voice>> {
        self.fetch_available_voices().await
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(8000),
            AudioFormat::wav(16000),
            AudioFormat::wav(22050),
            AudioFormat::wav(24000),
            AudioFormat::wav(48000),
            AudioFormat::mp3(16000),
            AudioFormat::mp3(24000),
            AudioFormat::mp3(48000),
            AudioFormat::opus(16000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        false // Azure TTS doesn't support real-time streaming
    }
    
    fn supports_ssml(&self) -> bool {
        true
    }
}

// Simulated streaming for Azure TTS (since it doesn't support real-time streaming)
struct AzureTtsStream {
    audio_data: Vec<u8>,
    position: usize,
    chunk_size: usize,
}

impl AzureTtsStream {
    fn new(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data,
            position: 0,
            chunk_size: 4096, // 4KB chunks
        }
    }
}

#[async_trait]
impl TtsStream for AzureTtsStream {
    async fn receive_chunk(&mut self) -> Result<Option<Bytes>> {
        if self.position >= self.audio_data.len() {
            return Ok(None);
        }
        
        let end = std::cmp::min(self.position + self.chunk_size, self.audio_data.len());
        let chunk = self.audio_data[self.position..end].to_vec();
        self.position = end;
        
        Ok(Some(Bytes::from(chunk)))
    }
    
    async fn close(&mut self) -> Result<()> {
        self.position = self.audio_data.len();
        Ok(())
    }
}

// Azure TTS API Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AzureTtsVoice {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "DisplayName")]
    display_name: String,
    #[serde(rename = "LocalName")]
    local_name: String,
    #[serde(rename = "ShortName")]
    short_name: String,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "Locale")]
    locale: String,
    #[serde(rename = "StyleList")]
    style_list: Option<Vec<String>>,
    #[serde(rename = "SampleRateHertz")]
    sample_rate_hertz: String,
    #[serde(rename = "VoiceType")]
    voice_type: String,
    #[serde(rename = "Status")]
    status: String,
    #[serde(rename = "WordsPerMinute")]
    words_per_minute: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    
    fn create_test_config() -> ProviderConfig {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key"));
        config.insert("region".to_string(), json!("eastus"));
        config.insert("language".to_string(), json!("en-US"));
        ProviderConfig::Simple(config)
    }
    
    #[test]
    fn test_provider_config_extraction() {
        let config = create_test_config();
        
        assert_eq!(config.get_api_key(), Some("test-api-key".to_string()));
        assert_eq!(config.get_region(), Some("eastus".to_string()));
        assert_eq!(config.get_value("language").unwrap().as_str(), Some("en-US"));
    }
    
    #[test]
    fn test_supported_languages() {
        let languages = AzureSttProvider::get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"es-ES".to_string()));
        assert!(languages.contains(&"fr-FR".to_string()));
        assert!(languages.contains(&"de-DE".to_string()));
        assert!(languages.contains(&"zh-CN".to_string()));
        assert!(languages.contains(&"ja-JP".to_string()));
        assert!(languages.contains(&"ar-SA".to_string()));
    }
    
    #[test]
    fn test_language_normalization() {
        // Test direct matches
        assert_eq!(AzureSttProvider::normalize_language("en-US").unwrap(), "en-US");
        assert_eq!(AzureSttProvider::normalize_language("es-ES").unwrap(), "es-ES");
        
        // Test short code mapping
        assert_eq!(AzureSttProvider::normalize_language("en").unwrap(), "en-US");
        assert_eq!(AzureSttProvider::normalize_language("fr").unwrap(), "fr-FR");
        assert_eq!(AzureSttProvider::normalize_language("de").unwrap(), "de-DE");
        assert_eq!(AzureSttProvider::normalize_language("ar").unwrap(), "ar-SA");
        
        // Test case insensitive
        assert_eq!(AzureSttProvider::normalize_language("EN-US").unwrap(), "en-US");
        
        // Test unsupported language
        assert!(AzureSttProvider::normalize_language("xyz").is_err());
    }
    
    #[test]
    fn test_azure_speech_response_parsing() {
        let response_json = r#"
        {
            "RecognitionStatus": "Success",
            "DisplayText": "Hello, how are you today?",
            "Offset": 0,
            "Duration": 25000000,
            "NBest": [
                {
                    "Confidence": 0.95,
                    "Lexical": "hello how are you today",
                    "ITN": "hello how are you today",
                    "MaskedITN": "hello how are you today",
                    "Display": "Hello, how are you today?",
                    "Words": [
                        {
                            "Word": "Hello",
                            "Offset": 0.0,
                            "Duration": 5000000.0,
                            "Confidence": 0.98
                        },
                        {
                            "Word": "how",
                            "Offset": 6000000.0,
                            "Duration": 3000000.0,
                            "Confidence": 0.96
                        }
                    ]
                }
            ]
        }
        "#;
        
        let response: AzureSpeechResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.recognition_status, "Success");
        assert_eq!(response.display_text, Some("Hello, how are you today?".to_string()));
        assert_eq!(response.offset, Some(0));
        assert_eq!(response.duration, Some(25000000));
        
        let nbest = response.nbest.unwrap();
        assert_eq!(nbest.len(), 1);
        
        let result = &nbest[0];
        assert_eq!(result.confidence, Some(0.95));
        assert_eq!(result.display, "Hello, how are you today?");
        
        let words = result.words.as_ref().unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[0].confidence, Some(0.98));
        assert_eq!(words[1].word, "how");
        assert_eq!(words[1].confidence, Some(0.96));
    }
    
    #[test]
    fn test_websocket_response_parsing() {
        let response_json = r#"
        {
            "Path": "speech.hypothesis",
            "Text": "Testing streaming response",
            "Confidence": 0.92,
            "Properties": {
                "RequestId": "12345"
            }
        }
        "#;
        
        let response: AzureWebSocketResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.path, "speech.hypothesis");
        assert_eq!(response.text, Some("Testing streaming response".to_string()));
        assert_eq!(response.confidence, Some(0.92));
        assert!(response.properties.is_some());
    }
    
    #[test]
    fn test_final_websocket_response_parsing() {
        let response_json = r#"
        {
            "Path": "speech.phrase",
            "Text": "Final transcription result",
            "Confidence": 0.98
        }
        "#;
        
        let response: AzureWebSocketResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.path, "speech.phrase");
        assert_eq!(response.text, Some("Final transcription result".to_string()));
        assert_eq!(response.confidence, Some(0.98));
    }
    
    #[test]
    fn test_word_timing_conversion() {
        // Test conversion from Azure's 100ns ticks to seconds
        let offset_ticks = 10_000_000.0; // 1 second in 100ns ticks
        let duration_ticks = 5_000_000.0; // 0.5 seconds in 100ns ticks
        
        let start_seconds = offset_ticks / 10_000_000.0;
        let end_seconds = (offset_ticks + duration_ticks) / 10_000_000.0;
        
        assert_eq!(start_seconds, 1.0);
        assert_eq!(end_seconds, 1.5);
    }
    
    #[test]
    fn test_url_construction() {
        let region = "eastus";
        let base_url = AZURE_STT_REST_API_BASE.replace("{region}", region);
        let expected = "https://eastus.stt.speech.microsoft.com/speech/recognition";
        assert_eq!(base_url, expected);
        
        let ws_base_url = AZURE_STT_WS_BASE.replace("{region}", region);
        let expected_ws = "wss://eastus.stt.speech.microsoft.com/speech/recognition";
        assert_eq!(ws_base_url, expected_ws);
        
        let tts_base_url = AZURE_TTS_API_BASE.replace("{region}", region);
        let expected_tts = "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1";
        assert_eq!(tts_base_url, expected_tts);
    }
    
    // Azure TTS Provider Tests
    
    #[allow(dead_code)]
    fn create_test_tts_config() -> ProviderConfig {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key"));
        config.insert("region".to_string(), json!("eastus"));
        config.insert("voice".to_string(), json!("en-US-JennyNeural"));
        config.insert("speaking_rate".to_string(), json!("1.0"));
        config.insert("pitch".to_string(), json!("+0Hz"));
        config.insert("volume".to_string(), json!("100"));
        ProviderConfig::Simple(config)
    }
    
    #[test]
    fn test_tts_supported_languages() {
        let languages = AzureTtsProvider::get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"es-ES".to_string()));
        assert!(languages.contains(&"fr-FR".to_string()));
        assert!(languages.contains(&"de-DE".to_string()));
        assert!(languages.contains(&"zh-CN".to_string()));
        assert!(languages.contains(&"ja-JP".to_string()));
        assert!(languages.contains(&"ar-SA".to_string()));
    }
    
    #[test]
    fn test_tts_language_normalization() {
        // Test direct matches
        assert_eq!(AzureTtsProvider::normalize_language("en-US"), "en-US");
        assert_eq!(AzureTtsProvider::normalize_language("es-ES"), "es-ES");
        
        // Test short code mapping
        assert_eq!(AzureTtsProvider::normalize_language("en"), "en-US");
        assert_eq!(AzureTtsProvider::normalize_language("fr"), "fr-FR");
        assert_eq!(AzureTtsProvider::normalize_language("de"), "de-DE");
        assert_eq!(AzureTtsProvider::normalize_language("ar"), "ar-SA");
        
        // Test unsupported language fallback
        assert_eq!(AzureTtsProvider::normalize_language("xyz"), "en-US");
    }
    
    #[test]
    fn test_default_voices() {
        let voices = AzureTtsProvider::get_default_voices();
        assert!(!voices.is_empty());
        assert_eq!(voices.get("en-US"), Some(&"en-US-JennyNeural"));
        assert_eq!(voices.get("es-ES"), Some(&"es-ES-ElviraNeural"));
        assert_eq!(voices.get("fr-FR"), Some(&"fr-FR-DeniseNeural"));
        assert_eq!(voices.get("de-DE"), Some(&"de-DE-KatjaNeural"));
        assert_eq!(voices.get("ja-JP"), Some(&"ja-JP-NanamiNeural"));
    }
    
    #[test]
    fn test_audio_format_mapping() {
        let provider = AzureTtsProvider {
            client: reqwest::Client::new(),
            api_key: "test".to_string(),
            region: "eastus".to_string(),
            default_voice: "en-US-JennyNeural".to_string(),
            speaking_rate: "1.0".to_string(),
            pitch: "+0Hz".to_string(),
            volume: "100".to_string(),
        };
        
        // Test WAV formats
        let wav_16k = AudioFormat::wav(16000);
        assert_eq!(provider.map_audio_format_to_azure(&wav_16k), "riff-16khz-16bit-mono-pcm");
        
        let wav_24k = AudioFormat::wav(24000);
        assert_eq!(provider.map_audio_format_to_azure(&wav_24k), "riff-24khz-16bit-mono-pcm");
        
        // Test MP3 formats
        let mp3_16k = AudioFormat::mp3(16000);
        assert_eq!(provider.map_audio_format_to_azure(&mp3_16k), "audio-16khz-32kbitrate-mono-mp3");
        
        let mp3_24k = AudioFormat::mp3(24000);
        assert_eq!(provider.map_audio_format_to_azure(&mp3_24k), "audio-24khz-48kbitrate-mono-mp3");
        
        // Test OPUS
        let opus = AudioFormat::opus(16000);
        assert_eq!(provider.map_audio_format_to_azure(&opus), "ogg-16khz-16bit-mono-opus");
    }
    
    #[test]
    fn test_ssml_generation() {
        let provider = AzureTtsProvider {
            client: reqwest::Client::new(),
            api_key: "test".to_string(),
            region: "eastus".to_string(),
            default_voice: "en-US-JennyNeural".to_string(),
            speaking_rate: "1.0".to_string(),
            pitch: "+0Hz".to_string(),
            volume: "100".to_string(),
        };
        
        let voice = Voice {
            voice_id: "en-US-JennyNeural".to_string(),
            name: "Jenny".to_string(),
            description: None,
            language: "en-US".to_string(),
            gender: Some("Female".to_string()),
            age: None,
            accent: None,
            style: None,
            use_case: None,
            preview_url: None,
            metadata: None,
        };
        
        let options = SynthesisOptions {
            voice,
            model: None,
            speed: Some(1.2),
            pitch: Some(5.0),
            volume_gain_db: Some(2.0),
            format: AudioFormat::wav(16000),
            sample_rate: None,
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        };
        
        let ssml = provider.build_ssml("Hello world", &options);
        
        assert!(ssml.contains("<speak"));
        assert!(ssml.contains("en-US-JennyNeural"));
        assert!(ssml.contains("Hello world"));
        assert!(ssml.contains("xml:lang=\"en-US\""));
        assert!(ssml.contains("<prosody"));
        assert!(ssml.contains("rate=\"1.2\""));
        assert!(ssml.contains("pitch=\"5Hz\""));
    }
    
    #[test]
    fn test_azure_tts_voice_parsing() {
        let voice_json = r#"
        {
            "Name": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
            "DisplayName": "Jenny",
            "LocalName": "Jenny",
            "ShortName": "en-US-JennyNeural",
            "Gender": "Female",
            "Locale": "en-US",
            "StyleList": ["cheerful", "sad", "angry"],
            "SampleRateHertz": "24000",
            "VoiceType": "Neural",
            "Status": "GA",
            "WordsPerMinute": "180"
        }
        "#;
        
        let voice: AzureTtsVoice = serde_json::from_str(voice_json).unwrap();
        
        assert_eq!(voice.short_name, "en-US-JennyNeural");
        assert_eq!(voice.display_name, "Jenny");
        assert_eq!(voice.gender, "Female");
        assert_eq!(voice.locale, "en-US");
        assert_eq!(voice.voice_type, "Neural");
        assert_eq!(voice.status, "GA");
        assert_eq!(voice.sample_rate_hertz, "24000");
        assert_eq!(voice.words_per_minute, "180");
        assert_eq!(voice.style_list, Some(vec!["cheerful".to_string(), "sad".to_string(), "angry".to_string()]));
    }
    
    #[test]
    fn test_tts_stream_chunks() {
        let audio_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut stream = AzureTtsStream::new(audio_data.clone());
        
        // Test chunk size calculation
        let expected_chunks = (audio_data.len() as f32 / stream.chunk_size as f32).ceil() as usize;
        assert!(expected_chunks >= 1);
        
        // Test position tracking
        assert_eq!(stream.position, 0);
        stream.position = 5;
        assert_eq!(stream.position, 5);
    }
    
    #[test]
    fn test_voice_for_language() {
        let provider = AzureTtsProvider {
            client: reqwest::Client::new(),
            api_key: "test".to_string(),
            region: "eastus".to_string(),
            default_voice: "en-US-JennyNeural".to_string(),
            speaking_rate: "1.0".to_string(),
            pitch: "+0Hz".to_string(),
            volume: "100".to_string(),
        };
        
        // Test known languages
        assert_eq!(provider.get_voice_for_language("en-US"), "en-US-JennyNeural");
        assert_eq!(provider.get_voice_for_language("es-ES"), "es-ES-ElviraNeural");
        assert_eq!(provider.get_voice_for_language("fr-FR"), "fr-FR-DeniseNeural");
        
        // Test fallback for unknown language
        assert_eq!(provider.get_voice_for_language("xyz"), "en-US-JennyNeural");
        
        // Test short code mapping
        assert_eq!(provider.get_voice_for_language("en"), "en-US-JennyNeural");
        assert_eq!(provider.get_voice_for_language("es"), "es-ES-ElviraNeural");
    }
}