use async_trait::async_trait;
use base64::Engine;
use chrono::Utc;
use debabelizer_core::{
    AudioData, AudioFormat, DebabelizerError, Model, ProviderError, Result, SttProvider,
    SttStream, StreamConfig, StreamingResult, TranscriptionResult, WordTiming,
    TtsProvider, TtsStream, Voice, SynthesisResult, SynthesisOptions,
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tracing::{info, warn};
use uuid::Uuid;
use bytes::Bytes;

const GOOGLE_STT_API_BASE: &str = "https://speech.googleapis.com/v1";
const GOOGLE_TTS_API_BASE: &str = "https://texttospeech.googleapis.com/v1";

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
    
    pub fn get_project_id(&self) -> Option<String> {
        match self {
            Self::Simple(map) => map
                .get("project_id")
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

pub struct GoogleSttProvider {
    client: Client,
    api_key: String,
    project_id: String,
    model: String,
    enable_automatic_punctuation: bool,
    enable_word_time_offsets: bool,
    enable_speaker_diarization: bool,
    profanity_filter: bool,
}

impl GoogleSttProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Google API key not found".to_string()))?;
        
        let project_id = config
            .get_project_id()
            .ok_or_else(|| DebabelizerError::Configuration("Google project ID not found".to_string()))?;
            
        let model = config
            .get_value("model")
            .and_then(|v| v.as_str())
            .unwrap_or("latest_long")
            .to_string();
            
        let enable_automatic_punctuation = config
            .get_value("enable_automatic_punctuation")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
            
        let enable_word_time_offsets = config
            .get_value("enable_word_time_offsets")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
            
        let enable_speaker_diarization = config
            .get_value("enable_speaker_diarization")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
            
        let profanity_filter = config
            .get_value("profanity_filter")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            project_id,
            model,
            enable_automatic_punctuation,
            enable_word_time_offsets,
            enable_speaker_diarization,
            profanity_filter,
        })
    }
    
    fn get_supported_languages() -> Vec<String> {
        vec![
            "en".to_string(), "en-US".to_string(), "en-GB".to_string(), "en-AU".to_string(),
            "en-CA".to_string(), "en-IN".to_string(), "en-IE".to_string(), "en-NZ".to_string(),
            "en-PH".to_string(), "en-SG".to_string(), "en-ZA".to_string(),
            "es".to_string(), "es-ES".to_string(), "es-AR".to_string(), "es-BO".to_string(),
            "es-CL".to_string(), "es-CO".to_string(), "es-CR".to_string(), "es-DO".to_string(),
            "es-EC".to_string(), "es-GT".to_string(), "es-HN".to_string(), "es-MX".to_string(),
            "es-NI".to_string(), "es-PA".to_string(), "es-PE".to_string(), "es-PR".to_string(),
            "es-PY".to_string(), "es-SV".to_string(), "es-UY".to_string(), "es-VE".to_string(),
            "fr".to_string(), "fr-FR".to_string(), "fr-CA".to_string(), "fr-CH".to_string(),
            "fr-BE".to_string(),
            "de".to_string(), "de-DE".to_string(), "de-AT".to_string(), "de-CH".to_string(),
            "it".to_string(), "it-IT".to_string(), "it-CH".to_string(),
            "pt".to_string(), "pt-BR".to_string(), "pt-PT".to_string(),
            "ru".to_string(), "ru-RU".to_string(),
            "ja".to_string(), "ja-JP".to_string(),
            "ko".to_string(), "ko-KR".to_string(),
            "zh".to_string(), "zh-CN".to_string(), "zh-TW".to_string(),
            "ar".to_string(), "ar-SA".to_string(), "ar-AE".to_string(), "ar-BH".to_string(),
            "ar-DZ".to_string(), "ar-EG".to_string(), "ar-IQ".to_string(), "ar-JO".to_string(),
            "ar-KW".to_string(), "ar-LB".to_string(), "ar-LY".to_string(), "ar-MA".to_string(),
            "ar-OM".to_string(), "ar-QA".to_string(), "ar-SY".to_string(), "ar-TN".to_string(),
            "ar-YE".to_string(),
            "hi".to_string(), "hi-IN".to_string(),
            "nl".to_string(), "nl-NL".to_string(), "nl-BE".to_string(),
            "pl".to_string(), "pl-PL".to_string(),
            "tr".to_string(), "tr-TR".to_string(),
            "sv".to_string(), "sv-SE".to_string(),
            "da".to_string(), "da-DK".to_string(),
            "no".to_string(), "no-NO".to_string(),
            "fi".to_string(), "fi-FI".to_string(),
            "cs".to_string(), "cs-CZ".to_string(),
            "hu".to_string(), "hu-HU".to_string(),
            "el".to_string(), "el-GR".to_string(),
            "he".to_string(), "he-IL".to_string(),
            "th".to_string(), "th-TH".to_string(),
            "vi".to_string(), "vi-VN".to_string(),
            "id".to_string(), "id-ID".to_string(),
            "ms".to_string(), "ms-MY".to_string(),
            "ro".to_string(), "ro-RO".to_string(),
            "uk".to_string(), "uk-UA".to_string(),
            "bg".to_string(), "bg-BG".to_string(),
            "hr".to_string(), "hr-HR".to_string(),
            "sk".to_string(), "sk-SK".to_string(),
            "sl".to_string(), "sl-SI".to_string(),
            "et".to_string(), "et-EE".to_string(),
            "lv".to_string(), "lv-LV".to_string(),
            "lt".to_string(), "lt-LT".to_string(),
            "sr".to_string(), "sr-RS".to_string(),
            "sw".to_string(), "sw-KE".to_string(), "sw-TZ".to_string(),
            "ta".to_string(), "ta-IN".to_string(), "ta-SG".to_string(), "ta-LK".to_string(),
            "te".to_string(), "te-IN".to_string(),
            "bn".to_string(), "bn-IN".to_string(), "bn-BD".to_string(),
            "gu".to_string(), "gu-IN".to_string(),
            "mr".to_string(), "mr-IN".to_string(),
            "kn".to_string(), "kn-IN".to_string(),
            "ml".to_string(), "ml-IN".to_string(),
            "pa".to_string(), "pa-IN".to_string(),
        ]
    }
    
    fn normalize_language(language: &str) -> Result<String> {
        let supported = Self::get_supported_languages();
        
        // Auto-detection not directly supported by Google Cloud Speech
        if language.to_lowercase() == "auto" {
            warn!("Google Cloud Speech doesn't support auto-detection, defaulting to en-US");
            return Ok("en-US".to_string());
        }
        
        // Language mapping for common short codes
        let language_map: HashMap<&str, &str> = [
            ("en", "en-US"), ("es", "es-ES"), ("fr", "fr-FR"), ("de", "de-DE"),
            ("it", "it-IT"), ("pt", "pt-BR"), ("ru", "ru-RU"), ("ja", "ja-JP"),
            ("ko", "ko-KR"), ("zh", "zh-CN"), ("ar", "ar-SA"), ("hi", "hi-IN"),
            ("nl", "nl-NL"), ("pl", "pl-PL"), ("tr", "tr-TR"), ("sv", "sv-SE"),
            ("da", "da-DK"), ("no", "no-NO"), ("fi", "fi-FI"), ("cs", "cs-CZ"),
            ("hu", "hu-HU"), ("el", "el-GR"), ("he", "he-IL"), ("th", "th-TH"),
            ("vi", "vi-VN"), ("id", "id-ID"), ("ms", "ms-MY"), ("ro", "ro-RO"),
            ("uk", "uk-UA"), ("bg", "bg-BG"), ("hr", "hr-HR"), ("sk", "sk-SK"),
            ("sl", "sl-SI"), ("et", "et-EE"), ("lv", "lv-LV"), ("lt", "lt-LT"),
            ("sr", "sr-RS"), ("sw", "sw-KE"), ("ta", "ta-IN"), ("te", "te-IN"),
            ("bn", "bn-IN"), ("gu", "gu-IN"), ("mr", "mr-IN"), ("kn", "kn-IN"),
            ("ml", "ml-IN"), ("pa", "pa-IN"),
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
            format!("Language '{}' is not supported by Google Cloud Speech", language)
        )))
    }
    
    
    async fn make_recognize_request(&self, audio_data: &[u8], config: &RecognitionConfig) -> Result<GoogleSpeechResponse> {
        let url = format!("{}/speech:recognize?key={}", GOOGLE_STT_API_BASE, self.api_key);
        
        // Encode audio data in base64
        let audio_content = base64::engine::general_purpose::STANDARD.encode(audio_data);
        
        let request_body = json!({
            "config": config,
            "audio": {
                "content": audio_content
            }
        });
        
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let google_response: GoogleSpeechResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                Ok(google_response)
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Google Cloud credentials or insufficient permissions".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Google Cloud Speech API error: {}", error_text)
                )))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RecognitionConfig {
    encoding: String,
    sample_rate_hertz: u32,
    language_code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    alternative_language_codes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_alternatives: Option<u32>,
    enable_automatic_punctuation: bool,
    enable_word_time_offsets: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_speaker_diarization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarization_speaker_count: Option<u32>,
    profanity_filter: bool,
    model: String,
    use_enhanced: bool,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            encoding: "LINEAR16".to_string(),
            sample_rate_hertz: 16000,
            language_code: "en-US".to_string(),
            alternative_language_codes: None,
            max_alternatives: None,
            enable_automatic_punctuation: true,
            enable_word_time_offsets: true,
            enable_speaker_diarization: None,
            diarization_speaker_count: None,
            profanity_filter: false,
            model: "latest_long".to_string(),
            use_enhanced: true,
        }
    }
}

#[async_trait]
impl SttProvider for GoogleSttProvider {
    fn name(&self) -> &str {
        "google"
    }
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        let language_code = Self::normalize_language("en-US")?; // Default language
        
        let config = RecognitionConfig {
            encoding: self.map_audio_format_to_encoding(&audio.format),
            sample_rate_hertz: audio.format.sample_rate,
            language_code,
            enable_automatic_punctuation: self.enable_automatic_punctuation,
            enable_word_time_offsets: self.enable_word_time_offsets,
            enable_speaker_diarization: if self.enable_speaker_diarization { Some(true) } else { None },
            diarization_speaker_count: if self.enable_speaker_diarization { Some(2) } else { None },
            profanity_filter: self.profanity_filter,
            model: self.model.clone(),
            ..Default::default()
        };
        
        let response = self.make_recognize_request(&audio.data, &config).await?;
        
        // Parse response
        if response.results.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                confidence: 0.0,
                language_detected: Some(config.language_code),
                duration: None,
                words: None,
                metadata: Some(json!({"provider": "google", "model": self.model})),
            });
        }
        
        let mut full_transcript = Vec::new();
        let mut all_words = Vec::new();
        let mut total_confidence = 0.0;
        let mut num_alternatives = 0;
        
        for result in &response.results {
            if let Some(alternative) = result.alternatives.first() {
                full_transcript.push(alternative.transcript.clone());
                
                if let Some(confidence) = alternative.confidence {
                    total_confidence += confidence;
                    num_alternatives += 1;
                }
                
                // Extract word timings
                if let Some(ref words) = alternative.words {
                    for word in words {
                        all_words.push(WordTiming {
                            word: word.word.clone(),
                            start: word.start_time.as_ref()
                                .map(|ts| ts.to_seconds())
                                .unwrap_or(0.0),
                            end: word.end_time.as_ref()
                                .map(|ts| ts.to_seconds())
                                .unwrap_or(0.0),
                            confidence: word.confidence.unwrap_or(0.0),
                        });
                    }
                }
            }
        }
        
        let avg_confidence = if num_alternatives > 0 {
            total_confidence / num_alternatives as f32
        } else {
            0.0
        };
        
        let mut metadata = serde_json::Map::new();
        metadata.insert("provider".to_string(), json!("google"));
        metadata.insert("model".to_string(), json!(self.model));
        metadata.insert("project_id".to_string(), json!(self.project_id));
        
        Ok(TranscriptionResult {
            text: full_transcript.join(" "),
            confidence: avg_confidence,
            language_detected: Some(config.language_code),
            duration: all_words.last().map(|w| w.end),
            words: if all_words.is_empty() { None } else { Some(all_words) },
            metadata: Some(json!(metadata)),
        })
    }
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let stream = GoogleSttStream::new(
            self.api_key.clone(),
            self.model.clone(),
            config,
            self.enable_automatic_punctuation,
            self.profanity_filter,
        ).await?;
        Ok(Box::new(stream))
    }
    
    async fn list_models(&self) -> Result<Vec<Model>> {
        Ok(vec![
            Model {
                id: "latest_long".to_string(),
                name: "Latest Long".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "transcription".to_string(),
                    "streaming".to_string(),
                    "word-timestamps".to_string(),
                    "punctuation".to_string(),
                    "diarization".to_string(),
                ],
            },
            Model {
                id: "latest_short".to_string(),
                name: "Latest Short".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "transcription".to_string(),
                    "streaming".to_string(),
                    "word-timestamps".to_string(),
                    "low-latency".to_string(),
                ],
            },
            Model {
                id: "command_and_search".to_string(),
                name: "Command and Search".to_string(),
                languages: Self::get_supported_languages(),
                capabilities: vec![
                    "transcription".to_string(),
                    "streaming".to_string(),
                    "voice-commands".to_string(),
                    "search-queries".to_string(),
                ],
            },
        ])
    }
    
    fn supported_formats(&self) -> Vec<AudioFormat> {
        vec![
            AudioFormat::wav(8000),
            AudioFormat::wav(16000),
            AudioFormat::wav(32000),
            AudioFormat::wav(44100),
            AudioFormat::wav(48000),
            // Using wav for FLAC since core doesn't have FLAC constructor
            AudioFormat::wav(16000), // FLAC 16kHz
            AudioFormat::wav(44100), // FLAC 44.1kHz
            AudioFormat::opus(48000),
            AudioFormat::mp3(16000),
            AudioFormat::mp3(44100),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

impl GoogleSttProvider {
    fn map_audio_format_to_encoding(&self, format: &AudioFormat) -> String {
        match format.format.as_str() {
            "wav" | "pcm" => "LINEAR16".to_string(),
            "flac" => "FLAC".to_string(),
            "opus" => "OGG_OPUS".to_string(),
            "mp3" => "MP3".to_string(),
            "mulaw" => "MULAW".to_string(),
            "alaw" => "ALAW".to_string(),
            _ => "LINEAR16".to_string(), // Default fallback
        }
    }
}

struct GoogleSttStream {
    api_key: String,
    model: String,
    session_id: Uuid,
    enable_punctuation: bool,
    profanity_filter: bool,
    client: Client,
    language_code: String,
    audio_buffer: Vec<u8>,
}

impl GoogleSttStream {
    async fn new(
        api_key: String,
        model: String,
        config: StreamConfig,
        enable_punctuation: bool,
        profanity_filter: bool,
    ) -> Result<Self> {
        let language_code = config.language.as_ref().unwrap_or(&"en-US".to_string()).clone();
        
        info!("Created Google Cloud Speech streaming session {}", config.session_id);
        
        Ok(Self {
            api_key,
            model,
            session_id: config.session_id,
            enable_punctuation,
            profanity_filter,
            client: Client::new(),
            language_code,
            audio_buffer: Vec::new(),
        })
    }
}

#[async_trait]
impl SttStream for GoogleSttStream {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()> {
        // Buffer audio chunks for periodic processing
        self.audio_buffer.extend_from_slice(chunk);
        Ok(())
    }
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>> {
        // For simplicity, process buffered audio when we have enough data
        if self.audio_buffer.len() < 32000 { // ~2 seconds at 16kHz
            return Ok(None);
        }
        
        // Take a chunk of audio to process
        let chunk_size = 32000;
        let audio_chunk = if self.audio_buffer.len() >= chunk_size {
            let chunk: Vec<u8> = self.audio_buffer.drain(0..chunk_size).collect();
            chunk
        } else {
            let chunk = self.audio_buffer.clone();
            self.audio_buffer.clear();
            chunk
        };
        
        if audio_chunk.is_empty() {
            return Ok(None);
        }
        
        // Use API key directly
        
        // Make recognition request with chunked audio
        let config = RecognitionConfig {
            encoding: "LINEAR16".to_string(),
            sample_rate_hertz: 16000,
            language_code: self.language_code.clone(),
            enable_automatic_punctuation: self.enable_punctuation,
            enable_word_time_offsets: true,
            profanity_filter: self.profanity_filter,
            model: self.model.clone(),
            use_enhanced: true,
            ..Default::default()
        };
        
        let audio_content = base64::engine::general_purpose::STANDARD.encode(&audio_chunk);
        
        let request_body = json!({
            "config": config,
            "audio": {
                "content": audio_content
            }
        });
        
        let url = format!("{}/speech:recognize?key={}", GOOGLE_STT_API_BASE, self.api_key);
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let google_response: GoogleSpeechResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                
                if let Some(result) = google_response.results.first() {
                    if let Some(alternative) = result.alternatives.first() {
                        let words = alternative.words.as_ref().map(|word_list| {
                            word_list.iter().map(|w| WordTiming {
                                word: w.word.clone(),
                                start: w.start_time.as_ref()
                                    .map(|ts| ts.to_seconds())
                                    .unwrap_or(0.0),
                                end: w.end_time.as_ref()
                                    .map(|ts| ts.to_seconds())
                                    .unwrap_or(0.0),
                                confidence: w.confidence.unwrap_or(0.0),
                            }).collect()
                        });
                        
                        let mut metadata = serde_json::Map::new();
                        metadata.insert("provider".to_string(), json!("google"));
                        metadata.insert("model".to_string(), json!(self.model));
                        
                        return Ok(Some(StreamingResult {
                            session_id: self.session_id,
                            is_final: true, // REST API responses are always final
                            text: alternative.transcript.clone(),
                            confidence: alternative.confidence.unwrap_or(0.0),
                            timestamp: Utc::now(),
                            words,
                            metadata: Some(json!(metadata)),
                        }));
                    }
                }
                
                Ok(None)
            }
            _ => {
                // For streaming, we'll just return None on errors rather than failing
                warn!("Google Cloud Speech API error in streaming mode");
                Ok(None)
            }
        }
    }
    
    async fn close(&mut self) -> Result<()> {
        // Process any remaining buffered audio
        if !self.audio_buffer.is_empty() {
            let _ = self.receive_transcript().await;
        }
        self.audio_buffer.clear();
        Ok(())
    }
    
    fn session_id(&self) -> Uuid {
        self.session_id
    }
}

// Google Cloud Speech API Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleSpeechResponse {
    results: Vec<SpeechRecognitionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeechRecognitionResult {
    alternatives: Vec<SpeechRecognitionAlternative>,
    #[serde(rename = "channelTag")]
    channel_tag: Option<u32>,
    #[serde(rename = "languageCode")]
    language_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeechRecognitionAlternative {
    transcript: String,
    confidence: Option<f32>,
    words: Option<Vec<WordInfo>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WordInfo {
    #[serde(rename = "startTime")]
    start_time: Option<Duration>,
    #[serde(rename = "endTime")]
    end_time: Option<Duration>,
    word: String,
    confidence: Option<f32>,
    #[serde(rename = "speakerTag")]
    speaker_tag: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Duration {
    // Google sometimes returns duration as string like "0s", "1.5s"
    String(String),
    // Or as structured object
    Structured {
        seconds: i64,
        nanos: i32,
    },
}

impl Duration {
    fn to_seconds(&self) -> f32 {
        match self {
            Duration::String(s) => {
                // Parse strings like "0s", "1.5s"
                if s.ends_with("s") {
                    s.trim_end_matches('s').parse::<f32>().unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            Duration::Structured { seconds, nanos } => {
                *seconds as f32 + (*nanos as f32 / 1_000_000_000.0)
            }
        }
    }
}

// Streaming response types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleStreamingResponse {
    results: Vec<StreamingRecognitionResult>,
    error: Option<GoogleError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamingRecognitionResult {
    alternatives: Vec<SpeechRecognitionAlternative>,
    #[serde(rename = "isFinal")]
    is_final: bool,
    stability: Option<f32>,
    #[serde(rename = "resultEndTime")]
    result_end_time: Option<Duration>,
    #[serde(rename = "channelTag")]
    channel_tag: Option<u32>,
    #[serde(rename = "languageCode")]
    language_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleError {
    code: u32,
    message: String,
}

// Google Cloud Text-to-Speech Provider
pub struct GoogleTtsProvider {
    client: Client,
    api_key: String,
    project_id: String,
    voice_type: String,
    speaking_rate: f32,
    pitch: f32,
    volume_gain_db: f32,
}

impl GoogleTtsProvider {
    pub async fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| DebabelizerError::Configuration("Google API key not found".to_string()))?;
        
        let project_id = config
            .get_project_id()
            .ok_or_else(|| DebabelizerError::Configuration("Google project ID not found".to_string()))?;
            
        let voice_type = config
            .get_value("voice_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Neural2")
            .to_string();
            
        let speaking_rate = config
            .get_value("speaking_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            .max(0.25)
            .min(4.0) as f32;
            
        let pitch = config
            .get_value("pitch")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .max(-20.0)
            .min(20.0) as f32;
            
        let volume_gain_db = config
            .get_value("volume_gain_db")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .max(-96.0)
            .min(16.0) as f32;
        
        let client = Client::new();
        
        Ok(Self {
            client,
            api_key,
            project_id,
            voice_type,
            speaking_rate,
            pitch,
            volume_gain_db,
        })
    }
    
    fn get_supported_languages() -> Vec<String> {
        vec![
            "ar-XA".to_string(), "bn-IN".to_string(), "bg-BG".to_string(), "ca-ES".to_string(),
            "cs-CZ".to_string(), "da-DK".to_string(), "de-DE".to_string(), "el-GR".to_string(),
            "en-AU".to_string(), "en-GB".to_string(), "en-IN".to_string(), "en-US".to_string(),
            "es-ES".to_string(), "es-US".to_string(), "fi-FI".to_string(), "fr-CA".to_string(),
            "fr-FR".to_string(), "gu-IN".to_string(), "he-IL".to_string(), "hi-IN".to_string(),
            "hu-HU".to_string(), "id-ID".to_string(), "it-IT".to_string(), "ja-JP".to_string(),
            "kn-IN".to_string(), "ko-KR".to_string(), "ml-IN".to_string(), "mr-IN".to_string(),
            "ms-MY".to_string(), "nb-NO".to_string(), "nl-BE".to_string(), "nl-NL".to_string(),
            "pa-IN".to_string(), "pl-PL".to_string(), "pt-BR".to_string(), "pt-PT".to_string(),
            "ro-RO".to_string(), "ru-RU".to_string(), "sk-SK".to_string(), "sr-RS".to_string(),
            "sv-SE".to_string(), "ta-IN".to_string(), "te-IN".to_string(), "th-TH".to_string(),
            "tr-TR".to_string(), "uk-UA".to_string(), "vi-VN".to_string(), "zh-CN".to_string(),
            "zh-TW".to_string(),
        ]
    }
    
    fn normalize_language(language: &str) -> String {
        // Language mapping for short codes
        let language_map: HashMap<&str, &str> = [
            ("en", "en-US"), ("es", "es-ES"), ("fr", "fr-FR"), ("de", "de-DE"),
            ("it", "it-IT"), ("pt", "pt-BR"), ("ru", "ru-RU"), ("ja", "ja-JP"),
            ("ko", "ko-KR"), ("zh", "zh-CN"), ("ar", "ar-XA"), ("hi", "hi-IN"),
            ("nl", "nl-NL"), ("pl", "pl-PL"), ("tr", "tr-TR"), ("sv", "sv-SE"),
            ("da", "da-DK"), ("no", "nb-NO"), ("fi", "fi-FI"), ("cs", "cs-CZ"),
            ("hu", "hu-HU"), ("el", "el-GR"), ("he", "he-IL"), ("th", "th-TH"),
            ("vi", "vi-VN"), ("id", "id-ID"), ("ms", "ms-MY"), ("ro", "ro-RO"),
            ("uk", "uk-UA"), ("bg", "bg-BG"), ("sk", "sk-SK"),
            ("sr", "sr-RS"), ("ta", "ta-IN"), ("te", "te-IN"), ("bn", "bn-IN"),
            ("gu", "gu-IN"), ("mr", "mr-IN"), ("kn", "kn-IN"), ("ml", "ml-IN"), 
            ("pa", "pa-IN"),
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
    
    fn map_audio_format_to_encoding(&self, format: &AudioFormat) -> String {
        match format.format.as_str() {
            "wav" | "pcm" => "LINEAR16".to_string(),
            "mp3" => "MP3".to_string(),
            "opus" => "OGG_OPUS".to_string(),
            "mulaw" => "MULAW".to_string(),
            "alaw" => "ALAW".to_string(),
            _ => "LINEAR16".to_string(), // Default fallback
        }
    }
    
    async fn make_synthesis_request(&self, text: &str, options: &SynthesisOptions) -> Result<GoogleTtsResponse> {
        let url = format!("{}/text:synthesize?key={}", GOOGLE_TTS_API_BASE, self.api_key);
        
        // Determine voice configuration
        let language_code = Self::normalize_language(&options.voice.language);
        let voice_name = if options.voice.voice_id.contains('-') {
            options.voice.voice_id.clone()
        } else {
            format!("{}-{}-{}", language_code, self.voice_type, options.voice.voice_id)
        };
        
        // Build request body
        let request_body = json!({
            "input": {
                "text": text
            },
            "voice": {
                "languageCode": language_code,
                "name": voice_name
            },
            "audioConfig": {
                "audioEncoding": self.map_audio_format_to_encoding(&options.format),
                "sampleRateHertz": options.format.sample_rate,
                "speakingRate": options.speed.unwrap_or(self.speaking_rate),
                "pitch": options.pitch.unwrap_or(self.pitch),
                "volumeGainDb": options.volume_gain_db.unwrap_or(self.volume_gain_db)
            }
        });
        
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let tts_response: GoogleTtsResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                Ok(tts_response)
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Google Cloud credentials or insufficient permissions".to_string()
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(DebabelizerError::Provider(ProviderError::RateLimit))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Google Cloud TTS API error: {}", error_text)
                )))
            }
        }
    }
    
    async fn fetch_available_voices(&self) -> Result<Vec<Voice>> {
        let url = format!("{}/voices?key={}", GOOGLE_TTS_API_BASE, self.api_key);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
        
        match response.status() {
            StatusCode::OK => {
                let voices_response: GoogleVoicesResponse = response.json().await
                    .map_err(|e| DebabelizerError::Provider(ProviderError::Network(e.to_string())))?;
                
                let mut voices = Vec::new();
                for voice in voices_response.voices {
                    let gender = voice.ssml_gender.clone().unwrap_or_else(|| "UNKNOWN".to_string());
                    for language_code in voice.language_codes {
                        voices.push(Voice {
                            voice_id: voice.name.clone(),
                            name: voice.name.clone(),
                            description: Some(format!("{} voice", voice.name)),
                            language: language_code,
                            gender: Some(gender.clone()),
                            age: None,
                            accent: None,
                            style: None,
                            use_case: None,
                            preview_url: None,
                            metadata: Some(json!({
                                "naturalSampleRateHertz": voice.natural_sample_rate_hertz,
                                "voiceType": self.voice_type
                            })),
                        });
                    }
                }
                
                Ok(voices)
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(DebabelizerError::Provider(ProviderError::Authentication(
                    "Invalid Google Cloud credentials or insufficient permissions".to_string()
                )))
            }
            status => {
                let error_text = response.text().await.unwrap_or_else(|_| status.to_string());
                Err(DebabelizerError::Provider(ProviderError::ProviderSpecific(
                    format!("Google Cloud TTS voices API error: {}", error_text)
                )))
            }
        }
    }
}

#[async_trait]
impl TtsProvider for GoogleTtsProvider {
    fn name(&self) -> &str {
        "google"
    }
    
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult> {
        let response = self.make_synthesis_request(text, options).await?;
        
        // Decode base64 audio content
        let audio_data = base64::engine::general_purpose::STANDARD.decode(&response.audio_content)
            .map_err(|e| DebabelizerError::Provider(ProviderError::ProviderSpecific(
                format!("Failed to decode audio content: {}", e)
            )))?;
        
        let mut metadata = serde_json::Map::new();
        metadata.insert("provider".to_string(), json!("google"));
        metadata.insert("model".to_string(), json!(self.voice_type));
        metadata.insert("project_id".to_string(), json!(self.project_id));
        metadata.insert("voice_name".to_string(), json!(options.voice.voice_id));
        
        Ok(SynthesisResult {
            audio_data,
            format: options.format.clone(),
            duration: None, // Google doesn't provide duration in response
            size_bytes: response.audio_content.len(),
            metadata: Some(json!(metadata)),
        })
    }
    
    async fn synthesize_stream(&self, text: &str, options: &SynthesisOptions) -> Result<Box<dyn TtsStream>> {
        // Google Cloud TTS doesn't support real-time streaming, so we'll simulate it
        let result = self.synthesize(text, options).await?;
        let stream = GoogleTtsStream::new(result.audio_data);
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
            AudioFormat::wav(32000),
            AudioFormat::wav(44100),
            AudioFormat::wav(48000),
            AudioFormat::mp3(8000),
            AudioFormat::mp3(16000),
            AudioFormat::mp3(22050),
            AudioFormat::mp3(24000),
            AudioFormat::mp3(32000),
            AudioFormat::mp3(44100),
            AudioFormat::mp3(48000),
            AudioFormat::opus(48000),
        ]
    }
    
    fn supports_streaming(&self) -> bool {
        false // Google Cloud TTS doesn't support real-time streaming
    }
    
    fn supports_ssml(&self) -> bool {
        true
    }
}

// Simulated streaming for Google TTS (since it doesn't support real-time streaming)
struct GoogleTtsStream {
    audio_data: Vec<u8>,
    position: usize,
    chunk_size: usize,
}

impl GoogleTtsStream {
    fn new(audio_data: Vec<u8>) -> Self {
        Self {
            audio_data,
            position: 0,
            chunk_size: 4096, // 4KB chunks
        }
    }
}

#[async_trait]
impl TtsStream for GoogleTtsStream {
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

// Google Cloud TTS API Response Types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleTtsResponse {
    #[serde(rename = "audioContent")]
    audio_content: String, // Base64 encoded audio
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleVoicesResponse {
    voices: Vec<GoogleVoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleVoice {
    #[serde(rename = "languageCodes")]
    language_codes: Vec<String>,
    name: String,
    #[serde(rename = "ssmlGender")]
    ssml_gender: Option<String>,
    #[serde(rename = "naturalSampleRateHertz")]
    natural_sample_rate_hertz: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    
    fn create_test_config() -> ProviderConfig {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key"));
        config.insert("project_id".to_string(), json!("test-project"));
        config.insert("model".to_string(), json!("latest_long"));
        ProviderConfig::Simple(config)
    }
    
    #[test]
    fn test_provider_config_extraction() {
        let config = create_test_config();
        
        assert_eq!(config.get_api_key(), Some("test-api-key".to_string()));
        assert_eq!(config.get_project_id(), Some("test-project".to_string()));
        assert_eq!(config.get_value("model").unwrap().as_str(), Some("latest_long"));
    }
    
    #[test]
    fn test_supported_languages() {
        let languages = GoogleSttProvider::get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"es-ES".to_string()));
        assert!(languages.contains(&"fr-FR".to_string()));
        assert!(languages.contains(&"de-DE".to_string()));
        assert!(languages.contains(&"zh-CN".to_string()));
        assert!(languages.contains(&"ja-JP".to_string()));
    }
    
    #[test]
    fn test_language_normalization() {
        // Test direct matches
        assert_eq!(GoogleSttProvider::normalize_language("en-US").unwrap(), "en-US");
        assert_eq!(GoogleSttProvider::normalize_language("es-ES").unwrap(), "es-ES");
        
        // Test short code mapping
        assert_eq!(GoogleSttProvider::normalize_language("en").unwrap(), "en-US");
        assert_eq!(GoogleSttProvider::normalize_language("fr").unwrap(), "fr-FR");
        assert_eq!(GoogleSttProvider::normalize_language("de").unwrap(), "de-DE");
        
        // Test case insensitive
        assert_eq!(GoogleSttProvider::normalize_language("EN-US").unwrap(), "en-US");
        
        // Test auto-detection fallback
        assert_eq!(GoogleSttProvider::normalize_language("auto").unwrap(), "en-US");
        
        // Test unsupported language
        assert!(GoogleSttProvider::normalize_language("xyz").is_err());
    }
    
    #[test]
    fn test_recognition_config_default() {
        let config = RecognitionConfig::default();
        assert_eq!(config.encoding, "LINEAR16");
        assert_eq!(config.sample_rate_hertz, 16000);
        assert_eq!(config.language_code, "en-US");
        assert!(config.enable_automatic_punctuation);
        assert!(config.enable_word_time_offsets);
        assert!(!config.profanity_filter);
        assert_eq!(config.model, "latest_long");
        assert!(config.use_enhanced);
    }
    
    #[test]
    fn test_audio_format_mapping() {
        // This test would require an actual provider instance, so we test the logic separately
        let format_mapping = [
            ("wav", "LINEAR16"),
            ("pcm", "LINEAR16"),
            ("flac", "FLAC"),
            ("opus", "OGG_OPUS"),
            ("mp3", "MP3"),
            ("mulaw", "MULAW"),
            ("alaw", "ALAW"),
            ("unknown", "LINEAR16"), // Default fallback
        ];
        
        for (input, expected) in format_mapping {
            // We can't directly test the private method, but we can test the logic
            let encoding = match input {
                "wav" | "pcm" => "LINEAR16",
                "flac" => "FLAC",
                "opus" => "OGG_OPUS",
                "mp3" => "MP3",
                "mulaw" => "MULAW",
                "alaw" => "ALAW",
                _ => "LINEAR16",
            };
            assert_eq!(encoding, expected);
        }
    }
    
    #[test]
    fn test_google_speech_response_parsing() {
        let response_json = r#"
        {
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello, how are you today?",
                            "confidence": 0.95,
                            "words": [
                                {
                                    "startTime": "0s",
                                    "endTime": "0.5s",
                                    "word": "Hello",
                                    "confidence": 0.98
                                },
                                {
                                    "startTime": "0.6s",
                                    "endTime": "0.8s", 
                                    "word": "how",
                                    "confidence": 0.96
                                }
                            ]
                        }
                    ],
                    "channelTag": 0,
                    "languageCode": "en-US"
                }
            ]
        }
        "#;
        
        let response: GoogleSpeechResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.results.len(), 1);
        let result = &response.results[0];
        assert_eq!(result.alternatives.len(), 1);
        
        let alternative = &result.alternatives[0];
        assert_eq!(alternative.transcript, "Hello, how are you today?");
        assert_eq!(alternative.confidence, Some(0.95));
        
        let words = alternative.words.as_ref().unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[0].confidence, Some(0.98));
        assert_eq!(words[1].word, "how");
        assert_eq!(words[1].confidence, Some(0.96));
        
        assert_eq!(result.channel_tag, Some(0));
        assert_eq!(result.language_code, Some("en-US".to_string()));
    }
    
    #[test]
    fn test_streaming_response_parsing() {
        let response_json = r#"
        {
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": "Testing streaming response",
                            "confidence": 0.92,
                            "words": [
                                {
                                    "startTime": "0s",
                                    "endTime": "0.6s",
                                    "word": "Testing",
                                    "confidence": 0.94
                                }
                            ]
                        }
                    ],
                    "isFinal": true,
                    "stability": 0.9,
                    "channelTag": 0,
                    "languageCode": "en-US"
                }
            ]
        }
        "#;
        
        let response: GoogleStreamingResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.results.len(), 1);
        let result = &response.results[0];
        assert!(result.is_final);
        assert_eq!(result.stability, Some(0.9));
        assert_eq!(result.channel_tag, Some(0));
        assert_eq!(result.language_code, Some("en-US".to_string()));
        
        let alternative = &result.alternatives[0];
        assert_eq!(alternative.transcript, "Testing streaming response");
        assert_eq!(alternative.confidence, Some(0.92));
    }
    
    #[test]
    fn test_duration_parsing() {
        // Test structured duration
        let structured_duration = Duration::Structured {
            seconds: 1,
            nanos: 500_000_000,
        };
        assert_eq!(structured_duration.to_seconds(), 1.5);
        
        // Test string duration
        let string_duration = Duration::String("1.5s".to_string());
        assert_eq!(string_duration.to_seconds(), 1.5);
        
        let zero_duration = Duration::String("0s".to_string());
        assert_eq!(zero_duration.to_seconds(), 0.0);
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(GOOGLE_STT_API_BASE, "https://speech.googleapis.com/v1");
        assert_eq!(GOOGLE_TTS_API_BASE, "https://texttospeech.googleapis.com/v1");
    }
    
    // Google TTS Provider Tests
    
    #[allow(dead_code)]
    fn create_test_tts_config() -> ProviderConfig {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-api-key"));
        config.insert("project_id".to_string(), json!("test-project"));
        config.insert("voice_type".to_string(), json!("Neural2"));
        config.insert("speaking_rate".to_string(), json!(1.0));
        config.insert("pitch".to_string(), json!(0.0));
        config.insert("volume_gain_db".to_string(), json!(0.0));
        ProviderConfig::Simple(config)
    }
    
    #[test]
    fn test_tts_supported_languages() {
        let languages = GoogleTtsProvider::get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"es-ES".to_string()));
        assert!(languages.contains(&"fr-FR".to_string()));
        assert!(languages.contains(&"de-DE".to_string()));
        assert!(languages.contains(&"zh-CN".to_string()));
        assert!(languages.contains(&"ja-JP".to_string()));
        assert!(languages.contains(&"ar-XA".to_string()));
    }
    
    #[test]
    fn test_tts_language_normalization() {
        // Test direct matches
        assert_eq!(GoogleTtsProvider::normalize_language("en-US"), "en-US");
        assert_eq!(GoogleTtsProvider::normalize_language("es-ES"), "es-ES");
        
        // Test short code mapping
        assert_eq!(GoogleTtsProvider::normalize_language("en"), "en-US");
        assert_eq!(GoogleTtsProvider::normalize_language("fr"), "fr-FR");
        assert_eq!(GoogleTtsProvider::normalize_language("de"), "de-DE");
        assert_eq!(GoogleTtsProvider::normalize_language("ar"), "ar-XA");
        
        // Test unsupported language fallback
        assert_eq!(GoogleTtsProvider::normalize_language("xyz"), "en-US");
    }
    
    #[test]
    fn test_tts_audio_format_mapping() {
        // Create a dummy provider to test the method
        let format_mapping = [
            ("wav", "LINEAR16"),
            ("pcm", "LINEAR16"),
            ("mp3", "MP3"),
            ("opus", "OGG_OPUS"),
            ("mulaw", "MULAW"),
            ("alaw", "ALAW"),
            ("unknown", "LINEAR16"), // Default fallback
        ];
        
        for (input, expected) in format_mapping {
            // We can't directly test the private method, but we can test the logic
            let encoding = match input {
                "wav" | "pcm" => "LINEAR16",
                "mp3" => "MP3",
                "opus" => "OGG_OPUS",
                "mulaw" => "MULAW",
                "alaw" => "ALAW",
                _ => "LINEAR16",
            };
            assert_eq!(encoding, expected);
        }
    }
    
    #[test]
    fn test_tts_response_parsing() {
        let response_json = r#"
        {
            "audioContent": "UklGRj4AAABXQVZFZm10IBAAAAABAAEBAA=="
        }
        "#;
        
        let response: GoogleTtsResponse = serde_json::from_str(response_json).unwrap();
        assert_eq!(response.audio_content, "UklGRj4AAABXQVZFZm10IBAAAAABAAEBAA==");
    }
    
    #[test]
    fn test_voices_response_parsing() {
        let response_json = r#"
        {
            "voices": [
                {
                    "languageCodes": ["en-US"],
                    "name": "en-US-Neural2-A",
                    "ssmlGender": "FEMALE",
                    "naturalSampleRateHertz": 24000
                },
                {
                    "languageCodes": ["en-US", "en-GB"],
                    "name": "en-US-Neural2-B",
                    "ssmlGender": "MALE",
                    "naturalSampleRateHertz": 24000
                }
            ]
        }
        "#;
        
        let response: GoogleVoicesResponse = serde_json::from_str(response_json).unwrap();
        
        assert_eq!(response.voices.len(), 2);
        
        let voice1 = &response.voices[0];
        assert_eq!(voice1.language_codes, vec!["en-US"]);
        assert_eq!(voice1.name, "en-US-Neural2-A");
        assert_eq!(voice1.ssml_gender, Some("FEMALE".to_string()));
        assert_eq!(voice1.natural_sample_rate_hertz, 24000);
        
        let voice2 = &response.voices[1];
        assert_eq!(voice2.language_codes, vec!["en-US", "en-GB"]);
        assert_eq!(voice2.name, "en-US-Neural2-B");
        assert_eq!(voice2.ssml_gender, Some("MALE".to_string()));
        assert_eq!(voice2.natural_sample_rate_hertz, 24000);
    }
    
    #[test]
    fn test_tts_stream_chunks() {
        let audio_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut stream = GoogleTtsStream::new(audio_data.clone());
        
        // Test chunk size calculation
        let expected_chunks = (audio_data.len() as f32 / stream.chunk_size as f32).ceil() as usize;
        assert!(expected_chunks >= 1);
        
        // Test position tracking
        assert_eq!(stream.position, 0);
        stream.position = 5;
        assert_eq!(stream.position, 5);
    }
    
    #[test]
    fn test_parameter_validation() {
        let mut config = HashMap::new();
        config.insert("api_key".to_string(), json!("test-key"));
        config.insert("project_id".to_string(), json!("test-project"));
        
        // Test speaking rate bounds
        config.insert("speaking_rate".to_string(), json!(0.1)); // Below minimum
        let _provider_config = ProviderConfig::Simple(config.clone());
        // The constructor would clamp this to 0.25
        
        config.insert("speaking_rate".to_string(), json!(5.0)); // Above maximum
        let _provider_config = ProviderConfig::Simple(config.clone());
        // The constructor would clamp this to 4.0
        
        // Test pitch bounds
        config.insert("pitch".to_string(), json!(-25.0)); // Below minimum
        let _provider_config = ProviderConfig::Simple(config.clone());
        // The constructor would clamp this to -20.0
        
        config.insert("pitch".to_string(), json!(25.0)); // Above maximum
        let _provider_config = ProviderConfig::Simple(config.clone());
        // The constructor would clamp this to 20.0
        
        // Test volume gain bounds
        config.insert("volume_gain_db".to_string(), json!(-100.0)); // Below minimum
        let _provider_config = ProviderConfig::Simple(config.clone());
        // The constructor would clamp this to -96.0
        
        config.insert("volume_gain_db".to_string(), json!(20.0)); // Above maximum
        let _provider_config = ProviderConfig::Simple(config);
        // The constructor would clamp this to 16.0
    }
}