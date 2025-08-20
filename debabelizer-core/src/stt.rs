use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{AudioData, AudioFormat, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub language_detected: Option<String>,
    pub duration: Option<f32>,
    pub words: Option<Vec<WordTiming>>,
    pub metadata: Option<serde_json::Value>,
}

impl Default for TranscriptionResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            confidence: 0.0,
            language_detected: None,
            duration: None,
            words: None,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResult {
    pub session_id: Uuid,
    pub is_final: bool,
    pub text: String,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
    pub words: Option<Vec<WordTiming>>,
    pub metadata: Option<serde_json::Value>,
}

impl StreamingResult {
    pub fn new(session_id: Uuid, text: String, is_final: bool, confidence: f32) -> Self {
        Self {
            session_id,
            is_final,
            text,
            confidence,
            timestamp: Utc::now(),
            words: None,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub session_id: Uuid,
    pub language: Option<String>,
    pub model: Option<String>,
    pub format: AudioFormat,
    pub interim_results: bool,
    pub punctuate: bool,
    pub profanity_filter: bool,
    pub diarization: bool,
    pub metadata: Option<serde_json::Value>,
    // Legacy/compatibility fields for test compatibility
    pub enable_word_time_offsets: bool,
    pub enable_automatic_punctuation: bool,
    pub enable_language_identification: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            language: None,
            model: None,
            format: AudioFormat::default(),
            interim_results: true,
            punctuate: true,
            profanity_filter: false,
            diarization: false,
            metadata: None,
            enable_word_time_offsets: false,
            enable_automatic_punctuation: false,
            enable_language_identification: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub languages: Vec<String>,
    pub capabilities: Vec<String>,
}

#[async_trait]
pub trait SttProvider: Send + Sync {
    fn name(&self) -> &str;
    
    async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult>;
    
    async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>>;
    
    async fn list_models(&self) -> Result<Vec<Model>>;
    
    fn supported_formats(&self) -> Vec<AudioFormat>;
    
    fn supports_streaming(&self) -> bool {
        true
    }
}

#[async_trait]
pub trait SttStream: Send {
    async fn send_audio(&mut self, chunk: &[u8]) -> Result<()>;
    
    async fn receive_transcript(&mut self) -> Result<Option<StreamingResult>>;
    
    async fn close(&mut self) -> Result<()>;
    
    fn session_id(&self) -> Uuid;
}