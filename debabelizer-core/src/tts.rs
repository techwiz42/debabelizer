use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{AudioFormat, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub voice_id: String,
    pub name: String,
    pub description: Option<String>,
    pub language: String,
    pub gender: Option<String>,
    pub age: Option<String>,
    pub accent: Option<String>,
    pub style: Option<String>,
    pub use_case: Option<String>,
    pub preview_url: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

impl Voice {
    pub fn new(voice_id: String, name: String, language: String) -> Self {
        Self {
            voice_id,
            name,
            description: None,
            language,
            gender: None,
            age: None,
            accent: None,
            style: None,
            use_case: None,
            preview_url: None,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub audio_data: Vec<u8>,
    pub format: AudioFormat,
    pub duration: Option<f32>,
    pub size_bytes: usize,
    pub metadata: Option<serde_json::Value>,
}

impl SynthesisResult {
    pub fn new(audio_data: Vec<u8>, format: AudioFormat) -> Self {
        let size_bytes = audio_data.len();
        Self {
            audio_data,
            format,
            duration: None,
            size_bytes,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisOptions {
    pub voice: Voice,
    pub model: Option<String>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    pub volume_gain_db: Option<f32>,
    pub format: AudioFormat,
    pub sample_rate: Option<u32>,
    pub metadata: Option<serde_json::Value>,
    // Legacy/compatibility fields for test compatibility
    pub voice_id: Option<String>,
    pub stability: Option<f32>,
    pub similarity_boost: Option<f32>,
    pub output_format: Option<AudioFormat>,
}

impl SynthesisOptions {
    pub fn new(voice: Voice) -> Self {
        Self {
            voice,
            model: None,
            speed: None,
            pitch: None,
            volume_gain_db: None,
            format: AudioFormat::default(),
            sample_rate: None,
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        }
    }
}

#[async_trait]
pub trait TtsProvider: Send + Sync {
    fn name(&self) -> &str;
    
    async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult>;
    
    async fn synthesize_stream(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsStream>>;
    
    async fn list_voices(&self) -> Result<Vec<Voice>>;
    
    fn supported_formats(&self) -> Vec<AudioFormat>;
    
    fn supports_streaming(&self) -> bool {
        true
    }
    
    fn supports_ssml(&self) -> bool {
        false
    }
}

#[async_trait]
pub trait TtsStream: Send {
    async fn receive_chunk(&mut self) -> Result<Option<Bytes>>;
    
    async fn close(&mut self) -> Result<()>;
}