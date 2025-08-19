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
    
    pub fn mp3(sample_rate: u32) -> Self {
        Self {
            format: "mp3".to_string(),
            sample_rate,
            channels: 1,
            bit_depth: None,
        }
    }
    
    pub fn opus(sample_rate: u32) -> Self {
        Self {
            format: "opus".to_string(),
            sample_rate,
            channels: 1,
            bit_depth: None,
        }
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::wav(16000)
    }
}

#[derive(Debug, Clone)]
pub struct AudioData {
    pub data: Vec<u8>,
    pub format: AudioFormat,
}

impl AudioData {
    pub fn new(data: Vec<u8>, format: AudioFormat) -> Self {
        Self { data, format }
    }
    
    pub fn from_wav(data: Vec<u8>, sample_rate: u32) -> Self {
        Self::new(data, AudioFormat::wav(sample_rate))
    }
}