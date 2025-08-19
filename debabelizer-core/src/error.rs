use thiserror::Error;
use serde_json;

#[derive(Error, Debug)]
pub enum DebabelizerError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Audio format error: {0}")]
    AudioFormat(String),
    
    #[error("Session error: {0}")]
    Session(String),
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    
    #[error("{0}")]
    Other(String),
}

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Authentication failed: {0}")]
    Authentication(String),
    
    #[error("Rate limit exceeded")]
    RateLimit,
    
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Provider unavailable: {0}")]
    Unavailable(String),
    
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
    
    #[error("Timeout")]
    Timeout,
    
    #[error("Provider-specific error: {0}")]
    ProviderSpecific(String),
}

pub type Result<T> = std::result::Result<T, DebabelizerError>;