//! Universal voice processing library with support for multiple STT/TTS providers
//!
//! Debabelizer provides a unified interface for Speech-to-Text (STT) and Text-to-Speech (TTS)
//! services across multiple providers including OpenAI, Google Cloud, Azure, and more.

pub mod config;
pub mod error;
pub mod processor;
pub mod providers;
pub mod session;

#[cfg(test)]
mod error_tests;

#[cfg(test)]
mod performance_tests;

// Re-export core types
pub use debabelizer_core::{
    AudioData, AudioFormat, DebabelizerError, Model, ProviderError, Result, StreamConfig,
    StreamingResult, SttProvider, SttStream, SynthesisOptions, SynthesisResult, TranscriptionResult,
    TtsProvider, TtsStream, Voice, WordTiming,
};

// Re-export main types
pub use config::DebabelizerConfig;
pub use processor::VoiceProcessor;
pub use session::{Session, SessionManager, SessionStatus};

// Re-export utils
pub use debabelizer_utils as utils;