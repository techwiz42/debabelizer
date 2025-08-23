//! Core traits and types for the Debabelizer voice processing library
//!
//! This crate provides the fundamental abstractions used across all Debabelizer providers.

pub mod audio;
pub mod error;
pub mod stt;
pub mod tts;

pub use audio::*;
pub use error::*;
pub use stt::*;
pub use tts::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_creation() {
        let format = AudioFormat::wav(16000);
        assert_eq!(format.format, "wav");
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, Some(16));
    }

    #[test]
    fn test_audio_format_mp3() {
        let format = AudioFormat::mp3(44100);
        assert_eq!(format.format, "mp3");
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, None);
    }

    #[test]
    fn test_audio_format_opus() {
        let format = AudioFormat::opus(48000);
        assert_eq!(format.format, "opus");
        assert_eq!(format.sample_rate, 48000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bit_depth, None);
    }

    #[test]
    fn test_audio_data_creation() {
        let data = vec![1, 2, 3, 4];
        let format = AudioFormat::wav(16000);
        let audio = AudioData::new(data.clone(), format.clone());
        
        assert_eq!(audio.data, data);
        assert_eq!(audio.format, format);
    }

    #[test]
    fn test_transcription_result_default() {
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            confidence: 0.95,
            language_detected: Some("en".to_string()),
            duration: Some(2.5),
            words: None,
            metadata: None,
        };
        
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.language_detected, Some("en".to_string()));
        assert_eq!(result.duration, Some(2.5));
        assert!(result.words.is_none());
    }

    #[test]
    fn test_word_timing() {
        let word = WordTiming {
            word: "hello".to_string(),
            start: 0.5,
            end: 1.0,
            confidence: 0.98,
        };
        
        assert_eq!(word.word, "hello");
        assert_eq!(word.start, 0.5);
        assert_eq!(word.end, 1.0);
        assert_eq!(word.confidence, 0.98);
    }

    #[test]
    fn test_streaming_result() {
        let session_id = uuid::Uuid::new_v4();
        let result = StreamingResult::new(session_id, "Hello".to_string(), true, 0.9);
        
        assert_eq!(result.session_id, session_id);
        assert!(result.is_final);
        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.9);
        assert!(result.words.is_none());
    }

    #[test]
    fn test_synthesis_result() {
        let audio_data = vec![0u8; 1000];
        let result = SynthesisResult {
            audio_data: audio_data.clone(),
            format: AudioFormat::wav(22050),
            duration: Some(1.5),
            size_bytes: 1000,
            metadata: None,
        };
        
        assert_eq!(result.audio_data, audio_data);
        assert_eq!(result.format.sample_rate, 22050);
        assert_eq!(result.duration, Some(1.5));
        assert_eq!(result.size_bytes, 1000);
    }

    #[test]
    fn test_voice_creation() {
        let voice = Voice::new(
            "voice-1".to_string(),
            "Alice".to_string(),
            "en-US".to_string(),
        );
        
        assert_eq!(voice.voice_id, "voice-1");
        assert_eq!(voice.name, "Alice");
        assert_eq!(voice.language, "en-US");
        assert!(voice.description.is_none());
    }

    #[test]
    fn test_voice_with_description() {
        let mut voice = Voice::new(
            "voice-2".to_string(),
            "Bob".to_string(),
            "en-US".to_string(),
        );
        voice.description = Some("A friendly male voice".to_string());
        
        assert_eq!(voice.description, Some("A friendly male voice".to_string()));
    }

    #[test]
    fn test_model_creation() {
        let model = Model {
            id: "whisper-large".to_string(),
            name: "OpenAI Whisper Large".to_string(),
            languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
            capabilities: vec!["transcription".to_string(), "translation".to_string()],
        };
        
        assert_eq!(model.id, "whisper-large");
        assert_eq!(model.name, "OpenAI Whisper Large");
        assert_eq!(model.languages.len(), 3);
        assert_eq!(model.capabilities.len(), 2);
    }

    #[test]
    fn test_stream_config() {
        let config = StreamConfig {
            session_id: uuid::Uuid::new_v4(),
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
        
        assert_eq!(config.language, Some("en-US".to_string()));
        assert_eq!(config.model, Some("nova-2".to_string()));
        assert!(config.interim_results);
        assert!(config.punctuate);
        assert!(!config.profanity_filter);
    }

    #[test]
    fn test_synthesis_options() {
        let voice = Voice::new("voice-123".to_string(), "Test Voice".to_string(), "en-US".to_string());
        let options = SynthesisOptions {
            voice: voice.clone(),
            model: Some("tts-1".to_string()),
            speed: Some(1.2),
            pitch: Some(0.5),
            volume_gain_db: Some(-5.0),
            format: AudioFormat::mp3(44100),
            sample_rate: Some(44100),
            metadata: None,
            voice_id: Some("voice-123".to_string()),
            stability: Some(0.5),
            similarity_boost: Some(0.8),
            output_format: Some(AudioFormat::mp3(44100)),
        };
        
        assert_eq!(options.voice.voice_id, "voice-123");
        assert_eq!(options.model, Some("tts-1".to_string()));
        assert_eq!(options.speed, Some(1.2));
        assert_eq!(options.pitch, Some(0.5));
    }

    #[test]
    fn test_error_types() {
        let config_error = DebabelizerError::Configuration("Invalid API key".to_string());
        assert!(matches!(config_error, DebabelizerError::Configuration(_)));
        
        let provider_error = DebabelizerError::Provider(ProviderError::Network("API timeout".to_string()));
        assert!(matches!(provider_error, DebabelizerError::Provider(_)));
        
        let audio_error = DebabelizerError::AudioFormat("Unsupported format".to_string());
        assert!(matches!(audio_error, DebabelizerError::AudioFormat(_)));
    }

    #[test]
    fn test_provider_error() {
        let error = ProviderError::RateLimit;
        assert!(matches!(error, ProviderError::RateLimit));
        
        let auth_error = ProviderError::Authentication("Invalid key".to_string());
        assert!(matches!(auth_error, ProviderError::Authentication(_)));
        
        let network_error = ProviderError::Network("Connection failed".to_string());
        assert!(matches!(network_error, ProviderError::Network(_)));
    }
}