//! Comprehensive error handling and edge case tests

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::config::ProviderConfig;
    use debabelizer_core::{
        AudioData, AudioFormat, DebabelizerError, ProviderError, Result,
        SynthesisOptions, TranscriptionResult, StreamConfig, StreamingResult,
        Voice, WordTiming, SynthesisResult, Model,
    };

    #[test]
    fn test_debabelizer_error_types() {
        // Test Configuration error
        let config_error = DebabelizerError::Configuration("Invalid API key".to_string());
        assert!(matches!(config_error, DebabelizerError::Configuration(_)));
        assert!(config_error.to_string().contains("Invalid API key"));

        // Test Provider error
        let provider_error = DebabelizerError::Provider(ProviderError::Network("Connection failed".to_string()));
        assert!(matches!(provider_error, DebabelizerError::Provider(_)));
        assert!(provider_error.to_string().contains("Connection failed"));

        // Test Audio format error
        let audio_error = DebabelizerError::AudioFormat("Unsupported format".to_string());
        assert!(matches!(audio_error, DebabelizerError::AudioFormat(_)));

        // Test Session error
        let session_error = DebabelizerError::Session("Session not found".to_string());
        assert!(matches!(session_error, DebabelizerError::Session(_)));

        // Test Network error
        let network_error = DebabelizerError::Network("Timeout".to_string());
        assert!(matches!(network_error, DebabelizerError::Network(_)));
    }

    #[test]
    fn test_provider_error_creation() {
        let error = ProviderError::RateLimit;
        assert!(matches!(error, ProviderError::RateLimit));
        assert!(error.to_string().contains("Rate limit"));

        let auth_error = ProviderError::Authentication("Invalid API key".to_string());
        assert!(matches!(auth_error, ProviderError::Authentication(_)));
        assert!(auth_error.to_string().contains("Invalid API key"));
    }

    #[test]
    fn test_provider_error_variants() {
        let timeout_error = ProviderError::Timeout;
        assert!(matches!(timeout_error, ProviderError::Timeout));
        
        let unavailable_error = ProviderError::Unavailable("Service down".to_string());
        assert!(matches!(unavailable_error, ProviderError::Unavailable(_)));
        assert!(unavailable_error.to_string().contains("Service down"));
    }

    #[test]
    fn test_audio_format_edge_cases() {
        // Test zero sample rate
        let format = AudioFormat {
            format: "wav".to_string(),
            sample_rate: 0,
            channels: 1,
            bit_depth: Some(16),
        };
        assert_eq!(format.sample_rate, 0);

        // Test very high sample rate
        let format = AudioFormat {
            format: "wav".to_string(),
            sample_rate: 192000,
            channels: 8,
            bit_depth: Some(32),
        };
        assert_eq!(format.sample_rate, 192000);
        assert_eq!(format.channels, 8);

        // Test no bit depth
        let format = AudioFormat {
            format: "mp3".to_string(),
            sample_rate: 44100,
            channels: 2,
            bit_depth: None,
        };
        assert!(format.bit_depth.is_none());
    }

    #[test]
    fn test_audio_data_edge_cases() {
        // Test empty audio data
        let empty_data = vec![];
        let format = AudioFormat::wav(16000);
        let audio = AudioData::new(empty_data.clone(), format.clone());
        assert!(audio.data.is_empty());

        // Test very large audio data
        let large_data = vec![0u8; 1_000_000]; // 1MB
        let audio = AudioData::new(large_data.clone(), format);
        assert_eq!(audio.data.len(), 1_000_000);
    }

    #[test]
    fn test_transcription_result_edge_cases() {
        // Test empty transcription
        let result = TranscriptionResult {
            text: "".to_string(),
            confidence: 0.0,
            language_detected: None,
            duration: None,
            words: None,
            metadata: None,
        };
        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);

        // Test maximum confidence
        let result = TranscriptionResult {
            text: "test".to_string(),
            confidence: 1.0,
            language_detected: Some("unknown".to_string()),
            duration: Some(0.0),
            words: Some(vec![]),
            metadata: None,
        };
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.duration, Some(0.0));
        assert!(result.words.as_ref().unwrap().is_empty());

        // Test very long text
        let long_text = "word ".repeat(10000);
        let result = TranscriptionResult {
            text: long_text.clone(),
            confidence: 0.5,
            language_detected: None,
            duration: None,
            words: None,
            metadata: None,
        };
        assert_eq!(result.text.len(), long_text.len());
    }

    #[test]
    fn test_word_timing_edge_cases() {
        // Test zero duration word
        let word = WordTiming {
            word: "".to_string(),
            start: 0.0,
            end: 0.0,
            confidence: 0.0,
        };
        assert!(word.word.is_empty());
        assert_eq!(word.start, word.end);

        // Test negative timing
        let word = WordTiming {
            word: "test".to_string(),
            start: -1.0,
            end: -0.5,
            confidence: 0.5,
        };
        assert!(word.start < 0.0);
        assert!(word.end < 0.0);

        // Test very long word
        let long_word = "a".repeat(1000);
        let word = WordTiming {
            word: long_word.clone(),
            start: 0.0,
            end: 1.0,
            confidence: 1.0,
        };
        assert_eq!(word.word.len(), 1000);
    }

    #[test]
    fn test_synthesis_options_edge_cases() {
        // Test minimal options
        let voice = Voice::new("".to_string(), "".to_string(), "".to_string());
        let options = SynthesisOptions {
            voice: voice.clone(),
            model: None,
            speed: None,
            pitch: None,
            volume_gain_db: None,
            format: AudioFormat::wav(8000),
            sample_rate: None,
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        };
        assert!(options.voice.voice_id.is_empty());
        assert!(options.speed.is_none());

        // Test extreme values
        let options = SynthesisOptions {
            voice: voice,
            model: Some("".to_string()),
            speed: Some(0.0),
            pitch: Some(-12.0),
            volume_gain_db: Some(-60.0),
            format: AudioFormat::wav(1),
            sample_rate: Some(1),
            metadata: None,
            voice_id: None,
            stability: None,
            similarity_boost: None,
            output_format: None,
        };
        assert_eq!(options.model, Some("".to_string()));
        assert_eq!(options.speed, Some(0.0));
        assert_eq!(options.format.sample_rate, 1);
    }

    #[test]
    fn test_stream_config_edge_cases() {
        // Test minimal config
        let config = StreamConfig {
            session_id: uuid::Uuid::nil(),
            language: None,
            model: None,
            format: AudioFormat::wav(8000),
            interim_results: false,
            punctuate: false,
            profanity_filter: false,
            diarization: false,
            metadata: None,
            enable_word_time_offsets: false,
            enable_automatic_punctuation: false,
            enable_language_identification: false,
        };
        assert!(config.language.is_none());
        assert!(!config.interim_results);

        // Test empty strings
        let config = StreamConfig {
            session_id: uuid::Uuid::new_v4(),
            language: Some("".to_string()),
            model: Some("".to_string()),
            format: AudioFormat::wav(48000),
            interim_results: true,
            punctuate: true,
            profanity_filter: true,
            diarization: true,
            metadata: None,
            enable_word_time_offsets: true,
            enable_automatic_punctuation: true,
            enable_language_identification: false,
        };
        assert_eq!(config.language, Some("".to_string()));
        assert_eq!(config.model, Some("".to_string()));
    }

    #[test]
    fn test_voice_edge_cases() {
        // Test empty voice data
        let voice = Voice::new("".to_string(), "".to_string(), "".to_string());
        assert!(voice.voice_id.is_empty());
        assert!(voice.name.is_empty());
        assert!(voice.language.is_empty());

        // Test very long voice data
        let long_string = "x".repeat(10000);
        let voice = Voice::new(
            long_string.clone(),
            long_string.clone(),
            long_string.clone(),
        );
        assert_eq!(voice.voice_id.len(), 10000);
        assert_eq!(voice.name.len(), 10000);
        assert_eq!(voice.language.len(), 10000);
    }

    #[test]
    fn test_model_edge_cases() {
        // Test empty model
        let model = Model {
            id: "".to_string(),
            name: "".to_string(),
            languages: vec![],
            capabilities: vec![],
        };
        assert!(model.id.is_empty());
        assert!(model.name.is_empty());
        assert!(model.languages.is_empty());
        assert!(model.capabilities.is_empty());

        // Test model with many languages
        let many_languages: Vec<String> = (0..1000).map(|i| format!("lang{}", i)).collect();
        let model = Model {
            id: "multilingual".to_string(),
            name: "Multilingual Model".to_string(),
            languages: many_languages.clone(),
            capabilities: vec!["transcription".to_string(), "translation".to_string()],
        };
        assert_eq!(model.languages.len(), 1000);
        assert_eq!(model.languages[999], "lang999");
    }

    #[tokio::test]
    async fn test_processor_with_invalid_config() {
        // Test processor creation with various invalid configurations
        let mut config = DebabelizerConfig::default();
        
        // Add invalid provider config
        config.providers.insert(
            "invalid".to_string(),
            ProviderConfig::Simple({
                let mut map = std::collections::HashMap::new();
                map.insert("invalid_key".to_string(), serde_json::json!("invalid_value"));
                map
            }),
        );

        // Processor should still create successfully
        let processor = VoiceProcessor::builder()
            .with_config(config)
            .build()
            .await;
        
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_processor_creation() {
        let mut handles = vec![];
        
        // Create multiple processors concurrently
        for _ in 0..10 {
            let handle = tokio::spawn(async {
                VoiceProcessor::new()
            });
            handles.push(handle);
        }
        
        // All should succeed
        let results = futures::future::join_all(handles).await;
        for result in results {
            assert!(result.unwrap().is_ok());
        }
    }

    #[test]
    fn test_error_serialization() {
        let error = DebabelizerError::Configuration("Test error".to_string());
        let error_string = error.to_string();
        assert!(error_string.contains("Test error"));
        
        // Test error can be formatted for display
        let formatted = format!("{}", error);
        assert!(formatted.contains("Test error"));
    }

    #[test]
    fn test_provider_error_serialization() {
        let error = ProviderError::ProviderSpecific("Test message".to_string());
        
        let error_string = error.to_string();
        assert!(error_string.contains("Test message"));
        
        // Test Display formatting
        let formatted = format!("{}", error);
        assert!(formatted.contains("Test message"));
        
        // Test other error types
        let auth_error = ProviderError::Authentication("Invalid credentials".to_string());
        assert!(auth_error.to_string().contains("Invalid credentials"));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that our Result type alias works correctly
        let success: Result<String> = Ok("success".to_string());
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), "success");
        
        let failure: Result<String> = Err(DebabelizerError::Configuration("failed".to_string()));
        assert!(failure.is_err());
    }

    #[test]
    fn test_streaming_result_edge_cases() {
        // Test empty streaming result
        let session_id = uuid::Uuid::nil();
        let result = StreamingResult::new(session_id, "".to_string(), false, 0.0);
        assert_eq!(result.session_id, session_id);
        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);

        // Test streaming result with empty word timings
        let session_id = uuid::Uuid::new_v4();
        let mut result = StreamingResult::new(session_id, "hello".to_string(), true, 1.0);
        result.words = Some(vec![]);
        assert!(result.words.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_synthesis_result_edge_cases() {
        // Test empty synthesis result
        let result = SynthesisResult {
            audio_data: vec![],
            format: AudioFormat::wav(8000),
            duration: None,
            size_bytes: 0,
            metadata: None,
        };
        assert!(result.audio_data.is_empty());
        assert_eq!(result.size_bytes, 0);

        // Test large synthesis result
        let large_audio = vec![0u8; 10_000_000]; // 10MB
        let result = SynthesisResult {
            audio_data: large_audio.clone(),
            format: AudioFormat::wav(48000),
            duration: Some(100.0),
            size_bytes: large_audio.len(),
            metadata: None,
        };
        assert_eq!(result.audio_data.len(), 10_000_000);
        assert_eq!(result.size_bytes, 10_000_000);
    }
}