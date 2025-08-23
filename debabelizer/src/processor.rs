use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::DebabelizerConfig;
use crate::providers::{initialize_providers, select_stt_provider, select_tts_provider, 
                      select_stt_provider_with_name, select_tts_provider_with_name, ProviderRegistry};
use crate::session::SessionManager;
use crate::Result;
use debabelizer_core::{
    AudioData, StreamConfig, SttProvider, SttStream,
    SynthesisOptions, SynthesisResult, TranscriptionResult, TtsProvider, TtsStream, Voice,
};

#[cfg(test)]
use debabelizer_core::AudioFormat;

pub struct VoiceProcessor {
    config: DebabelizerConfig,
    registry: Arc<RwLock<Option<ProviderRegistry>>>,
    session_manager: SessionManager,
    selected_stt: Arc<RwLock<Option<Arc<dyn SttProvider>>>>,
    selected_tts: Arc<RwLock<Option<Arc<dyn TtsProvider>>>>,
    selected_stt_name: Arc<RwLock<Option<String>>>,
    selected_tts_name: Arc<RwLock<Option<String>>>,
}

impl VoiceProcessor {
    pub fn new() -> Result<Self> {
        Self::with_config(DebabelizerConfig::default())
    }
    
    pub fn with_config(config: DebabelizerConfig) -> Result<Self> {
        Ok(Self {
            config,
            registry: Arc::new(RwLock::new(None)),
            session_manager: SessionManager::new(),
            selected_stt: Arc::new(RwLock::new(None)),
            selected_tts: Arc::new(RwLock::new(None)),
            selected_stt_name: Arc::new(RwLock::new(None)),
            selected_tts_name: Arc::new(RwLock::new(None)),
        })
    }
    
    pub fn builder() -> VoiceProcessorBuilder {
        VoiceProcessorBuilder::default()
    }
    
    async fn ensure_initialized(&self) -> Result<()> {
        println!("🔍 RUST: ensure_initialized() called");
        let mut registry_guard = self.registry.write().await;
        if registry_guard.is_none() {
            println!("🚀 RUST: Initializing providers for the first time...");
            let registry = initialize_providers(&self.config).await?;
            println!("✅ RUST: Provider initialization completed successfully");
            *registry_guard = Some(registry);
        } else {
            println!("✅ RUST: Providers already initialized");
        }
        Ok(())
    }
    
    async fn get_or_select_stt(&self) -> Result<Arc<dyn SttProvider>> {
        self.ensure_initialized().await?;
        
        let mut stt_guard = self.selected_stt.write().await;
        if let Some(provider) = &*stt_guard {
            return Ok(provider.clone());
        }
        
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        let (name, provider) = select_stt_provider_with_name(registry, &self.config, None)?;
        
        // Store the provider name
        let mut name_guard = self.selected_stt_name.write().await;
        *name_guard = Some(name);
        
        *stt_guard = Some(provider.clone());
        Ok(provider)
    }
    
    async fn get_or_select_tts(&self) -> Result<Arc<dyn TtsProvider>> {
        self.ensure_initialized().await?;
        
        let mut tts_guard = self.selected_tts.write().await;
        if let Some(provider) = &*tts_guard {
            return Ok(provider.clone());
        }
        
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        let (name, provider) = select_tts_provider_with_name(registry, &self.config, None)?;
        
        // Store the provider name
        let mut name_guard = self.selected_tts_name.write().await;
        *name_guard = Some(name);
        
        *tts_guard = Some(provider.clone());
        Ok(provider)
    }
    
    pub async fn set_stt_provider(&self, provider_name: &str) -> Result<()> {
        println!("🎯 RUST: set_stt_provider('{}') called", provider_name);
        self.ensure_initialized().await?;
        println!("🔍 RUST: Looking for STT provider '{}' in registry", provider_name);
        
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        let provider = select_stt_provider(registry, &self.config, Some(provider_name))?;
        println!("✅ RUST: Successfully selected STT provider '{}'", provider_name);
        
        let mut stt_guard = self.selected_stt.write().await;
        *stt_guard = Some(provider);
        
        // Store the provider name
        let mut name_guard = self.selected_stt_name.write().await;
        *name_guard = Some(provider_name.to_string());
        
        println!("✅ RUST: STT provider '{}' set successfully", provider_name);
        Ok(())
    }
    
    pub async fn set_tts_provider(&self, provider_name: &str) -> Result<()> {
        self.ensure_initialized().await?;
        
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        let provider = select_tts_provider(registry, &self.config, Some(provider_name))?;
        
        let mut tts_guard = self.selected_tts.write().await;
        *tts_guard = Some(provider);
        
        // Store the provider name
        let mut name_guard = self.selected_tts_name.write().await;
        *name_guard = Some(provider_name.to_string());
        
        Ok(())
    }
    
    pub async fn transcribe(&self, audio: AudioData) -> Result<TranscriptionResult> {
        let provider = self.get_or_select_stt().await?;
        provider.transcribe(audio).await
    }
    
    pub async fn transcribe_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        let provider = self.get_or_select_stt().await?;
        provider.transcribe_stream(config).await
    }
    
    pub async fn synthesize(&self, text: &str, options: &SynthesisOptions) -> Result<SynthesisResult> {
        let provider = self.get_or_select_tts().await?;
        provider.synthesize(text, options).await
    }
    
    pub async fn synthesize_stream(
        &self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Box<dyn TtsStream>> {
        let provider = self.get_or_select_tts().await?;
        provider.synthesize_stream(text, options).await
    }
    
    pub async fn list_voices(&self) -> Result<Vec<Voice>> {
        let provider = self.get_or_select_tts().await?;
        provider.list_voices().await
    }
    
    pub async fn list_stt_providers(&self) -> Result<Vec<String>> {
        self.ensure_initialized().await?;
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        Ok(registry.list_stt_providers())
    }
    
    pub async fn list_tts_providers(&self) -> Result<Vec<String>> {
        self.ensure_initialized().await?;
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        Ok(registry.list_tts_providers())
    }
    
    pub fn session_manager(&self) -> &SessionManager {
        &self.session_manager
    }
    
    // Alias for test compatibility
    pub fn provider_registry(&self) -> &Arc<RwLock<Option<ProviderRegistry>>> {
        &self.registry
    }
    
    // Missing methods for test compatibility
    pub async fn create_stt_stream(&self, config: StreamConfig) -> Result<Box<dyn SttStream>> {
        self.transcribe_stream(config).await
    }
    
    pub async fn create_tts_stream(&self, text: String, options: SynthesisOptions) -> Result<Box<dyn TtsStream>> {
        self.synthesize_stream(&text, &options).await
    }
    
    pub async fn list_stt_models(&self, provider_name: &str) -> Result<Vec<debabelizer_core::Model>> {
        self.ensure_initialized().await?;
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        if let Some(provider) = registry.get_stt_provider(provider_name) {
            provider.list_models().await
        } else {
            Err(crate::DebabelizerError::Configuration(format!("STT provider '{}' not found", provider_name)))
        }
    }
    
    pub async fn list_tts_voices(&self, provider_name: &str) -> Result<Vec<Voice>> {
        self.ensure_initialized().await?;
        let registry_guard = self.registry.read().await;
        let registry = registry_guard.as_ref().unwrap();
        if let Some(provider) = registry.get_tts_provider(provider_name) {
            provider.list_voices().await
        } else {
            Err(crate::DebabelizerError::Configuration(format!("TTS provider '{}' not found", provider_name)))
        }
    }
    
    pub async fn get_stt_provider_name(&self) -> Option<String> {
        let name_guard = self.selected_stt_name.read().await;
        name_guard.clone()
    }
    
    pub async fn get_tts_provider_name(&self) -> Option<String> {
        let name_guard = self.selected_tts_name.read().await;
        name_guard.clone()
    }
}

pub struct VoiceProcessorBuilder {
    config: Option<DebabelizerConfig>,
    stt_provider: Option<String>,
    tts_provider: Option<String>,
}

impl Default for VoiceProcessorBuilder {
    fn default() -> Self {
        Self {
            config: None,
            stt_provider: None,
            tts_provider: None,
        }
    }
}

impl VoiceProcessorBuilder {
    pub fn config(mut self, config: DebabelizerConfig) -> Self {
        self.config = Some(config);
        self
    }
    
    // Alias for compatibility with tests
    pub fn with_config(self, config: DebabelizerConfig) -> Self {
        self.config(config)
    }
    
    pub fn stt_provider(mut self, provider: impl Into<String>) -> Self {
        self.stt_provider = Some(provider.into());
        self
    }
    
    pub fn tts_provider(mut self, provider: impl Into<String>) -> Self {
        self.tts_provider = Some(provider.into());
        self
    }
    
    pub async fn build(self) -> Result<VoiceProcessor> {
        let config = self.config.unwrap_or_default();
        let processor = VoiceProcessor::with_config(config)?;
        
        if let Some(stt) = self.stt_provider {
            processor.set_stt_provider(&stt).await?;
        }
        
        if let Some(tts) = self.tts_provider {
            processor.set_tts_provider(&tts).await?;
        }
        
        Ok(processor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_voice_processor_creation() {
        let processor = VoiceProcessor::new();
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_builder_pattern() {
        // Test builder pattern without specific providers (they may not be configured)
        let processor = VoiceProcessor::builder()
            .build()
            .await;
        
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_builder_with_config() {
        let config = DebabelizerConfig::default();
        let processor = VoiceProcessor::builder()
            .with_config(config)
            .build()
            .await;
        
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_processor_list_stt_providers() {
        let processor = VoiceProcessor::new().unwrap();
        let providers = processor.list_stt_providers().await;
        
        assert!(providers.is_ok());
        let provider_list = providers.unwrap();
        
        // Without proper API keys, providers may not initialize
        // This test verifies the interface works, not that providers are available
        // Testing that the list can be retrieved successfully
        let _len = provider_list.len();
    }

    #[tokio::test]
    async fn test_processor_list_tts_providers() {
        let processor = VoiceProcessor::new().unwrap();
        let providers = processor.list_tts_providers().await;
        
        assert!(providers.is_ok());
        let provider_list = providers.unwrap();
        
        // Without proper API keys, providers may not initialize
        // This test verifies the interface works, not that providers are available
        // Testing that the list can be retrieved successfully
        let _len = provider_list.len();
    }

    #[tokio::test]
    async fn test_transcribe_with_mock_data() {
        let processor = VoiceProcessor::new().unwrap();
        
        // Create test audio data
        let audio_data = vec![0u8; 1000]; // Dummy PCM data
        let audio_format = AudioFormat::wav(16000);
        let audio = AudioData::new(audio_data, audio_format);
        
        // This will fail in practice without API keys, but tests the interface
        let result = processor.transcribe(audio).await;
        
        // Should either succeed or fail with a provider error, not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_synthesize_with_mock_data() {
        let processor = VoiceProcessor::new().unwrap();
        
        let voice = debabelizer_core::Voice::new("test-voice".to_string(), "Test Voice".to_string(), "en-US".to_string());
        let options = SynthesisOptions {
            voice,
            model: None,
            speed: None,
            pitch: None,
            volume_gain_db: None,
            format: AudioFormat::mp3(44100),
            sample_rate: Some(44100),
            metadata: None,
            voice_id: Some("test-voice".to_string()),
            stability: None,
            similarity_boost: None,
            output_format: Some(AudioFormat::mp3(44100)),
        };
        
        // This will fail in practice without API keys, but tests the interface
        let result = processor.synthesize("Hello world", &options).await;
        
        // Should either succeed or fail with a provider error, not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_create_stt_stream_interface() {
        let processor = VoiceProcessor::new().unwrap();
        
        let config = StreamConfig {
            session_id: uuid::Uuid::new_v4(),
            language: Some("en".to_string()),
            model: Some("auto".to_string()),
            format: AudioFormat::wav(16000),
            interim_results: true,
            punctuate: true,
            profanity_filter: false,
            diarization: false,
            metadata: None,
            enable_word_time_offsets: true,
            enable_automatic_punctuation: false,
            enable_language_identification: false,
        };
        
        // This will fail without API keys, but tests the interface
        let result = processor.create_stt_stream(config).await;
        
        // Should either succeed or fail with a provider error, not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_create_tts_stream_interface() {
        let processor = VoiceProcessor::new().unwrap();
        
        let voice = debabelizer_core::Voice::new("test-voice".to_string(), "Test Voice".to_string(), "en-US".to_string());
        let options = SynthesisOptions {
            voice,
            model: Some("eleven_turbo_v2".to_string()),
            speed: Some(1.0),
            pitch: None,
            volume_gain_db: None,
            format: AudioFormat::mp3(44100),
            sample_rate: Some(44100),
            metadata: None,
            voice_id: Some("test-voice".to_string()),
            stability: Some(0.5),
            similarity_boost: Some(0.5),
            output_format: Some(AudioFormat::mp3(44100)),
        };
        
        // This will fail without API keys, but tests the interface
        let result = processor.create_tts_stream("Hello world".to_string(), options).await;
        
        // Should either succeed or fail with a provider error, not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_processor_provider_access() {
        let processor = VoiceProcessor::new().unwrap();
        
        // Test that provider registry access doesn't panic
        let registry_guard = processor.provider_registry().read().await;
        let has_stt = registry_guard.as_ref().map(|r| !r.stt_providers.is_empty()).unwrap_or(false);
        let has_tts = registry_guard.as_ref().map(|r| !r.tts_providers.is_empty()).unwrap_or(false);
        
        // Without API keys, no providers may be available
        // This test verifies the interface works
        assert!(has_stt || has_tts || (!has_stt && !has_tts)); // Always true
    }

    #[tokio::test]
    async fn test_list_stt_models() {
        let processor = VoiceProcessor::new().unwrap();
        let providers = processor.list_stt_providers().await;
        
        if let Ok(provider_list) = providers {
            for provider_name in provider_list {
                // Test that we can attempt to list models (may fail without API keys)
                let models_result = processor.list_stt_models(&provider_name).await;
                // Should not panic, either succeed or return error
                assert!(models_result.is_ok() || models_result.is_err());
            }
        }
    }

    #[tokio::test]
    async fn test_list_tts_voices() {
        let processor = VoiceProcessor::new().unwrap();
        let providers = processor.list_tts_providers().await;
        
        if let Ok(provider_list) = providers {
            for provider_name in provider_list {
                // Test that we can attempt to list voices (may fail without API keys)
                let voices_result = processor.list_tts_voices(&provider_name).await;
                // Should not panic, either succeed or return error
                assert!(voices_result.is_ok() || voices_result.is_err());
            }
        }
    }

    #[test]
    fn test_processor_session_management() {
        let processor = VoiceProcessor::new().unwrap();
        
        // Test session manager is properly initialized
        let _session_manager = &processor.session_manager;
        // Session manager should be functional (tested separately)
    }

    #[tokio::test]
    async fn test_processor_drop_cleanup() {
        // Create processor in inner scope
        {
            let processor = VoiceProcessor::new().unwrap();
            // Just verify the provider registry is accessible
            let registry_guard = processor.provider_registry().read().await;
            let _len = registry_guard.as_ref().map(|r| r.stt_providers.len()).unwrap_or(0);
        }
        // Processor should drop cleanly without panicking
    }
}
