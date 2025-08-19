use std::sync::Arc;

use crate::config::{DebabelizerConfig, OptimizationStrategy};
use crate::Result;
use debabelizer_core::{DebabelizerError, SttProvider, TtsProvider};

#[derive(Clone)]
pub struct ProviderRegistry {
    pub stt_providers: Vec<(String, Arc<dyn SttProvider>)>,
    pub tts_providers: Vec<(String, Arc<dyn TtsProvider>)>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            stt_providers: Vec::new(),
            tts_providers: Vec::new(),
        }
    }
    
    pub fn register_stt(&mut self, name: String, provider: Arc<dyn SttProvider>) {
        self.stt_providers.push((name, provider));
    }
    
    pub fn register_tts(&mut self, name: String, provider: Arc<dyn TtsProvider>) {
        self.tts_providers.push((name, provider));
    }
    
    pub fn get_stt(&self, name: &str) -> Option<Arc<dyn SttProvider>> {
        self.stt_providers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.clone())
    }
    
    pub fn get_tts(&self, name: &str) -> Option<Arc<dyn TtsProvider>> {
        self.tts_providers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.clone())
    }
    
    pub fn list_stt_providers(&self) -> Vec<String> {
        self.stt_providers.iter().map(|(n, _)| n.clone()).collect()
    }
    
    pub fn list_tts_providers(&self) -> Vec<String> {
        self.tts_providers.iter().map(|(n, _)| n.clone()).collect()
    }
    
    // Alias methods for test compatibility
    pub fn get_stt_provider(&self, name: &str) -> Option<Arc<dyn SttProvider>> {
        self.get_stt(name)
    }
    
    pub fn get_tts_provider(&self, name: &str) -> Option<Arc<dyn TtsProvider>> {
        self.get_tts(name)
    }
}

#[allow(unused_variables)]
pub async fn initialize_providers(config: &DebabelizerConfig) -> Result<ProviderRegistry> {
    let mut registry = ProviderRegistry::new();
    
    // Initialize Soniox (default STT)
    #[cfg(feature = "soniox")]
    {
        if let Some(provider_config) = config.get_provider_config("soniox") {
            let soniox_config = convert_to_soniox_config(provider_config);
            if let Ok(provider) = debabelizer_soniox::SonioxProvider::new(&soniox_config).await {
                registry.register_stt("soniox".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize Deepgram STT
    #[cfg(feature = "deepgram")]
    {
        if let Some(provider_config) = config.get_provider_config("deepgram") {
            let deepgram_config = convert_to_deepgram_config(provider_config);
            if let Ok(provider) = debabelizer_deepgram::DeepgramProvider::new(&deepgram_config).await {
                registry.register_stt("deepgram".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize ElevenLabs (default TTS)
    #[cfg(feature = "elevenlabs")]
    {
        if let Some(provider_config) = config.get_provider_config("elevenlabs") {
            let elevenlabs_config = convert_to_elevenlabs_config(provider_config);
            if let Ok(provider) = debabelizer_elevenlabs::ElevenLabsProvider::new(&elevenlabs_config).await {
                registry.register_tts("elevenlabs".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize OpenAI TTS
    #[cfg(feature = "openai")]
    {
        if let Some(provider_config) = config.get_provider_config("openai") {
            let openai_config = convert_to_openai_config(provider_config);
            if let Ok(provider) = debabelizer_openai::OpenAIProvider::new(&openai_config).await {
                registry.register_tts("openai".to_string(), Arc::new(provider));
            }
        }
    }
    
    Ok(registry)
}

pub fn select_stt_provider(
    registry: &ProviderRegistry,
    config: &DebabelizerConfig,
    requested_provider: Option<&str>,
) -> Result<Arc<dyn SttProvider>> {
    // 1. Explicit provider request
    if let Some(name) = requested_provider {
        return registry
            .get_stt(name)
            .ok_or_else(|| DebabelizerError::Configuration(format!("STT provider '{}' not found", name)));
    }
    
    // 2. User preference
    if let Some(name) = config.get_preferred_stt_provider() {
        if let Some(provider) = registry.get_stt(name) {
            return Ok(provider);
        }
    }
    
    // 3. Auto-selection
    if config.is_auto_select_enabled() {
        if let Some(provider) = auto_select_stt_provider(registry, config.get_optimization_strategy()) {
            return Ok(provider);
        }
    }
    
    // 4. First available
    registry
        .stt_providers
        .first()
        .map(|(_, p)| p.clone())
        .ok_or_else(|| DebabelizerError::Configuration("No STT providers available".to_string()))
}

pub fn select_tts_provider(
    registry: &ProviderRegistry,
    config: &DebabelizerConfig,
    requested_provider: Option<&str>,
) -> Result<Arc<dyn TtsProvider>> {
    // 1. Explicit provider request
    if let Some(name) = requested_provider {
        return registry
            .get_tts(name)
            .ok_or_else(|| DebabelizerError::Configuration(format!("TTS provider '{}' not found", name)));
    }
    
    // 2. User preference
    if let Some(name) = config.get_preferred_tts_provider() {
        if let Some(provider) = registry.get_tts(name) {
            return Ok(provider);
        }
    }
    
    // 3. Auto-selection
    if config.is_auto_select_enabled() {
        if let Some(provider) = auto_select_tts_provider(registry, config.get_optimization_strategy()) {
            return Ok(provider);
        }
    }
    
    // 4. First available
    registry
        .tts_providers
        .first()
        .map(|(_, p)| p.clone())
        .ok_or_else(|| DebabelizerError::Configuration("No TTS providers available".to_string()))
}

fn auto_select_stt_provider(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<Arc<dyn SttProvider>> {
    // Simple strategy-based selection
    let preferred_order = match strategy {
        OptimizationStrategy::Cost => vec!["whisper", "deepgram", "google", "azure", "soniox"],
        OptimizationStrategy::Latency => vec!["deepgram", "soniox", "google", "azure", "whisper"],
        OptimizationStrategy::Quality => vec!["google", "azure", "deepgram", "soniox", "whisper"],
        OptimizationStrategy::Balanced => vec!["deepgram", "google", "soniox", "azure", "whisper"],
    };
    
    for name in preferred_order {
        if let Some(provider) = registry.get_stt(name) {
            return Some(provider);
        }
    }
    
    None
}

#[cfg(feature = "soniox")]
fn convert_to_soniox_config(config: &crate::config::ProviderConfig) -> debabelizer_soniox::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_soniox::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_soniox::ProviderConfig::Simple(extra.clone())
        }
    }
}

#[cfg(feature = "deepgram")]
fn convert_to_deepgram_config(config: &crate::config::ProviderConfig) -> debabelizer_deepgram::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_deepgram::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_deepgram::ProviderConfig::Simple(extra.clone())
        }
    }
}

#[cfg(feature = "elevenlabs")]
fn convert_to_elevenlabs_config(config: &crate::config::ProviderConfig) -> debabelizer_elevenlabs::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_elevenlabs::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_elevenlabs::ProviderConfig::Simple(extra.clone())
        }
    }
}

#[cfg(feature = "openai")]
fn convert_to_openai_config(config: &crate::config::ProviderConfig) -> debabelizer_openai::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_openai::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_openai::ProviderConfig::Simple(extra.clone())
        }
    }
}

fn auto_select_tts_provider(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<Arc<dyn TtsProvider>> {
    // Simple strategy-based selection
    let preferred_order = match strategy {
        OptimizationStrategy::Cost => vec!["openai", "google", "azure", "elevenlabs"],
        OptimizationStrategy::Latency => vec!["openai", "elevenlabs", "google", "azure"],
        OptimizationStrategy::Quality => vec!["elevenlabs", "azure", "google", "openai"],
        OptimizationStrategy::Balanced => vec!["openai", "google", "elevenlabs", "azure"],
    };
    
    for name in preferred_order {
        if let Some(provider) = registry.get_tts(name) {
            return Some(provider);
        }
    }
    
    None
}
