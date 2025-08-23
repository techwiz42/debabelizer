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
        println!("🔍 RUST: Checking for Soniox provider config...");
        if let Some(provider_config) = config.get_provider_config("soniox") {
            println!("✅ RUST: Found Soniox provider config: {:?}", provider_config);
            let soniox_config = convert_to_soniox_config(provider_config);
            println!("🚀 RUST: Creating Soniox provider...");
            if let Ok(provider) = debabelizer_soniox::SonioxProvider::new(&soniox_config).await {
                println!("✅ RUST: Soniox provider created successfully, registering...");
                registry.register_stt("soniox".to_string(), Arc::new(provider));
                println!("✅ RUST: Soniox provider registered as 'soniox'");
            } else {
                println!("❌ RUST: Failed to create Soniox provider");
            }
        } else {
            println!("❌ RUST: No Soniox provider config found");
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
    
    // Initialize Google Cloud STT
    #[cfg(feature = "google")]
    {
        if let Some(provider_config) = config.get_provider_config("google") {
            let google_config = convert_to_google_config(provider_config);
            if let Ok(provider) = debabelizer_google::GoogleSttProvider::new(&google_config).await {
                registry.register_stt("google".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize Google Cloud TTS
    #[cfg(feature = "google")]
    {
        if let Some(provider_config) = config.get_provider_config("google") {
            let google_config = convert_to_google_config(provider_config);
            if let Ok(provider) = debabelizer_google::GoogleTtsProvider::new(&google_config).await {
                registry.register_tts("google".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize Azure STT
    #[cfg(feature = "azure")]
    {
        if let Some(provider_config) = config.get_provider_config("azure") {
            let azure_config = convert_to_azure_config(provider_config);
            if let Ok(provider) = debabelizer_azure::AzureSttProvider::new(&azure_config).await {
                registry.register_stt("azure".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize Azure TTS
    #[cfg(feature = "azure")]
    {
        if let Some(provider_config) = config.get_provider_config("azure") {
            let azure_config = convert_to_azure_config(provider_config);
            if let Ok(provider) = debabelizer_azure::AzureTtsProvider::new(&azure_config).await {
                registry.register_tts("azure".to_string(), Arc::new(provider));
            }
        }
    }
    
    // Initialize Whisper STT (local)
    #[cfg(feature = "whisper")]
    {
        if let Some(provider_config) = config.get_provider_config("whisper") {
            let whisper_config = convert_to_whisper_config(provider_config);
            if let Ok(provider) = debabelizer_whisper::WhisperSttProvider::new(&whisper_config).await {
                registry.register_stt("whisper".to_string(), Arc::new(provider));
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
    let (_, provider) = select_stt_provider_with_name(registry, config, requested_provider)?;
    Ok(provider)
}

pub fn select_stt_provider_with_name(
    registry: &ProviderRegistry,
    config: &DebabelizerConfig,
    requested_provider: Option<&str>,
) -> Result<(String, Arc<dyn SttProvider>)> {
    // 1. Explicit provider request
    if let Some(name) = requested_provider {
        return registry
            .get_stt(name)
            .map(|p| (name.to_string(), p))
            .ok_or_else(|| DebabelizerError::Configuration(format!("STT provider '{}' not found", name)));
    }
    
    // 2. User preference
    if let Some(name) = config.get_preferred_stt_provider() {
        if let Some(provider) = registry.get_stt(name) {
            return Ok((name.to_string(), provider));
        }
    }
    
    // 3. Auto-selection
    if config.is_auto_select_enabled() {
        if let Some((name, provider)) = auto_select_stt_provider_with_name(registry, config.get_optimization_strategy()) {
            return Ok((name, provider));
        }
    }
    
    // 4. First available
    registry
        .stt_providers
        .first()
        .map(|(name, p)| (name.clone(), p.clone()))
        .ok_or_else(|| DebabelizerError::Configuration("No STT providers available".to_string()))
}

pub fn select_tts_provider(
    registry: &ProviderRegistry,
    config: &DebabelizerConfig,
    requested_provider: Option<&str>,
) -> Result<Arc<dyn TtsProvider>> {
    let (_, provider) = select_tts_provider_with_name(registry, config, requested_provider)?;
    Ok(provider)
}

pub fn select_tts_provider_with_name(
    registry: &ProviderRegistry,
    config: &DebabelizerConfig,
    requested_provider: Option<&str>,
) -> Result<(String, Arc<dyn TtsProvider>)> {
    // 1. Explicit provider request
    if let Some(name) = requested_provider {
        return registry
            .get_tts(name)
            .map(|p| (name.to_string(), p))
            .ok_or_else(|| DebabelizerError::Configuration(format!("TTS provider '{}' not found", name)));
    }
    
    // 2. User preference
    if let Some(name) = config.get_preferred_tts_provider() {
        if let Some(provider) = registry.get_tts(name) {
            return Ok((name.to_string(), provider));
        }
    }
    
    // 3. Auto-selection
    if config.is_auto_select_enabled() {
        if let Some((name, provider)) = auto_select_tts_provider_with_name(registry, config.get_optimization_strategy()) {
            return Ok((name, provider));
        }
    }
    
    // 4. First available
    registry
        .tts_providers
        .first()
        .map(|(name, p)| (name.clone(), p.clone()))
        .ok_or_else(|| DebabelizerError::Configuration("No TTS providers available".to_string()))
}

fn auto_select_stt_provider(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<Arc<dyn SttProvider>> {
    auto_select_stt_provider_with_name(registry, strategy)
        .map(|(_, provider)| provider)
}

fn auto_select_stt_provider_with_name(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<(String, Arc<dyn SttProvider>)> {
    // Simple strategy-based selection
    let preferred_order = match strategy {
        OptimizationStrategy::Cost => vec!["whisper", "deepgram", "google", "azure", "soniox"],
        OptimizationStrategy::Latency => vec!["deepgram", "soniox", "google", "azure", "whisper"],
        OptimizationStrategy::Quality => vec!["google", "azure", "deepgram", "soniox", "whisper"],
        OptimizationStrategy::Balanced => vec!["deepgram", "google", "soniox", "azure", "whisper"],
    };
    
    for name in preferred_order {
        if let Some(provider) = registry.get_stt(name) {
            return Some((name.to_string(), provider));
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

#[cfg(feature = "google")]
fn convert_to_google_config(config: &crate::config::ProviderConfig) -> debabelizer_google::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_google::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_google::ProviderConfig::Simple(extra.clone())
        }
    }
}

#[cfg(feature = "azure")]
fn convert_to_azure_config(config: &crate::config::ProviderConfig) -> debabelizer_azure::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_azure::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_azure::ProviderConfig::Simple(extra.clone())
        }
    }
}

#[cfg(feature = "whisper")]
fn convert_to_whisper_config(config: &crate::config::ProviderConfig) -> debabelizer_whisper::ProviderConfig {
    match config {
        crate::config::ProviderConfig::Simple(map) => {
            debabelizer_whisper::ProviderConfig::Simple(map.clone())
        }
        crate::config::ProviderConfig::Detailed { extra, .. } => {
            debabelizer_whisper::ProviderConfig::Simple(extra.clone())
        }
    }
}

fn auto_select_tts_provider(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<Arc<dyn TtsProvider>> {
    auto_select_tts_provider_with_name(registry, strategy)
        .map(|(_, provider)| provider)
}

fn auto_select_tts_provider_with_name(
    registry: &ProviderRegistry,
    strategy: OptimizationStrategy,
) -> Option<(String, Arc<dyn TtsProvider>)> {
    // Simple strategy-based selection
    let preferred_order = match strategy {
        OptimizationStrategy::Cost => vec!["openai", "google", "azure", "elevenlabs"],
        OptimizationStrategy::Latency => vec!["openai", "elevenlabs", "google", "azure"],
        OptimizationStrategy::Quality => vec!["elevenlabs", "azure", "google", "openai"],
        OptimizationStrategy::Balanced => vec!["openai", "google", "elevenlabs", "azure"],
    };
    
    for name in preferred_order {
        if let Some(provider) = registry.get_tts(name) {
            return Some((name.to_string(), provider));
        }
    }
    
    None
}
