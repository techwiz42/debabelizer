use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebabelizerConfig {
    #[serde(default)]
    pub preferences: Preferences,
    
    #[serde(flatten)]
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preferences {
    pub stt_provider: Option<String>,
    pub tts_provider: Option<String>,
    #[serde(default = "default_auto_select")]
    pub auto_select: bool,
    #[serde(default)]
    pub optimize_for: OptimizationStrategy,
    pub timeout_seconds: Option<u64>,
    pub max_retries: Option<u32>,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            stt_provider: None,
            tts_provider: None,
            auto_select: default_auto_select(),
            optimize_for: OptimizationStrategy::default(),
            timeout_seconds: None,
            max_retries: None,
        }
    }
}

fn default_auto_select() -> bool {
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizationStrategy {
    Cost,
    Latency,
    Quality,
    Balanced,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

impl std::fmt::Display for OptimizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cost => write!(f, "cost"),
            Self::Latency => write!(f, "latency"),
            Self::Quality => write!(f, "quality"),
            Self::Balanced => write!(f, "balanced"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProviderConfig {
    Simple(HashMap<String, serde_json::Value>),
    Detailed {
        api_key: Option<String>,
        #[serde(flatten)]
        extra: HashMap<String, serde_json::Value>,
    },
}

impl ProviderConfig {
    pub fn get_api_key(&self) -> Option<String> {
        match self {
            Self::Simple(map) => map
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            Self::Detailed { api_key, .. } => api_key.clone(),
        }
    }
    
    pub fn get_value(&self, key: &str) -> Option<&serde_json::Value> {
        match self {
            Self::Simple(map) => map.get(key),
            Self::Detailed { extra, .. } => extra.get(key),
        }
    }
}

impl DebabelizerConfig {
    pub fn new() -> Result<Self, ConfigError> {
        let mut config = Config::builder();
        
        // Check for config file in standard locations
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("debabelizer").join("config");
            for ext in &["toml", "json", "yaml", "yml"] {
                let file_path = config_path.with_extension(ext);
                if file_path.exists() {
                    config = config.add_source(File::from(file_path));
                }
            }
        }
        
        // Check current directory
        for filename in &[".debabelizer", "debabelizer"] {
            for ext in &["toml", "json", "yaml", "yml"] {
                let file_path = PathBuf::from(format!("{}.{}", filename, ext));
                if file_path.exists() {
                    config = config.add_source(File::from(file_path));
                }
            }
        }
        
        // Add environment variables
        config = config.add_source(
            Environment::with_prefix("DEBABELIZER")
                .separator("_")
                .try_parsing(true),
        );
        
        // Also check for provider-specific env vars
        config = add_provider_env_vars(config);
        
        config.build()?.try_deserialize()
    }
    
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::new()
    }
    
    pub fn from_map(map: HashMap<String, serde_json::Value>) -> Result<Self, serde_json::Error> {
        serde_json::from_value(serde_json::Value::Object(
            map.into_iter().collect(),
        ))
    }
    
    pub fn get_provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }
    
    pub fn get_preferred_stt_provider(&self) -> Option<&str> {
        self.preferences.stt_provider.as_deref()
    }
    
    pub fn get_preferred_tts_provider(&self) -> Option<&str> {
        self.preferences.tts_provider.as_deref()
    }
    
    pub fn is_auto_select_enabled(&self) -> bool {
        self.preferences.auto_select
    }
    
    pub fn get_optimization_strategy(&self) -> OptimizationStrategy {
        self.preferences.optimize_for
    }
    
    pub fn get_timeout_seconds(&self) -> Option<u64> {
        self.preferences.timeout_seconds
    }
    
    pub fn get_max_retries(&self) -> Option<u32> {
        self.preferences.max_retries
    }
}

fn add_provider_env_vars(mut config: config::ConfigBuilder<config::builder::DefaultState>) -> config::ConfigBuilder<config::builder::DefaultState> {
    // Soniox
    if let Ok(api_key) = env::var("SONIOX_API_KEY") {
        config = config.set_override("soniox.api_key", api_key).expect("Failed to set soniox api key");
    }
    
    // Deepgram
    if let Ok(api_key) = env::var("DEEPGRAM_API_KEY") {
        config = config.set_override("deepgram.api_key", api_key).expect("Failed to set deepgram api key");
    }
    
    // OpenAI
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        config = config.set_override("openai.api_key", api_key).expect("Failed to set openai api key");
    }
    
    // ElevenLabs
    if let Ok(api_key) = env::var("ELEVENLABS_API_KEY") {
        config = config.set_override("elevenlabs.api_key", api_key).expect("Failed to set elevenlabs api key");
    }
    
    // Google Cloud
    if let Ok(creds) = env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        config = config.set_override("google.credentials_path", creds).expect("Failed to set google credentials");
    }
    if let Ok(api_key) = env::var("GOOGLE_API_KEY") {
        config = config.set_override("google.api_key", api_key).expect("Failed to set google api key");
    }
    if let Ok(project_id) = env::var("GOOGLE_PROJECT_ID") {
        config = config.set_override("google.project_id", project_id).expect("Failed to set google project id");
    } else if env::var("GOOGLE_API_KEY").is_ok() {
        // If API key is set but no project ID, use a default
        config = config.set_override("google.project_id", "default-project").expect("Failed to set google project id");
    }
    
    // Azure
    if let Ok(api_key) = env::var("AZURE_SPEECH_KEY") {
        config = config.set_override("azure.api_key", api_key).expect("Failed to set azure api key");
    }
    if let Ok(region) = env::var("AZURE_SPEECH_REGION") {
        config = config.set_override("azure.region", region).expect("Failed to set azure region");
    }
    
    // Whisper (local, no API key required)
    // Set default configuration to enable it
    config = config.set_override("whisper.model_size", "base").expect("Failed to set whisper model_size");
    if let Ok(model) = env::var("WHISPER_MODEL_SIZE") {
        config = config.set_override("whisper.model_size", model).expect("Failed to set whisper model_size");
    }
    
    config
}

impl Default for DebabelizerConfig {
    fn default() -> Self {
        Self {
            preferences: Preferences::default(),
            providers: HashMap::new(),
        }
    }
}

// Standalone function for test compatibility
pub fn config_from_env() -> config::ConfigBuilder<config::builder::DefaultState> {
    let mut config = Config::builder();
    
    // Add environment variables
    config = config.add_source(
        Environment::with_prefix("DEBABELIZER")
            .separator("_")
            .try_parsing(true),
    );
    
    // Add provider-specific env vars
    config = add_provider_env_vars(config);
    
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = DebabelizerConfig::default();
        assert!(!config.is_auto_select_enabled());
        assert_eq!(config.get_optimization_strategy(), OptimizationStrategy::Balanced);
    }
    
    #[test]
    fn test_provider_config() {
        let mut providers = HashMap::new();
        providers.insert(
            "openai".to_string(),
            ProviderConfig::Simple({
                let mut map = HashMap::new();
                map.insert("api_key".to_string(), serde_json::Value::String("test-key".to_string()));
                map
            }),
        );
        
        let config = DebabelizerConfig {
            preferences: Preferences::default(),
            providers,
        };
        
        let provider_config = config.get_provider_config("openai").unwrap();
        assert_eq!(provider_config.get_api_key(), Some("test-key".to_string()));
    }

    #[test]
    fn test_preferences_defaults() {
        let prefs = Preferences::default();
        
        assert!(prefs.stt_provider.is_none());
        assert!(prefs.tts_provider.is_none());
        assert!(!prefs.auto_select);
        assert_eq!(prefs.optimize_for, OptimizationStrategy::Balanced);
        assert!(prefs.timeout_seconds.is_none());
        assert!(prefs.max_retries.is_none());
    }

    #[test]
    fn test_preferences_custom() {
        let prefs = Preferences {
            stt_provider: Some("deepgram".to_string()),
            tts_provider: Some("elevenlabs".to_string()),
            auto_select: true,
            optimize_for: OptimizationStrategy::Quality,
            timeout_seconds: Some(30),
            max_retries: Some(3),
        };
        
        assert_eq!(prefs.stt_provider, Some("deepgram".to_string()));
        assert_eq!(prefs.tts_provider, Some("elevenlabs".to_string()));
        assert!(prefs.auto_select);
        assert_eq!(prefs.optimize_for, OptimizationStrategy::Quality);
        assert_eq!(prefs.timeout_seconds, Some(30));
        assert_eq!(prefs.max_retries, Some(3));
    }

    #[test]
    fn test_optimization_strategies() {
        assert_eq!(OptimizationStrategy::Cost.to_string(), "cost");
        assert_eq!(OptimizationStrategy::Latency.to_string(), "latency");
        assert_eq!(OptimizationStrategy::Quality.to_string(), "quality");
        assert_eq!(OptimizationStrategy::Balanced.to_string(), "balanced");
    }

    #[test]
    fn test_config_auto_select() {
        let mut config = DebabelizerConfig::default();
        assert!(!config.is_auto_select_enabled());
        
        config.preferences.auto_select = true;
        assert!(config.is_auto_select_enabled());
    }

    #[test]
    fn test_config_get_preferred_stt_provider() {
        let mut config = DebabelizerConfig::default();
        assert!(config.get_preferred_stt_provider().is_none());
        
        config.preferences.stt_provider = Some("deepgram".to_string());
        assert_eq!(config.get_preferred_stt_provider(), Some("deepgram"));
    }

    #[test]
    fn test_config_get_preferred_tts_provider() {
        let mut config = DebabelizerConfig::default();
        assert!(config.get_preferred_tts_provider().is_none());
        
        config.preferences.tts_provider = Some("elevenlabs".to_string());
        assert_eq!(config.get_preferred_tts_provider(), Some("elevenlabs"));
    }

    #[test]
    fn test_config_get_timeout() {
        let mut config = DebabelizerConfig::default();
        assert!(config.get_timeout_seconds().is_none());
        
        config.preferences.timeout_seconds = Some(45);
        assert_eq!(config.get_timeout_seconds(), Some(45));
    }

    #[test]
    fn test_config_get_max_retries() {
        let mut config = DebabelizerConfig::default();
        assert!(config.get_max_retries().is_none());
        
        config.preferences.max_retries = Some(5);
        assert_eq!(config.get_max_retries(), Some(5));
    }

    #[test]
    fn test_provider_config_missing() {
        let config = DebabelizerConfig::default();
        assert!(config.get_provider_config("nonexistent").is_none());
    }

    #[test]
    fn test_provider_config_with_multiple_providers() {
        let mut providers = HashMap::new();
        
        // Add ElevenLabs config
        providers.insert(
            "elevenlabs".to_string(),
            ProviderConfig::Simple({
                let mut map = HashMap::new();
                map.insert("api_key".to_string(), serde_json::Value::String("el-key".to_string()));
                map.insert("model".to_string(), serde_json::Value::String("eleven_turbo_v2".to_string()));
                map
            }),
        );
        
        // Add Deepgram config
        providers.insert(
            "deepgram".to_string(),
            ProviderConfig::Simple({
                let mut map = HashMap::new();
                map.insert("api_key".to_string(), serde_json::Value::String("dg-key".to_string()));
                map.insert("model".to_string(), serde_json::Value::String("nova-2".to_string()));
                map
            }),
        );
        
        let config = DebabelizerConfig {
            preferences: Preferences::default(),
            providers,
        };
        
        let el_config = config.get_provider_config("elevenlabs").unwrap();
        assert_eq!(el_config.get_api_key(), Some("el-key".to_string()));
        assert_eq!(el_config.get_value("model").unwrap().as_str(), Some("eleven_turbo_v2"));
        
        let dg_config = config.get_provider_config("deepgram").unwrap();
        assert_eq!(dg_config.get_api_key(), Some("dg-key".to_string()));
        assert_eq!(dg_config.get_value("model").unwrap().as_str(), Some("nova-2"));
    }

    #[test]
    fn test_provider_config_get_value() {
        let provider_config = ProviderConfig::Simple({
            let mut map = HashMap::new();
            map.insert("api_key".to_string(), serde_json::Value::String("test-key".to_string()));
            map.insert("model".to_string(), serde_json::Value::String("test-model".to_string()));
            map.insert("timeout".to_string(), serde_json::Value::Number(serde_json::Number::from(30)));
            map.insert("enabled".to_string(), serde_json::Value::Bool(true));
            map
        });
        
        assert_eq!(provider_config.get_api_key(), Some("test-key".to_string()));
        assert_eq!(provider_config.get_value("model").unwrap().as_str(), Some("test-model"));
        assert_eq!(provider_config.get_value("timeout").unwrap().as_u64(), Some(30));
        assert_eq!(provider_config.get_value("enabled").unwrap().as_bool(), Some(true));
        assert!(provider_config.get_value("nonexistent").is_none());
    }

    #[test]
    fn test_provider_config_no_api_key() {
        let provider_config = ProviderConfig::Simple({
            let mut map = HashMap::new();
            map.insert("model".to_string(), serde_json::Value::String("test-model".to_string()));
            map
        });
        
        assert!(provider_config.get_api_key().is_none());
    }

    #[test]
    fn test_config_from_env_vars() {
        // This test would need to set environment variables
        // For now, just test that the function exists and returns a config builder
        let config_builder = config_from_env();
        assert!(config_builder.build().and_then(|c| c.try_deserialize::<DebabelizerConfig>()).is_ok());
    }

    #[test]
    fn test_complex_config_scenario() {
        let mut providers = HashMap::new();
        
        // Add multiple providers with different configurations
        providers.insert(
            "soniox".to_string(),
            ProviderConfig::Simple({
                let mut map = HashMap::new();
                map.insert("api_key".to_string(), serde_json::Value::String("soniox-key".to_string()));
                map.insert("model".to_string(), serde_json::Value::String("en".to_string()));
                map.insert("auto_detect_language".to_string(), serde_json::Value::Bool(false));
                map
            }),
        );
        
        providers.insert(
            "elevenlabs".to_string(),
            ProviderConfig::Simple({
                let mut map = HashMap::new();
                map.insert("api_key".to_string(), serde_json::Value::String("elevenlabs-key".to_string()));
                map.insert("model".to_string(), serde_json::Value::String("eleven_monolingual_v1".to_string()));
                map.insert("voice_id".to_string(), serde_json::Value::String("21m00Tcm4TlvDq8ikWAM".to_string()));
                map
            }),
        );
        
        let preferences = Preferences {
            stt_provider: Some("soniox".to_string()),
            tts_provider: Some("elevenlabs".to_string()),
            auto_select: false,
            optimize_for: OptimizationStrategy::Latency,
            timeout_seconds: Some(60),
            max_retries: Some(2),
        };
        
        let config = DebabelizerConfig {
            preferences,
            providers,
        };
        
        // Test all configuration aspects
        assert_eq!(config.get_preferred_stt_provider(), Some("soniox"));
        assert_eq!(config.get_preferred_tts_provider(), Some("elevenlabs"));
        assert!(!config.is_auto_select_enabled());
        assert_eq!(config.get_optimization_strategy(), OptimizationStrategy::Latency);
        assert_eq!(config.get_timeout_seconds(), Some(60));
        assert_eq!(config.get_max_retries(), Some(2));
        
        // Test provider configurations
        let soniox_config = config.get_provider_config("soniox").unwrap();
        assert_eq!(soniox_config.get_api_key(), Some("soniox-key".to_string()));
        assert_eq!(soniox_config.get_value("model").unwrap().as_str(), Some("en"));
        assert_eq!(soniox_config.get_value("auto_detect_language").unwrap().as_bool(), Some(false));
        
        let elevenlabs_config = config.get_provider_config("elevenlabs").unwrap();
        assert_eq!(elevenlabs_config.get_api_key(), Some("elevenlabs-key".to_string()));
        assert_eq!(elevenlabs_config.get_value("voice_id").unwrap().as_str(), Some("21m00Tcm4TlvDq8ikWAM"));
    }
}