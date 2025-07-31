"""
Configuration management for Debabelizer
"""

from typing import Dict, Any, Optional, List
import os


class DebabelizerConfig:
    """Configuration manager for Debabelizer"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration
        
        Args:
            config: Configuration dictionary or None to use environment variables
        """
        self.config = config or {}
        self._load_from_environment()
        
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # STT Provider configurations
        env_configs = {
            # Soniox
            "soniox": {
                "api_key": os.getenv("SONIOX_API_KEY"),
                "model": os.getenv("SONIOX_MODEL", "stt-rt-preview"),
            },
            
            # Deepgram
            "deepgram": {
                "api_key": os.getenv("DEEPGRAM_API_KEY"),
                "model": os.getenv("DEEPGRAM_MODEL", "nova-2"),
                "language": os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
            },
            
            # ElevenLabs
            "elevenlabs": {
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "default_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
                "model": os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
                "optimize_streaming_latency": int(os.getenv("ELEVENLABS_OPTIMIZE_STREAMING_LATENCY", "1")),
            },
            
            # Azure
            "azure": {
                "api_key": os.getenv("AZURE_SPEECH_KEY"),
                "region": os.getenv("AZURE_SPEECH_REGION"),
            },
            
            # OpenAI
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "whisper-1"),
                "tts_model": os.getenv("OPENAI_TTS_MODEL", "tts-1"),
                "tts_voice": os.getenv("OPENAI_TTS_VOICE", "alloy"),
            },
            
            # OpenAI Whisper API (uses same key as OpenAI)
            "openai_whisper": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_WHISPER_MODEL", "whisper-1"),
                "temperature": float(os.getenv("OPENAI_WHISPER_TEMPERATURE", "0.0")),
                "response_format": os.getenv("OPENAI_WHISPER_RESPONSE_FORMAT", "json"),
            },
            
            # Google Cloud
            "google": {
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
                "stt_model": os.getenv("GOOGLE_STT_MODEL", "latest_long"),
                "tts_voice": os.getenv("GOOGLE_TTS_VOICE", "en-US-Standard-A"),
            },
        }
        
        # General settings
        general_settings = {
            "preferences": {
                "stt_provider": os.getenv("DEBABELIZER_STT_PROVIDER"),  # Preferred STT provider
                "tts_provider": os.getenv("DEBABELIZER_TTS_PROVIDER"),  # Preferred TTS provider
                "auto_select": os.getenv("DEBABELIZER_AUTO_SELECT", "false").lower() == "true",
                "optimize_for": os.getenv("DEBABELIZER_OPTIMIZE_FOR", "balanced"),  # cost, latency, quality, balanced
            }
        }
        
        # Merge general settings
        for section, settings in general_settings.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in settings.items():
                if key not in self.config[section] and value is not None:
                    self.config[section][key] = value
        
        # Merge provider configs (provided config takes precedence)
        for provider, provider_config in env_configs.items():
            if provider not in self.config:
                self.config[provider] = {}
                
            for key, value in provider_config.items():
                if key not in self.config[provider] and value is not None:
                    self.config[provider][key] = value
                    
    def get_provider_config(self, provider: str, key: Optional[str] = None) -> Any:
        """
        Get configuration for a specific provider
        
        Args:
            provider: Provider name (e.g., 'soniox', 'elevenlabs')
            key: Specific config key (optional)
            
        Returns:
            Configuration value or dictionary
        """
        provider_config = self.config.get(provider, {})
        
        if key is None:
            return provider_config
            
        return provider_config.get(key)
        
    def set_provider_config(self, provider: str, key: str, value: Any) -> None:
        """
        Set configuration for a specific provider
        
        Args:
            provider: Provider name
            key: Configuration key
            value: Configuration value
        """
        if provider not in self.config:
            self.config[provider] = {}
            
        self.config[provider][key] = value
        
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        return self.get_provider_config(provider, "api_key")
        
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider"""
        self.set_provider_config(provider, "api_key", api_key)
        
    def is_provider_configured(self, provider: str) -> bool:
        """Check if a provider is properly configured"""
        # Special handling for Google Cloud providers
        if provider == "google":
            # Check for credentials file in environment
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                return True
            # Check for credentials_path in config
            provider_config = self.get_provider_config(provider)
            if provider_config.get("credentials_path"):
                return True
            # Fall back to api_key check
        
        # Special handling for Whisper (no API key required - local processing)
        if provider == "whisper":
            return True  # Always considered configured since it's local
        
        api_key = self.get_api_key(provider)
        return bool(api_key and api_key != "NOT_SET")
        
    def get_configured_providers(self) -> Dict[str, List[str]]:
        """Get list of configured providers by type"""
        stt_providers = []
        tts_providers = []
        
        # Define provider types
        stt_provider_names = ["soniox", "deepgram", "openai", "openai_whisper", "azure", "google", "whisper"]
        tts_provider_names = ["elevenlabs", "azure", "openai", "google"]
        
        for provider in stt_provider_names:
            if self.is_provider_configured(provider):
                stt_providers.append(provider)
                
        for provider in tts_provider_names:
            if self.is_provider_configured(provider):
                tts_providers.append(provider)
                
        return {
            "stt": stt_providers,
            "tts": tts_providers
        }
    
    def get_preferred_stt_provider(self) -> Optional[str]:
        """Get user's preferred STT provider"""
        return self.config.get("preferences", {}).get("stt_provider")
    
    def get_preferred_tts_provider(self) -> Optional[str]:
        """Get user's preferred TTS provider"""
        return self.config.get("preferences", {}).get("tts_provider")
    
    def should_auto_select(self) -> bool:
        """Check if auto-selection is enabled"""
        return self.config.get("preferences", {}).get("auto_select", False)
    
    def get_optimization_strategy(self) -> str:
        """Get optimization strategy (cost, latency, quality, balanced)"""
        return self.config.get("preferences", {}).get("optimize_for", "balanced")
    
    def set_preferred_stt_provider(self, provider: str) -> None:
        """Set preferred STT provider"""
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"]["stt_provider"] = provider
    
    def set_preferred_tts_provider(self, provider: str) -> None:
        """Set preferred TTS provider"""
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"]["tts_provider"] = provider
    
    def set_auto_select(self, enabled: bool) -> None:
        """Enable or disable auto-selection"""
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"]["auto_select"] = enabled
    
    def set_optimization_strategy(self, strategy: str) -> None:
        """Set optimization strategy"""
        valid_strategies = ["cost", "latency", "quality", "balanced"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid optimization strategy: {strategy}. Must be one of {valid_strategies}")
        
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"]["optimize_for"] = strategy
        
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self.config.copy()
        
    def __repr__(self) -> str:
        """String representation (hide API keys)"""
        safe_config = {}
        
        for provider, provider_config in self.config.items():
            safe_provider_config = {}
            for key, value in provider_config.items():
                if "key" in key.lower() or "secret" in key.lower():
                    safe_provider_config[key] = "***" if value else None
                else:
                    safe_provider_config[key] = value
            safe_config[provider] = safe_provider_config
            
        return f"DebabelizerConfig({safe_config})"