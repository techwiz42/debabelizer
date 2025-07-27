"""
Unit tests for Debabelizer configuration management
"""

import pytest
import os
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer.core.config import DebabelizerConfig


class TestDebabelizerConfig:
    """Test cases for DebabelizerConfig class"""

    def test_empty_config_initialization(self):
        """Test initialization with no config"""
        config = DebabelizerConfig()
        assert isinstance(config.config, dict)

    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        custom_config = {
            "soniox": {
                "api_key": "test_key",
                "model": "custom_model"
            }
        }
        config = DebabelizerConfig(custom_config)
        assert config.get_api_key("soniox") == "test_key"
        assert config.get_provider_config("soniox", "model") == "custom_model"

    @patch.dict(os.environ, {
        "SONIOX_API_KEY": "env_soniox_key",
        "DEEPGRAM_API_KEY": "env_deepgram_key",
        "ELEVENLABS_API_KEY": "env_elevenlabs_key"
    })
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables"""
        config = DebabelizerConfig()
        
        assert config.get_api_key("soniox") == "env_soniox_key"
        assert config.get_api_key("deepgram") == "env_deepgram_key"
        assert config.get_api_key("elevenlabs") == "env_elevenlabs_key"

    def test_provider_config_operations(self):
        """Test setting and getting provider configurations"""
        config = DebabelizerConfig()
        
        # Set provider config
        config.set_provider_config("test_provider", "test_key", "test_value")
        
        # Get provider config
        assert config.get_provider_config("test_provider", "test_key") == "test_value"
        assert config.get_provider_config("test_provider") == {"test_key": "test_value"}

    def test_api_key_operations(self):
        """Test API key setting and getting"""
        config = DebabelizerConfig()
        
        # Set API key
        config.set_api_key("test_provider", "test_api_key")
        
        # Get API key
        assert config.get_api_key("test_provider") == "test_api_key"

    def test_provider_configured_check(self):
        """Test checking if providers are configured"""
        config = DebabelizerConfig()
        
        # Provider not configured
        assert not config.is_provider_configured("unconfigured_provider")
        
        # Configure provider
        config.set_api_key("test_provider", "valid_key")
        assert config.is_provider_configured("test_provider")
        
        # Test with "NOT_SET" key (should be False)
        config.set_api_key("invalid_provider", "NOT_SET")
        assert not config.is_provider_configured("invalid_provider")

    @patch.dict(os.environ, {
        "SONIOX_API_KEY": "soniox_key",
        "DEEPGRAM_API_KEY": "deepgram_key",
        "ELEVENLABS_API_KEY": "elevenlabs_key",
        "OPENAI_API_KEY": "openai_key"
    })
    def test_get_configured_providers(self):
        """Test getting list of configured providers"""
        config = DebabelizerConfig()
        providers = config.get_configured_providers()
        
        assert "stt" in providers
        assert "tts" in providers
        
        # Check STT providers
        stt_providers = providers["stt"]
        assert "soniox" in stt_providers
        assert "deepgram" in stt_providers
        assert "openai" in stt_providers
        
        # Check TTS providers  
        tts_providers = providers["tts"]
        assert "elevenlabs" in tts_providers
        assert "openai" in tts_providers

    def test_config_to_dict(self):
        """Test exporting configuration to dictionary"""
        custom_config = {
            "test_provider": {
                "api_key": "test_key",
                "model": "test_model"
            }
        }
        config = DebabelizerConfig(custom_config)
        exported = config.to_dict()
        
        assert exported == custom_config
        # Ensure it's a copy, not reference
        assert exported is not config.config

    def test_config_repr_hides_secrets(self):
        """Test that __repr__ hides API keys and secrets"""
        config = DebabelizerConfig()
        config.set_api_key("test_provider", "secret_key")
        config.set_provider_config("test_provider", "secret_token", "secret_value")
        config.set_provider_config("test_provider", "public_setting", "public_value")
        
        repr_str = repr(config)
        
        # Should hide secrets
        assert "secret_key" not in repr_str
        assert "secret_value" not in repr_str
        assert "***" in repr_str
        
        # Should show public values
        assert "public_value" in repr_str

    def test_provider_precedence(self):
        """Test that provided config takes precedence over environment"""
        with patch.dict(os.environ, {"SONIOX_API_KEY": "env_key"}):
            custom_config = {
                "soniox": {
                    "api_key": "custom_key"
                }
            }
            config = DebabelizerConfig(custom_config)
            
            # Custom config should take precedence
            assert config.get_api_key("soniox") == "custom_key"

    def test_default_model_values(self):
        """Test that default model values are set correctly"""
        config = DebabelizerConfig()
        
        # Check default models
        assert config.get_provider_config("soniox", "model") == "stt-rt-preview"
        assert config.get_provider_config("deepgram", "model") == "nova-2"
        assert config.get_provider_config("deepgram", "language") == "en-US"
        assert config.get_provider_config("elevenlabs", "model") == "eleven_turbo_v2_5"