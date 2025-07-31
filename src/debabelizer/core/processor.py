"""
Main VoiceProcessor - The heart of Debabelizer

This class orchestrates STT and TTS providers, handles provider switching,
cost optimization, and provides a unified interface for voice processing.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, AsyncGenerator, Union
from datetime import datetime
from pathlib import Path

from ..providers.base import (
    STTProvider, TTSProvider, TranscriptionResult, SynthesisResult, 
    StreamingResult, Voice, AudioFormat
)
from ..providers.base.exceptions import ProviderError
from .config import DebabelizerConfig
from .session import SessionManager
from ..utils.audio import AudioConverter
from ..utils.formats import detect_audio_format

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Universal Voice Processing Engine
    
    The main interface for all voice processing operations in Debabelizer.
    Supports pluggable STT/TTS providers, cost optimization, and intelligent
    provider selection.
    """
    
    def __init__(
        self,
        stt_provider: Optional[Union[str, STTProvider]] = None,
        tts_provider: Optional[Union[str, TTSProvider]] = None,
        config: Optional[Union[Dict[str, Any], DebabelizerConfig]] = None
    ):
        """
        Initialize VoiceProcessor
        
        Args:
            stt_provider: STT provider name or instance (overrides config preferences)
            tts_provider: TTS provider name or instance (overrides config preferences)
            config: Configuration dictionary or DebabelizerConfig instance
        """
        if isinstance(config, DebabelizerConfig):
            self.config = config
        else:
            self.config = DebabelizerConfig(config or {})
        
        # Provider instances
        self._stt_provider: Optional[STTProvider] = None
        self._tts_provider: Optional[TTSProvider] = None
        
        # Session management
        self.session_manager = SessionManager()
        
        # Audio utilities
        self.audio_converter = AudioConverter()
        
        # Usage tracking
        self.usage_stats = {
            "stt_requests": 0,
            "tts_requests": 0,
            "stt_duration": 0.0,
            "tts_characters": 0,
            "cost_estimate": 0.0,
            "sessions_created": 0
        }
        
        # Determine which providers to use based on user preferences
        self._stt_provider_spec = self._determine_stt_provider(stt_provider)
        self._tts_provider_spec = self._determine_tts_provider(tts_provider)
        
        # Validate provider configurations if specific providers are requested
        if self._stt_provider_spec and isinstance(self._stt_provider_spec, str):
            self._validate_stt_provider_config(self._stt_provider_spec)
        if self._tts_provider_spec and isinstance(self._tts_provider_spec, str):
            self._validate_tts_provider_config(self._tts_provider_spec)
            
    async def _initialize_stt_provider(self, provider: Union[str, STTProvider]):
        """Initialize STT provider"""
        if isinstance(provider, STTProvider):
            self._stt_provider = provider
        else:
            self._stt_provider = await self._create_stt_provider(provider)
        
        # Clear the spec since provider is now initialized
        self._stt_provider_spec = None
            
    async def _initialize_tts_provider(self, provider: Union[str, TTSProvider]):
        """Initialize TTS provider"""
        if isinstance(provider, TTSProvider):
            self._tts_provider = provider
        else:
            self._tts_provider = await self._create_tts_provider(provider)
        
        # Clear the spec since provider is now initialized
        self._tts_provider_spec = None
    
    def _validate_stt_provider_config(self, provider_name: str) -> None:
        """Validate that STT provider is properly configured"""
        valid_providers = ["soniox", "deepgram", "openai", "openai_whisper", "azure", "google", "whisper"]
        
        if provider_name not in valid_providers:
            raise ProviderError(f"STT provider '{provider_name}' is not configured or not supported", provider_name)
        
        if not self.config.is_provider_configured(provider_name):
            raise ProviderError(f"STT provider '{provider_name}' is not configured or missing API key", provider_name)
    
    def _validate_tts_provider_config(self, provider_name: str) -> None:
        """Validate that TTS provider is properly configured"""
        valid_providers = ["elevenlabs", "azure", "openai", "google"]
        
        if provider_name not in valid_providers:
            raise ProviderError(f"TTS provider '{provider_name}' is not configured or not supported", provider_name)
        
        if not self.config.is_provider_configured(provider_name):
            raise ProviderError(f"TTS provider '{provider_name}' is not configured or missing API key", provider_name)
    
    def _determine_stt_provider(self, requested_provider: Optional[Union[str, STTProvider]]) -> Optional[Union[str, STTProvider]]:
        """Determine which STT provider to use based on user preferences"""
        # If a specific provider is requested, use that
        if requested_provider:
            return requested_provider
        
        # Check if user has a preferred provider configured
        preferred = self.config.get_preferred_stt_provider()
        if preferred and self.config.is_provider_configured(preferred):
            return preferred
        
        # If auto-selection is enabled, return None to trigger auto-selection later
        if self.config.should_auto_select():
            return None
        
        # Otherwise, try to find any configured provider
        configured_providers = self.config.get_configured_providers()["stt"]
        if configured_providers:
            return configured_providers[0]  # Use first available
        
        return None
    
    def _determine_tts_provider(self, requested_provider: Optional[Union[str, TTSProvider]]) -> Optional[Union[str, TTSProvider]]:
        """Determine which TTS provider to use based on user preferences"""
        # If a specific provider is requested, use that
        if requested_provider:
            return requested_provider
        
        # Check if user has a preferred provider configured
        preferred = self.config.get_preferred_tts_provider()
        if preferred and self.config.is_provider_configured(preferred):
            return preferred
        
        # If auto-selection is enabled, return None to trigger auto-selection later
        if self.config.should_auto_select():
            return None
        
        # Otherwise, try to find any configured provider
        configured_providers = self.config.get_configured_providers()["tts"]
        if configured_providers:
            return configured_providers[0]  # Use first available
        
        return None
            
    async def _create_stt_provider(self, provider_name: str) -> STTProvider:
        """Create STT provider instance"""
        from ..providers.stt.soniox import SonioxSTTProvider
        
        provider_classes = {
            "soniox": SonioxSTTProvider,
        }
        
        # Dynamically import providers to avoid dependency issues
        if provider_name == "deepgram":
            try:
                from ..providers.stt.deepgram import DeepgramSTTProvider
                provider_classes["deepgram"] = DeepgramSTTProvider
            except ImportError as e:
                raise ProviderError(f"Deepgram provider not available: {e}", provider_name)
        
        if provider_name == "google":
            try:
                from ..providers.stt.google import GoogleSTTProvider
                provider_classes["google"] = GoogleSTTProvider
            except ImportError as e:
                raise ProviderError(f"Google Cloud STT provider not available: {e}", provider_name)
        
        if provider_name == "azure":
            try:
                from ..providers.stt.azure import AzureSTTProvider
                provider_classes["azure"] = AzureSTTProvider
            except ImportError as e:
                raise ProviderError(f"Azure STT provider not available: {e}", provider_name)
        
        if provider_name == "whisper":
            try:
                from ..providers.stt.whisper import WhisperSTTProvider
                provider_classes["whisper"] = WhisperSTTProvider
            except ImportError as e:
                raise ProviderError(f"Whisper STT provider not available: {e}", provider_name)
        
        if provider_name == "openai_whisper":
            try:
                from ..providers.stt.openai_whisper import OpenAIWhisperSTTProvider
                provider_classes["openai_whisper"] = OpenAIWhisperSTTProvider
            except ImportError as e:
                raise ProviderError(f"OpenAI Whisper API provider not available: {e}", provider_name)
        
        if provider_name not in provider_classes:
            raise ProviderError(f"Unknown STT provider: {provider_name}")
            
        provider_class = provider_classes[provider_name]
        provider_config = self.config.get_provider_config(provider_name)
        
        # Special handling for Google Cloud provider
        if provider_name == "google":
            # Google uses credentials_path instead of api_key
            credentials_path = provider_config.get("credentials_path")
            if credentials_path:
                # Remove credentials_path from provider_config to avoid duplicate parameter error
                provider_config_clean = {k: v for k, v in provider_config.items() if k != "credentials_path"}
                return provider_class(credentials_path=credentials_path, **provider_config_clean)
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                return provider_class(**provider_config)
            else:
                raise ProviderError(f"Google Cloud credentials not configured for {provider_name}")
        
        # Special handling for Azure provider (requires region)
        if provider_name == "azure":
            api_key = self.config.get_provider_config(provider_name, "api_key")
            region = provider_config.get("region", "eastus")
            if not api_key:
                raise ProviderError(f"API key not configured for {provider_name}")
            # Remove api_key and region from provider_config to avoid duplicate parameter error
            provider_config_clean = {k: v for k, v in provider_config.items() if k not in ["api_key", "region"]}
            return provider_class(api_key, region=region, **provider_config_clean)
        
        # Special handling for Whisper (no API key required - local processing)
        if provider_name == "whisper":
            return provider_class(**provider_config)
        
        # Special handling for OpenAI Whisper (uses OpenAI API key)
        if provider_name == "openai_whisper":
            api_key = self.config.get_provider_config(provider_name, "api_key")
            if not api_key:
                # Fallback to standard openai API key if openai_whisper key not found
                api_key = self.config.get_provider_config("openai", "api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError(f"OpenAI API key not configured for {provider_name}")
            # Remove api_key from provider_config to avoid duplicate parameter error
            provider_config_clean = {k: v for k, v in provider_config.items() if k != "api_key"}
            return provider_class(api_key, **provider_config_clean)
        
        # Special handling for Soniox (expects api_key as keyword argument)
        if provider_name == "soniox":
            api_key = self.config.get_provider_config(provider_name, "api_key")
            if not api_key:
                raise ProviderError(f"API key not configured for {provider_name}")
            # Remove api_key from provider_config to avoid duplicate parameter error
            provider_config_clean = {k: v for k, v in provider_config.items() if k != "api_key"}
            return provider_class(api_key=api_key, **provider_config_clean)
        
        # Standard API key providers
        api_key = self.config.get_provider_config(provider_name, "api_key")
        if not api_key:
            raise ProviderError(f"API key not configured for {provider_name}")
        
        # Remove api_key from provider_config to avoid duplicate parameter error
        provider_config_clean = {k: v for k, v in provider_config.items() if k != "api_key"}
        return provider_class(api_key, **provider_config_clean)
        
    async def _create_tts_provider(self, provider_name: str) -> TTSProvider:
        """Create TTS provider instance"""
        from ..providers.tts.elevenlabs import ElevenLabsTTSProvider
        
        provider_classes = {
            "elevenlabs": ElevenLabsTTSProvider,
        }
        
        # Dynamically import providers to avoid dependency issues
        if provider_name == "openai":
            try:
                from ..providers.tts.openai import OpenAITTSProvider
                provider_classes["openai"] = OpenAITTSProvider
            except ImportError as e:
                raise ProviderError(f"OpenAI provider not available: {e}", provider_name)
        
        if provider_name == "google":
            try:
                from ..providers.tts.google import GoogleTTSProvider
                provider_classes["google"] = GoogleTTSProvider
            except ImportError as e:
                raise ProviderError(f"Google Cloud TTS provider not available: {e}", provider_name)
        
        if provider_name == "azure":
            try:
                from ..providers.tts.azure import AzureTTSProvider
                provider_classes["azure"] = AzureTTSProvider
            except ImportError as e:
                raise ProviderError(f"Azure TTS provider not available: {e}", provider_name)
        
        if provider_name not in provider_classes:
            raise ProviderError(f"Unknown TTS provider: {provider_name}")
            
        provider_class = provider_classes[provider_name]
        provider_config = self.config.get_provider_config(provider_name)
        
        # Special handling for Google Cloud provider
        if provider_name == "google":
            # Google uses credentials_path instead of api_key
            credentials_path = provider_config.get("credentials_path")
            if credentials_path:
                # Remove credentials_path from provider_config to avoid duplicate parameter error
                provider_config_clean = {k: v for k, v in provider_config.items() if k != "credentials_path"}
                return provider_class(credentials_path=credentials_path, **provider_config_clean)
            elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                return provider_class(**provider_config)
            else:
                raise ProviderError(f"Google Cloud credentials not configured for {provider_name}")
        
        # Special handling for Azure provider (requires region)
        if provider_name == "azure":
            api_key = self.config.get_provider_config(provider_name, "api_key")
            region = provider_config.get("region", "eastus")
            if not api_key:
                raise ProviderError(f"API key not configured for {provider_name}")
            # Remove api_key and region from provider_config to avoid duplicate parameter error
            provider_config_clean = {k: v for k, v in provider_config.items() if k not in ["api_key", "region"]}
            return provider_class(api_key, region=region, **provider_config_clean)
        
        # Standard API key providers
        api_key = self.config.get_provider_config(provider_name, "api_key")
        if not api_key:
            raise ProviderError(f"API key not configured for {provider_name}")
        
        # Remove api_key from provider_config to avoid duplicate parameter error
        provider_config_clean = {k: v for k, v in provider_config.items() if k != "api_key"}
        return provider_class(api_key, **provider_config_clean)
        
    # STT Methods
    
    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to audio file
            language: Expected language (None for auto-detection)
            language_hints: Language hints for better detection
            **kwargs: Provider-specific options
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        if not self._stt_provider:
            await self._auto_select_stt_provider()
        
        # Use _get_stt_provider for potential test mocking
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        logger.info(f"Transcribing file: {file_path} with {provider.name}")
        
        # Auto-detect format if not specified
        if "audio_format" not in kwargs:
            kwargs["audio_format"] = detect_audio_format(file_path)
            
        start_time = datetime.now()
        result = await provider.transcribe_file(
            str(file_path), language, language_hints, **kwargs
        )
        
        # Update usage stats
        duration = (datetime.now() - start_time).total_seconds()
        self.usage_stats["stt_requests"] += 1
        self.usage_stats["stt_duration"] += duration
        self.usage_stats["cost_estimate"] += provider.get_cost_estimate(duration)
        
        return result
        
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe raw audio data
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format
            sample_rate: Sample rate in Hz
            language: Expected language
            language_hints: Language hints
            **kwargs: Provider-specific options
            
        Returns:
            TranscriptionResult
        """
        if not self._stt_provider:
            await self._auto_select_stt_provider()
        
        # Use _get_stt_provider for potential test mocking
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        logger.info(f"Transcribing {len(audio_data)} bytes of {audio_format} audio with {provider.name}")
        
        start_time = datetime.now()
        result = await provider.transcribe_audio(
            audio_data, audio_format, sample_rate, language, language_hints, **kwargs
        )
        
        # Update usage stats
        duration = (datetime.now() - start_time).total_seconds()
        self.usage_stats["stt_requests"] += 1
        self.usage_stats["stt_duration"] += duration
        self.usage_stats["cost_estimate"] += provider.get_cost_estimate(duration)
        
        return result
        
    async def start_streaming_transcription(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Start a streaming transcription session
        
        Args:
            audio_format: Audio format for streaming
            sample_rate: Sample rate in Hz
            language: Expected language
            language_hints: Language hints
            **kwargs: Provider-specific options
            
        Returns:
            Session ID
        """
        if not self._stt_provider:
            await self._auto_select_stt_provider()
        
        # Use _get_stt_provider for potential test mocking
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        if not provider.supports_streaming:
            raise ProviderError(f"Provider {provider.name} does not support streaming")
            
        session_id = await provider.start_streaming(
            audio_format, sample_rate, language, language_hints, **kwargs
        )
        
        # Track session
        self.session_manager.create_session(session_id, "stt", provider.name)
        self.usage_stats["sessions_created"] += 1
        
        logger.info(f"Started streaming transcription session: {session_id}")
        return session_id
        
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming session"""
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        await provider.stream_audio(session_id, audio_chunk)
        
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results"""
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        async for result in provider.get_streaming_results(session_id):
            yield result
            
    async def stop_streaming_transcription(self, session_id: str) -> None:
        """Stop streaming transcription session"""
        provider = self._get_stt_provider()
        if provider:
            await provider.stop_streaming(session_id)
        self.session_manager.end_session(session_id)
        
    async def transcribe_chunk(
        self,
        audio_data: bytes,
        audio_format: str = "webm",
        sample_rate: int = 48000,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio chunk using file API (fake streaming approach).
        This method is optimized for processing buffered audio chunks from browsers.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (defaults to webm for browser compatibility)
            sample_rate: Sample rate in Hz (defaults to 48000 for WebM)
            language: Expected language
            **kwargs: Provider-specific options
            
        Returns:
            TranscriptionResult
        """
        if not self._stt_provider:
            await self._auto_select_stt_provider()
        
        # Use _get_stt_provider for potential test mocking
        provider = self._get_stt_provider()
        if not provider:
            raise ProviderError("No STT provider available")
            
        # Check if provider supports chunk transcription
        if not hasattr(provider, 'transcribe_chunk'):
            # Fallback to regular transcribe_audio if chunk method not available
            logger.warning(f"Provider {provider.name} doesn't support transcribe_chunk, falling back to transcribe_audio")
            return await self.transcribe_audio(
                audio_data, audio_format, sample_rate, language, **kwargs
            )
            
        logger.info(f"Transcribing {len(audio_data)} bytes of {audio_format} chunk with {provider.name}")
        
        start_time = datetime.now()
        result = await provider.transcribe_chunk(
            audio_data, audio_format, sample_rate, language, **kwargs
        )
        
        # Update usage stats
        duration = (datetime.now() - start_time).total_seconds()
        self.usage_stats["stt_requests"] += 1
        self.usage_stats["stt_duration"] += duration
        self.usage_stats["cost_estimate"] += provider.get_cost_estimate(duration)
        
        return result
        
    # TTS Methods
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[Union[Voice, str]] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice object or voice ID
            audio_format: Desired output format
            sample_rate: Sample rate in Hz
            **kwargs: Provider-specific options
            
        Returns:
            SynthesisResult with audio data
        """
        if not self._tts_provider:
            await self._auto_select_tts_provider()
            
        logger.info(f"Synthesizing {len(text)} characters with {self._tts_provider.name}")
        
        # Handle voice parameter
        voice_obj = None
        voice_id = None
        
        if isinstance(voice, Voice):
            voice_obj = voice
        elif isinstance(voice, str):
            voice_id = voice
            
        start_time = datetime.now()
        
        # Choose synthesis method based on text length
        if len(text) > 5000:  # Threshold for long text
            result = await self._tts_provider.synthesize_long_text(
                text, voice_obj, voice_id, audio_format, sample_rate, **kwargs
            )
        else:
            result = await self._tts_provider.synthesize(
                text, voice_obj, voice_id, audio_format, sample_rate, **kwargs
            )
            
        # Update usage stats
        duration = (datetime.now() - start_time).total_seconds()
        self.usage_stats["tts_requests"] += 1
        self.usage_stats["tts_characters"] += len(text)
        self.usage_stats["cost_estimate"] += self._tts_provider.get_cost_estimate(text)
        
        return result
        
    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[Union[Voice, str]] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis results in real-time"""
        if not self._tts_provider:
            await self._auto_select_tts_provider()
            
        if not self._tts_provider.supports_streaming:
            raise ProviderError(f"Provider {self._tts_provider.name} does not support streaming")
            
        voice_obj = None
        voice_id = None
        
        if isinstance(voice, Voice):
            voice_obj = voice
        elif isinstance(voice, str):
            voice_id = voice
            
        async for chunk in self._tts_provider.synthesize_streaming(
            text, voice_obj, voice_id, audio_format, sample_rate, **kwargs
        ):
            yield chunk
            
    async def get_available_voices(
        self, 
        language: Optional[str] = None
    ) -> List[Voice]:
        """Get available voices from TTS provider"""
        if not self._tts_provider:
            await self._auto_select_tts_provider()
            
        return await self._tts_provider.get_available_voices(language)
        
    # Provider Management
    
    async def set_stt_provider(self, provider: Union[str, STTProvider]) -> None:
        """Switch STT provider"""
        if self._stt_provider:
            await self._stt_provider.cleanup()
            
        await self._initialize_stt_provider(provider)
        logger.info(f"Switched to STT provider: {self._stt_provider.name}")
        
    async def set_tts_provider(self, provider: Union[str, TTSProvider]) -> None:
        """Switch TTS provider"""
        if self._tts_provider:
            await self._tts_provider.cleanup()
            
        await self._initialize_tts_provider(provider)
        logger.info(f"Switched to TTS provider: {self._tts_provider.name}")
        
    async def _auto_select_stt_provider(self) -> None:
        """Auto-select best STT provider based on optimization strategy"""
        # If a specific provider was requested during init, use that
        if self._stt_provider_spec:
            await self._initialize_stt_provider(self._stt_provider_spec)
            return
            
        # Get configured providers for auto-selection
        configured_providers = self.config.get_configured_providers()["stt"]
        if not configured_providers:
            raise ProviderError("No STT providers are configured")
            
        # Auto-select based on optimization strategy from config
        strategy = self.config.get_optimization_strategy()
        selected_provider = self._select_best_stt_provider(configured_providers, strategy)
        await self.set_stt_provider(selected_provider)
            
    async def _auto_select_tts_provider(self) -> None:
        """Auto-select best TTS provider based on optimization strategy"""
        # If a specific provider was requested during init, use that
        if self._tts_provider_spec:
            await self._initialize_tts_provider(self._tts_provider_spec)
            return
            
        # Get configured providers for auto-selection
        configured_providers = self.config.get_configured_providers()["tts"]
        if not configured_providers:
            raise ProviderError("No TTS providers are configured")
            
        # Auto-select based on optimization strategy from config
        strategy = self.config.get_optimization_strategy()
        selected_provider = self._select_best_tts_provider(configured_providers, strategy)
        await self.set_tts_provider(selected_provider)
    
    def _select_best_stt_provider(self, available_providers: List[str], strategy: str) -> str:
        """Select best STT provider based on optimization strategy"""
        # Define provider rankings for different strategies
        provider_rankings = {
            "cost": ["soniox", "deepgram", "azure", "google", "openai_whisper", "openai"],  # Cheapest first
            "latency": ["soniox", "deepgram", "google", "azure", "openai_whisper", "openai"],  # Fastest first  
            "quality": ["openai_whisper", "openai", "google", "deepgram", "soniox", "azure"],  # Best quality first
            "balanced": ["deepgram", "soniox", "openai_whisper", "google", "openai", "azure"]  # Balanced approach
        }
        
        ranking = provider_rankings.get(strategy, provider_rankings["balanced"])
        
        # Return first available provider in preference order
        for provider in ranking:
            if provider in available_providers:
                return provider
        
        # Fallback to first available
        return available_providers[0]
    
    def _select_best_tts_provider(self, available_providers: List[str], strategy: str) -> str:
        """Select best TTS provider based on optimization strategy"""
        # Define provider rankings for different strategies
        provider_rankings = {
            "cost": ["azure", "google", "openai", "elevenlabs"],  # Cheapest first
            "latency": ["elevenlabs", "google", "azure", "openai"],  # Fastest first
            "quality": ["elevenlabs", "google", "azure", "openai"],  # Best quality first
            "balanced": ["elevenlabs", "google", "azure", "openai"]  # Balanced approach
        }
        
        ranking = provider_rankings.get(strategy, provider_rankings["balanced"])
        
        # Return first available provider in preference order
        for provider in ranking:
            if provider in available_providers:
                return provider
        
        # Fallback to first available
        return available_providers[0]
            
    # Utility Methods
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.usage_stats.copy()
        
    def reset_usage_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_stats = {
            "stt_requests": 0,
            "tts_requests": 0,
            "stt_duration": 0.0,
            "tts_characters": 0,
            "cost_estimate": 0.0,
            "sessions_created": 0
        }
        
    async def test_providers(self) -> Dict[str, bool]:
        """Test connectivity to all configured providers"""
        results = {}
        
        if self._stt_provider:
            results[f"stt_{self._stt_provider.name}"] = await self._stt_provider.test_connection()
            
        if self._tts_provider:
            results[f"tts_{self._tts_provider.name}"] = await self._tts_provider.test_connection()
            
        return results
    
    # Properties
    
    @property
    def stt_provider(self) -> Optional[STTProvider]:
        """Get current STT provider instance"""
        return self._stt_provider
    
    def _get_stt_provider(self) -> Optional[STTProvider]:
        """Get STT provider (for testing purposes)"""
        return self._stt_provider
    
    @property
    def tts_provider(self) -> Optional[TTSProvider]:
        """Get current TTS provider instance"""
        return self._tts_provider
    
    @property
    def stt_provider_name(self) -> Optional[str]:
        """Get current STT provider name"""
        if self._stt_provider:
            return self._stt_provider.name
        elif self._stt_provider_spec:
            if isinstance(self._stt_provider_spec, str):
                return self._stt_provider_spec
            else:
                return self._stt_provider_spec.name
        return None
    
    @property
    def tts_provider_name(self) -> Optional[str]:
        """Get current TTS provider name"""
        if self._tts_provider:
            return self._tts_provider.name
        elif self._tts_provider_spec:
            if isinstance(self._tts_provider_spec, str):
                return self._tts_provider_spec
            else:
                return self._tts_provider_spec.name
        return None
        
    async def cleanup(self) -> None:
        """Cleanup all resources"""
        if self._stt_provider:
            await self._stt_provider.cleanup()
            
        if self._tts_provider:
            await self._tts_provider.cleanup()
            
        await self.session_manager.cleanup_all_sessions()
        logger.info("VoiceProcessor cleanup completed")