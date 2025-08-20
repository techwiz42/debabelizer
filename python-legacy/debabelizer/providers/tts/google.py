"""
Google Cloud Text-to-Speech Provider

Implements TTS using Google Cloud Text-to-Speech API with support for:
- 220+ voices in 40+ languages and variants
- WaveNet, Neural2, and Standard voice types
- SSML support for advanced speech control
- Multiple audio formats (MP3, LINEAR16, OGG_OPUS, MULAW)
- Pitch, speaking rate, and volume gain control
- Audio profiles for different devices
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from datetime import datetime
import base64

from google.cloud import texttospeech_v1 as texttospeech
from google.api_core import exceptions as google_exceptions
import google.auth

from ..base import TTSProvider, SynthesisResult, Voice, AudioFormat
from ..base.exceptions import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class GoogleTTSProvider(TTSProvider):
    """
    Google Cloud Text-to-Speech Provider
    
    High-quality neural TTS with extensive language and voice options.
    """
    
    name = "google"
    
    # Language mapping for Google Cloud TTS
    LANGUAGE_MAP = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-BR",
        "ru": "ru-RU",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "ar": "ar-XA",  # Arabic with Latin script pronunciation
        "hi": "hi-IN",
        "nl": "nl-NL",
        "pl": "pl-PL",
        "tr": "tr-TR",
        "sv": "sv-SE",
        "da": "da-DK",
        "no": "nb-NO",  # Norwegian Bokmål
        "fi": "fi-FI",
        "cs": "cs-CZ",
        "hu": "hu-HU",
        "el": "el-GR",
        "he": "he-IL",
        "th": "th-TH",
        "vi": "vi-VN",
        "id": "id-ID",
        "ms": "ms-MY",
        "ro": "ro-RO",
        "uk": "uk-UA",
        "bg": "bg-BG",
        "hr": "hr-HR",
        "sk": "sk-SK",
        "sr": "sr-RS",
        "ta": "ta-IN",
        "te": "te-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "mr": "mr-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "pa": "pa-IN",
    }
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        voice_type: str = "Neural2",  # WaveNet, Neural2, or Standard
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        volume_gain_db: float = 0.0,
        audio_profile: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google Cloud Text-to-Speech Provider
        
        Args:
            credentials_path: Path to service account JSON file
            project_id: Google Cloud project ID
            voice_type: Voice technology (Neural2, WaveNet, Standard)
            speaking_rate: Speaking rate (0.25 to 4.0, default 1.0)
            pitch: Voice pitch (-20.0 to 20.0 semitones, default 0.0)
            volume_gain_db: Volume gain (-96.0 to 16.0 dB, default 0.0)
            audio_profile: Device profile for audio optimization
        """
        super().__init__()
        
        self.project_id = project_id
        self.voice_type = voice_type
        self.speaking_rate = max(0.25, min(4.0, speaking_rate))
        self.pitch = max(-20.0, min(20.0, pitch))
        self.volume_gain_db = max(-96.0, min(16.0, volume_gain_db))
        self.audio_profile = audio_profile
        
        # Initialize credentials
        if credentials_path:
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            # Initialize client
            self.client = texttospeech.TextToSpeechClient()
            
            # Get project ID if not provided
            if not self.project_id:
                _, self.project_id = google.auth.default()
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Google Cloud TTS client: {e}",
                self.name
            )
        
        # Cache available voices
        self._voices_cache = None
        self._voices_cache_time = None
        self._cache_duration = 3600  # 1 hour
        
        logger.info(f"Initialized Google Cloud TTS with voice type: {voice_type}")
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supports_streaming(self) -> bool:
        """Google Cloud TTS doesn't support real-time streaming"""
        return False
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice object (optional)
            voice_id: Voice ID string (e.g., "en-US-Neural2-A")
            audio_format: Desired output format
            sample_rate: Sample rate in Hz
            **kwargs: Additional Google-specific options
            
        Returns:
            SynthesisResult with audio data
        """
        try:
            # Determine voice name
            if voice:
                voice_name = voice.voice_id
                language_code = self._map_language_to_google(voice.language)
            elif voice_id:
                voice_name = voice_id
                # Extract language from voice ID (e.g., "en-US" from "en-US-Neural2-A")
                parts = voice_id.split("-")
                if len(parts) >= 2:
                    language_code = f"{parts[0]}-{parts[1]}"
                else:
                    language_code = "en-US"
            else:
                # Use default voice
                language_code = "en-US"
                voice_name = f"en-US-{self.voice_type}-A"
            
            # Build synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice_params = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            
            # Configure audio
            audio_config = self._build_audio_config(
                audio_format=audio_format,
                sample_rate=sample_rate,
                **kwargs
            )
            
            # Perform synthesis
            logger.info(f"Synthesizing {len(text)} characters with voice: {voice_name}")
            start_time = datetime.now()
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Get audio data
            audio_data = response.audio_content
            
            # Determine actual format
            if audio_format:
                format_str = audio_format.format
            else:
                format_str = kwargs.get("output_format", "mp3")
            
            # Calculate audio duration (approximate)
            audio_duration = self._estimate_audio_duration(
                len(audio_data),
                format_str,
                sample_rate
            )
            
            return SynthesisResult(
                audio_data=audio_data,
                format=format_str,
                sample_rate=sample_rate,
                duration=audio_duration,
                size_bytes=len(audio_data),
                metadata={
                    "voice": voice_name,
                    "language": language_code,
                    "synthesis_time": duration
                }
            )
            
        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(f"Google API error: {e}", self.name)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise ProviderError(f"Synthesis failed: {e}", self.name)
    
    async def synthesize_long_text(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize long text by splitting into chunks
        
        Google Cloud TTS has a 5000 character limit per request.
        """
        # Check if text exceeds limit
        if len(text) <= 5000:
            return await self.synthesize(
                text, voice, voice_id, audio_format, sample_rate, **kwargs
            )
        
        # Split text into chunks
        chunks = self._split_text(text, 4900)  # Leave some margin
        
        logger.info(f"Synthesizing long text in {len(chunks)} chunks")
        
        # Synthesize each chunk
        audio_parts = []
        total_duration = 0.0
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Synthesizing chunk {i+1}/{len(chunks)}")
            
            result = await self.synthesize(
                chunk, voice, voice_id, audio_format, sample_rate, **kwargs
            )
            
            audio_parts.append(result.audio_data)
            total_duration += result.duration
        
        # Combine audio
        combined_audio = b"".join(audio_parts)
        
        # Determine format
        if audio_format:
            format_str = audio_format.format
        else:
            format_str = kwargs.get("output_format", "mp3")
        
        return SynthesisResult(
            audio_data=combined_audio,
            format=format_str,
            sample_rate=sample_rate,
            duration=total_duration,
            size_bytes=len(combined_audio),
            metadata={
                "chunks": len(chunks),
                "voice": voice_id or (voice.voice_id if voice else None)
            }
        )
    
    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Google Cloud TTS doesn't support streaming - simulate it"""
        # Synthesize complete audio
        result = await self.synthesize(
            text, voice, voice_id, audio_format, sample_rate, **kwargs
        )
        
        # Stream in chunks
        chunk_size = 4096
        audio_data = result.audio_data
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            yield chunk
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)
    
    async def get_available_voices(
        self, 
        language: Optional[str] = None
    ) -> List[Voice]:
        """Get available voices, optionally filtered by language"""
        try:
            # Check cache
            import time
            current_time = time.time()
            
            if (self._voices_cache is not None and 
                self._voices_cache_time is not None and
                current_time - self._voices_cache_time < self._cache_duration):
                voices = self._voices_cache
            else:
                # Fetch voices from API
                logger.info("Fetching available voices from Google Cloud TTS")
                response = self.client.list_voices()
                
                voices = []
                for voice_info in response.voices:
                    # Create Voice object for each available voice
                    voice = Voice(
                        voice_id=voice_info.name,
                        name=voice_info.name,
                        language=self._map_google_language_to_standard(
                            voice_info.language_codes[0]
                        ),
                        description=f"{voice_info.ssml_gender.name} - {', '.join(voice_info.language_codes)}"
                    )
                    voices.append(voice)
                
                # Update cache
                self._voices_cache = voices
                self._voices_cache_time = current_time
            
            # Filter by language if requested
            if language:
                google_lang = self._map_language_to_google(language)
                filtered_voices = []
                for voice in voices:
                    # Check if voice supports the language
                    if google_lang in voice.voice_id:
                        filtered_voices.append(voice)
                return filtered_voices
            
            return voices
            
        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(f"Failed to get voices: {e}", self.name)
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            raise ProviderError(f"Failed to get voices: {e}", self.name)
    
    def _build_audio_config(
        self,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 22050,
        **kwargs
    ) -> texttospeech.AudioConfig:
        """Build Google TTS audio configuration"""
        # Map audio encoding
        if audio_format:
            format_str = audio_format.format
        else:
            format_str = kwargs.get("output_format", "mp3")
        
        encoding_map = {
            "mp3": texttospeech.AudioEncoding.MP3,
            "wav": texttospeech.AudioEncoding.LINEAR16,
            "ogg": texttospeech.AudioEncoding.OGG_OPUS,
            "opus": texttospeech.AudioEncoding.OGG_OPUS,
            "mulaw": texttospeech.AudioEncoding.MULAW,
            "alaw": texttospeech.AudioEncoding.ALAW,
        }
        
        encoding = encoding_map.get(
            format_str.lower(),
            texttospeech.AudioEncoding.MP3
        )
        
        # Build config
        config_dict = {
            "audio_encoding": encoding,
            "sample_rate_hertz": sample_rate,
            "speaking_rate": kwargs.get("speaking_rate", self.speaking_rate),
            "pitch": kwargs.get("pitch", self.pitch),
            "volume_gain_db": kwargs.get("volume_gain_db", self.volume_gain_db),
        }
        
        # Add effects profile if specified
        if self.audio_profile:
            config_dict["effects_profile_id"] = [self.audio_profile]
        elif "audio_profile" in kwargs:
            config_dict["effects_profile_id"] = [kwargs["audio_profile"]]
        
        return texttospeech.AudioConfig(**config_dict)
    
    def _map_language_to_google(self, language: str) -> str:
        """Map standard language code to Google format"""
        # Handle full locale codes
        if "-" in language or "_" in language:
            return language.replace("_", "-")
        
        # Map short codes
        return self.LANGUAGE_MAP.get(language, "en-US")
    
    def _map_google_language_to_standard(self, google_lang: str) -> str:
        """Map Google language code back to standard format"""
        # Try to find in reverse mapping
        for std_lang, g_lang in self.LANGUAGE_MAP.items():
            if g_lang == google_lang:
                return std_lang
        
        # Return first part of locale
        return google_lang.split("-")[0]
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        chunks = []
        current_chunk = ""
        
        # Split by sentences (simple approach)
        sentences = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _estimate_audio_duration(
        self, 
        audio_size: int, 
        format: str, 
        sample_rate: int
    ) -> float:
        """Estimate audio duration from file size"""
        # Rough estimates based on format
        if format == "mp3":
            # MP3 at 128kbps
            return audio_size / (128 * 1024 / 8)
        elif format == "wav":
            # WAV is uncompressed
            bytes_per_second = sample_rate * 2  # 16-bit mono
            return audio_size / bytes_per_second
        elif format in ["ogg", "opus"]:
            # OGG/Opus at ~96kbps
            return audio_size / (96 * 1024 / 8)
        else:
            # Default estimate
            return audio_size / (128 * 1024 / 8)
    
    def get_cost_estimate(self, text: str) -> float:
        """
        Estimate cost for synthesis
        
        Google Cloud TTS pricing (as of 2024):
        - First 1 million characters free per month
        - Standard voices: $4.00 per 1 million characters
        - WaveNet voices: $16.00 per 1 million characters
        - Neural2 voices: $16.00 per 1 million characters
        """
        char_count = len(text)
        
        # Price per million characters
        if self.voice_type == "Standard":
            price_per_million = 4.00
        else:  # WaveNet or Neural2
            price_per_million = 16.00
        
        return (char_count / 1_000_000) * price_per_million
    
    async def test_connection(self) -> bool:
        """Test connection to Google Cloud TTS"""
        try:
            # Try to list voices (lightweight operation)
            response = self.client.list_voices()
            return len(response.voices) > 0
            
        except Exception as e:
            logger.error(f"Google Cloud TTS connection test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Close client connection
        if hasattr(self.client, 'transport') and hasattr(self.client.transport, 'close'):
            self.client.transport.close()