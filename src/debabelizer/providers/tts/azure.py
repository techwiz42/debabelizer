"""
Azure Cognitive Services Text-to-Speech Provider

Implements TTS using Azure Speech Services with support for:
- 300+ neural voices in 140+ languages
- Custom neural voices and voice tuning
- SSML (Speech Synthesis Markup Language) support
- Real-time and batch synthesis
- Voice styling and emotions
- Audio output formats (WAV, MP3, OGG, etc.)
- Pronunciation and phoneme control
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from datetime import datetime
import io
import wave

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import (
    SpeechConfig, AudioConfig, SpeechSynthesizer,
    SpeechSynthesisResult, ResultReason, CancellationReason
)

from ..base import TTSProvider, SynthesisResult, Voice, AudioFormat
from ..base.exceptions import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class AzureTTSProvider(TTSProvider):
    """
    Azure Cognitive Services Text-to-Speech Provider
    
    Enterprise-grade TTS with neural voices and extensive customization options.
    """
    
    name = "azure"
    
    # Language mapping for Azure Speech
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
        "ar": "ar-SA",
        "hi": "hi-IN",
        "nl": "nl-NL",
        "pl": "pl-PL",
        "tr": "tr-TR",
        "sv": "sv-SE",
        "da": "da-DK",
        "no": "nb-NO",
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
        "sl": "sl-SI",
        "et": "et-EE",
        "lv": "lv-LV",
        "lt": "lt-LT",
        "ca": "ca-ES",
        "eu": "eu-ES",
        "gl": "gl-ES",
        "af": "af-ZA",
        "sq": "sq-AL",
        "am": "am-ET",
        "hy": "hy-AM",
        "az": "az-AZ",
        "bn": "bn-IN",
        "bs": "bs-BA",
        "my": "my-MM",
        "zh-tw": "zh-TW",
        "cy": "cy-GB",
        "gu": "gu-IN",
        "is": "is-IS",
        "jv": "jv-ID",
        "kn": "kn-IN",
        "km": "km-KH",
        "lo": "lo-LA",
        "mk": "mk-MK",
        "ml": "ml-IN",
        "mn": "mn-MN",
        "ne": "ne-NP",
        "ps": "ps-AF",
        "si": "si-LK",
        "su": "su-ID",
        "sw": "sw-KE",
        "ta": "ta-IN",
        "te": "te-IN",
        "ur": "ur-PK",
        "uz": "uz-UZ",
        "zu": "zu-ZA",
    }
    
    # Popular neural voices by language
    DEFAULT_VOICES = {
        "en-US": "en-US-JennyNeural",
        "en-GB": "en-GB-SoniaNeural",
        "es-ES": "es-ES-ElviraNeural",
        "es-MX": "es-MX-DaliaNeural",
        "fr-FR": "fr-FR-DeniseNeural",
        "de-DE": "de-DE-KatjaNeural",
        "it-IT": "it-IT-ElsaNeural",
        "pt-BR": "pt-BR-FranciscaNeural",
        "ru-RU": "ru-RU-SvetlanaNeural",
        "ja-JP": "ja-JP-NanamiNeural",
        "ko-KR": "ko-KR-SunHiNeural",
        "zh-CN": "zh-CN-XiaoxiaoNeural",
        "ar-SA": "ar-SA-ZariyahNeural",
        "hi-IN": "hi-IN-SwaraNeural",
        "nl-NL": "nl-NL-ColetteNeural",
        "pl-PL": "pl-PL-ZofiaNeural",
        "tr-TR": "tr-TR-EmelNeural",
        "sv-SE": "sv-SE-SofieNeural",
        "da-DK": "da-DK-ChristelNeural",
        "nb-NO": "nb-NO-PernilleNeural",
        "fi-FI": "fi-FI-NooraNeural",
    }
    
    def __init__(
        self,
        api_key: str,
        region: str = "eastus",
        voice: str = "en-US-JennyNeural",
        speaking_rate: str = "1.0",
        pitch: str = "+0Hz",
        volume: str = "100",
        output_format: str = "Audio16Khz32KBitRateMonoMp3",
        **kwargs
    ):
        """
        Initialize Azure Speech TTS Provider
        
        Args:
            api_key: Azure Speech Services API key
            region: Azure region (e.g., 'eastus', 'westeurope')
            voice: Default voice name (e.g., 'en-US-JennyNeural')
            speaking_rate: Speaking rate (0.5-2.0 or percentage like '150%')
            pitch: Pitch adjustment (e.g., '+10Hz', '-5st', '150%')
            volume: Volume level (0-100 or percentage)
            output_format: Audio output format
        """
        super().__init__()
        
        self.api_key = api_key
        self.region = region
        self.default_voice = voice
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.volume = volume
        self.output_format = output_format
        
        # Initialize speech config
        try:
            self.speech_config = SpeechConfig(
                subscription=api_key,
                region=region
            )
            
            # Set default voice
            self.speech_config.speech_synthesis_voice_name = voice
            
            # Set output format
            format_map = {
                "wav": speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
                "mp3": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
                "ogg": speechsdk.SpeechSynthesisOutputFormat.Ogg16Khz16BitMonoOpus,
                "webm": speechsdk.SpeechSynthesisOutputFormat.Webm16Khz16BitMonoOpus,
                "alaw": speechsdk.SpeechSynthesisOutputFormat.Riff8Khz8BitMonoALaw,
                "mulaw": speechsdk.SpeechSynthesisOutputFormat.Riff8Khz8BitMonoMULaw,
            }
            
            if output_format.lower() in format_map:
                self.speech_config.set_speech_synthesis_output_format(format_map[output_format.lower()])
            elif hasattr(speechsdk.SpeechSynthesisOutputFormat, output_format):
                self.speech_config.set_speech_synthesis_output_format(
                    getattr(speechsdk.SpeechSynthesisOutputFormat, output_format)
                )
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Azure Speech config: {e}",
                self.name
            )
        
        # Cache for voices
        self._voices_cache = None
        self._voices_cache_time = None
        self._cache_duration = 3600  # 1 hour
        
        logger.info(f"Initialized Azure Speech TTS for region: {region}")
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supports_streaming(self) -> bool:
        """Azure Speech supports streaming synthesis"""
        return True
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 16000,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice object (optional)
            voice_id: Voice ID string (e.g., "en-US-JennyNeural")
            audio_format: Desired output format
            sample_rate: Sample rate in Hz
            **kwargs: Additional Azure-specific options
            
        Returns:
            SynthesisResult with audio data
        """
        try:
            # Create speech config for this request
            speech_config = SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            
            # Determine voice
            if voice:
                voice_name = voice.voice_id
            elif voice_id:
                voice_name = voice_id
            else:
                voice_name = self.default_voice
            
            # Set output format based on request
            if audio_format:
                format_str = audio_format.format
            else:
                format_str = kwargs.get("output_format", "mp3")
            
            self._set_output_format(speech_config, format_str, sample_rate)
            
            # Build SSML with voice styling
            ssml_text = self._build_ssml(
                text=text,
                voice_name=voice_name,
                speaking_rate=kwargs.get("speaking_rate", self.speaking_rate),
                pitch=kwargs.get("pitch", self.pitch),
                volume=kwargs.get("volume", self.volume),
                style=kwargs.get("style"),
                emotion=kwargs.get("emotion"),
                **kwargs
            )
            
            # Create synthesizer
            synthesizer = SpeechSynthesizer(speech_config=speech_config)
            
            # Perform synthesis
            logger.info(f"Synthesizing {len(text)} characters with voice: {voice_name}")
            start_time = datetime.now()
            
            result = synthesizer.speak_ssml(ssml_text)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Check result
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # Calculate audio duration
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
                        "synthesis_time": duration,
                        "region": self.region
                    }
                )
            elif result.reason == ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"Synthesis canceled: {cancellation.reason}"
                if cancellation.error_details:
                    error_msg += f" - {cancellation.error_details}"
                raise ProviderError(error_msg, self.name)
            else:
                raise ProviderError(f"Synthesis failed with reason: {result.reason}", self.name)
                
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            logger.error(f"Synthesis failed: {e}")
            raise ProviderError(f"Synthesis failed: {e}", self.name)
    
    async def synthesize_long_text(
        self,
        text: str,
        voice: Optional[Voice] = None,
        voice_id: Optional[str] = None,
        audio_format: Optional[AudioFormat] = None,
        sample_rate: int = 16000,
        **kwargs
    ) -> SynthesisResult:
        """
        Synthesize long text by splitting into chunks
        
        Azure Speech TTS has no strict character limit but we'll chunk for better performance.
        """
        # If text is reasonable length, use single synthesis
        if len(text) <= 10000:
            return await self.synthesize(
                text, voice, voice_id, audio_format, sample_rate, **kwargs
            )
        
        # Split text into chunks
        chunks = self._split_text(text, 9000)  # Leave some margin
        
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
        sample_rate: int = 16000,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis results in real-time"""
        try:
            # Create speech config
            speech_config = SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            
            # Determine voice
            if voice:
                voice_name = voice.voice_id
            elif voice_id:
                voice_name = voice_id
            else:
                voice_name = self.default_voice
            
            # Set output format
            if audio_format:
                format_str = audio_format.format
            else:
                format_str = kwargs.get("output_format", "mp3")
            
            self._set_output_format(speech_config, format_str, sample_rate)
            
            # Build SSML
            ssml_text = self._build_ssml(
                text=text,
                voice_name=voice_name,
                speaking_rate=kwargs.get("speaking_rate", self.speaking_rate),
                pitch=kwargs.get("pitch", self.pitch),
                volume=kwargs.get("volume", self.volume),
                **kwargs
            )
            
            # Create pull stream for streaming
            class AudioDataStream:
                def __init__(self):
                    self.data = b""
                    self.position = 0
                    self.complete = False
                
                def write(self, data):
                    self.data += data
                
                def read(self, size):
                    if self.position >= len(self.data):
                        return b""
                    end_pos = min(self.position + size, len(self.data))
                    chunk = self.data[self.position:end_pos]
                    self.position = end_pos
                    return chunk
                
                def close(self):
                    self.complete = True
            
            # Create audio config for streaming
            stream = AudioDataStream()
            audio_config = AudioConfig(use_default_speaker=False)
            
            # Create synthesizer
            synthesizer = SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Set up event handler to capture audio data
            def handle_audio_data(evt):
                if evt.audio_data:
                    stream.write(evt.audio_data)
            
            synthesizer.synthesizing.connect(handle_audio_data)
            
            # Start synthesis in background
            synthesis_task = asyncio.create_task(
                self._run_synthesis(synthesizer, ssml_text, stream)
            )
            
            # Stream chunks as they become available
            chunk_size = 4096
            last_position = 0
            
            while not synthesis_task.done() or last_position < len(stream.data):
                # Check for new data
                if len(stream.data) > last_position:
                    # Read available data in chunks
                    while last_position < len(stream.data):
                        end_pos = min(last_position + chunk_size, len(stream.data))
                        chunk = stream.data[last_position:end_pos]
                        if chunk:
                            yield chunk
                            last_position = end_pos
                        else:
                            break
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(0.01)
            
            # Wait for synthesis to complete
            await synthesis_task
            
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise ProviderError(f"Streaming synthesis failed: {e}", self.name)
    
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
                logger.info("Fetching available voices from Azure Speech")
                
                # Create synthesizer to get voices
                speech_config = SpeechConfig(
                    subscription=self.api_key,
                    region=self.region
                )
                synthesizer = SpeechSynthesizer(speech_config=speech_config)
                
                # Get voices
                voices_result = synthesizer.get_voices_async().get()
                
                voices = []
                if voices_result.reason == ResultReason.VoicesListRetrieved:
                    for voice_info in voices_result.voices:
                        # Create Voice object
                        voice = Voice(
                            voice_id=voice_info.short_name,
                            name=voice_info.local_name,
                            language=self._map_azure_language_to_standard(voice_info.locale),
                            description=f"{voice_info.gender.name} - {voice_info.locale} - {voice_info.voice_type.name}"
                        )
                        voices.append(voice)
                else:
                    logger.warning(f"Failed to get voices: {voices_result.reason}")
                
                # Update cache
                self._voices_cache = voices
                self._voices_cache_time = current_time
            
            # Filter by language if requested
            if language:
                azure_lang = self._map_language_to_azure(language)
                filtered_voices = []
                for voice in voices:
                    # Check if voice locale matches
                    if azure_lang in voice.voice_id or voice.language == language:
                        filtered_voices.append(voice)
                return filtered_voices
            
            return voices
            
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            raise ProviderError(f"Failed to get voices: {e}", self.name)
    
    def _build_ssml(
        self,
        text: str,
        voice_name: str,
        speaking_rate: str = "1.0",
        pitch: str = "+0Hz",
        volume: str = "100",
        style: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build SSML for synthesis with voice styling"""
        # Escape XML characters in text
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Build prosody attributes
        prosody_attrs = []
        
        # Speaking rate
        if speaking_rate != "1.0":
            if speaking_rate.endswith("%"):
                prosody_attrs.append(f'rate="{speaking_rate}"')
            else:
                # Convert numeric rate to percentage
                try:
                    rate_num = float(speaking_rate)
                    rate_percent = f"{int(rate_num * 100)}%"
                    prosody_attrs.append(f'rate="{rate_percent}"')
                except:
                    prosody_attrs.append(f'rate="{speaking_rate}"')
        
        # Pitch
        if pitch != "+0Hz" and pitch != "0":
            prosody_attrs.append(f'pitch="{pitch}"')
        
        # Volume
        if volume != "100" and volume != "100%":
            if not volume.endswith("%"):
                volume += "%"
            prosody_attrs.append(f'volume="{volume}"')
        
        prosody_str = " ".join(prosody_attrs)
        
        # Build SSML
        ssml = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        ssml += f'<voice name="{voice_name}">'
        
        # Add style if supported (neural voices)
        if style and "Neural" in voice_name:
            ssml += f'<mstts:express-as style="{style}"'
            if emotion:
                ssml += f' styledegree="{emotion}"'
            ssml += '>'
        
        # Add prosody
        if prosody_str:
            ssml += f'<prosody {prosody_str}>'
            ssml += text
            ssml += '</prosody>'
        else:
            ssml += text
        
        # Close style tag
        if style and "Neural" in voice_name:
            ssml += '</mstts:express-as>'
        
        ssml += '</voice>'
        ssml += '</speak>'
        
        return ssml
    
    def _set_output_format(self, speech_config: SpeechConfig, format_str: str, sample_rate: int):
        """Set output format on speech config"""
        format_map = {
            "wav": speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
            "mp3": speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
            "ogg": speechsdk.SpeechSynthesisOutputFormat.Ogg16Khz16BitMonoOpus,
            "webm": speechsdk.SpeechSynthesisOutputFormat.Webm16Khz16BitMonoOpus,
        }
        
        # Adjust for sample rate
        if sample_rate == 24000:
            if format_str == "wav":
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
                )
            elif format_str == "mp3":
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
                )
            else:
                speech_config.set_speech_synthesis_output_format(format_map.get(format_str, format_map["mp3"]))
        elif sample_rate == 48000:
            if format_str == "wav":
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm
                )
            else:
                speech_config.set_speech_synthesis_output_format(format_map.get(format_str, format_map["mp3"]))
        else:
            speech_config.set_speech_synthesis_output_format(format_map.get(format_str, format_map["mp3"]))
    
    def _map_language_to_azure(self, language: str) -> str:
        """Map standard language code to Azure format"""
        # Handle full locale codes
        if "-" in language or "_" in language:
            return language.replace("_", "-")
        
        # Map short codes
        return self.LANGUAGE_MAP.get(language, "en-US")
    
    def _map_azure_language_to_standard(self, azure_lang: str) -> str:
        """Map Azure language code back to standard format"""
        # Try to find in reverse mapping
        for std_lang, az_lang in self.LANGUAGE_MAP.items():
            if az_lang == azure_lang:
                return std_lang
        
        # Return first part of locale
        return azure_lang.split("-")[0]
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        chunks = []
        current_chunk = ""
        
        # Split by sentences
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
        if format == "wav":
            # WAV is uncompressed
            bytes_per_second = sample_rate * 2  # 16-bit mono
            return audio_size / bytes_per_second
        elif format == "mp3":
            # MP3 compression ratio varies, rough estimate
            return audio_size / (128 * 1024 / 8)  # Assume 128kbps
        elif format in ["ogg", "webm"]:
            # OGG/WebM compression
            return audio_size / (96 * 1024 / 8)  # Assume 96kbps
        else:
            # Default estimate
            return audio_size / (128 * 1024 / 8)
    
    async def _run_synthesis(self, synthesizer: SpeechSynthesizer, ssml: str, stream) -> None:
        """Run synthesis and handle completion"""
        try:
            result = synthesizer.speak_ssml(ssml)
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                # Synthesis completed successfully
                stream.close()
            else:
                # Handle errors
                stream.close()
                if result.reason == ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    logger.error(f"Synthesis canceled: {cancellation.reason}")
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            stream.close()
    
    def get_cost_estimate(self, text: str) -> float:
        """
        Estimate cost for synthesis
        
        Azure Speech TTS pricing (as of 2024):
        - Standard voices: $4.00 per 1 million characters
        - Neural voices: $16.00 per 1 million characters
        - Custom Neural voices: $24.00 per 1 million characters
        """
        char_count = len(text)
        
        # Use neural voice pricing (most common)
        price_per_million = 16.00
        
        return (char_count / 1_000_000) * price_per_million
    
    async def test_connection(self) -> bool:
        """Test connection to Azure Speech Services"""
        try:
            # Create a simple synthesizer to test credentials
            speech_config = SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            
            synthesizer = SpeechSynthesizer(speech_config=speech_config)
            
            # Try to get voices (lightweight operation)
            voices_result = synthesizer.get_voices_async().get()
            return voices_result.reason == ResultReason.VoicesListRetrieved
            
        except Exception as e:
            logger.error(f"Azure Speech connection test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Azure SDK handles cleanup automatically
        pass