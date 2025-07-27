"""
Google Cloud Speech-to-Text Provider

Implements STT using Google Cloud Speech-to-Text API with support for:
- 125+ languages and variants
- Real-time streaming transcription
- Word-level timestamps and confidence scores
- Automatic punctuation
- Speaker diarization
- Profanity filtering
- Multiple acoustic models
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
import json

from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.api_core import exceptions as google_exceptions
import google.auth

from ..base import STTProvider, TranscriptionResult, StreamingResult, WordTiming
from ..base.exceptions import ProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class GoogleSTTProvider(STTProvider):
    """
    Google Cloud Speech-to-Text Provider
    
    High-quality speech recognition with extensive language support
    and advanced features like speaker diarization.
    """
    
    name = "google"
    
    # Language mapping for Google Cloud Speech
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
        "no": "no-NO",
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
        "sr": "sr-RS",
        "sw": "sw-KE",
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
        model: str = "latest_long",
        enable_automatic_punctuation: bool = True,
        enable_word_time_offsets: bool = True,
        enable_speaker_diarization: bool = False,
        diarization_speaker_count: int = 2,
        profanity_filter: bool = False,
        **kwargs
    ):
        """
        Initialize Google Cloud Speech-to-Text Provider
        
        Args:
            credentials_path: Path to service account JSON file
            project_id: Google Cloud project ID
            model: Recognition model ('latest_long', 'latest_short', 'command_and_search', etc.)
            enable_automatic_punctuation: Add punctuation to results
            enable_word_time_offsets: Include word-level timestamps
            enable_speaker_diarization: Enable speaker detection
            diarization_speaker_count: Number of speakers (if diarization enabled)
            profanity_filter: Filter profanity in results
        """
        super().__init__()
        
        self.project_id = project_id
        self.model = model
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_word_time_offsets = enable_word_time_offsets
        self.enable_speaker_diarization = enable_speaker_diarization
        self.diarization_speaker_count = diarization_speaker_count
        self.profanity_filter = profanity_filter
        
        # Initialize credentials
        if credentials_path:
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            # Initialize client
            self.client = speech_v1.SpeechClient()
            
            # Get project ID if not provided
            if not self.project_id:
                _, self.project_id = google.auth.default()
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Google Cloud Speech client: {e}",
                self.name
            )
        
        # Streaming sessions
        self.streaming_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized Google Cloud STT with model: {model}")
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGE_MAP.keys())
    
    @property
    def supports_streaming(self) -> bool:
        """Google Cloud Speech supports streaming"""
        return True
    
    @property
    def supports_language_detection(self) -> bool:
        """Google Cloud Speech supports automatic language detection"""
        return True
    
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file
        
        Args:
            file_path: Path to audio file
            language: Language code (e.g., 'en', 'es')
            language_hints: Alternative language codes
            **kwargs: Additional Google-specific options
            
        Returns:
            TranscriptionResult with transcription
        """
        try:
            # Read audio file
            with open(file_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Configure audio
            audio = speech_v1.RecognitionAudio(content=content)
            
            # Build config
            config = self._build_recognition_config(
                language=language,
                language_hints=language_hints,
                **kwargs
            )
            
            # Perform transcription
            logger.info(f"Transcribing file with Google Cloud Speech: {file_path}")
            start_time = datetime.now()
            
            # Use long running recognize for files > 1 minute
            if len(content) > 10 * 1024 * 1024:  # 10MB threshold
                operation = self.client.long_running_recognize(
                    config=config,
                    audio=audio
                )
                response = operation.result(timeout=300)
            else:
                response = self.client.recognize(
                    config=config,
                    audio=audio
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Process results
            if not response.results:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected=language,
                    duration=duration
                )
            
            # Combine all results
            full_transcript = []
            all_words = []
            total_confidence = 0.0
            num_alternatives = 0
            
            for result in response.results:
                # Get best alternative
                alternative = result.alternatives[0]
                full_transcript.append(alternative.transcript)
                
                # Track confidence
                if alternative.confidence:
                    total_confidence += alternative.confidence
                    num_alternatives += 1
                
                # Extract word timings
                if self.enable_word_time_offsets and alternative.words:
                    for word_info in alternative.words:
                        all_words.append(WordTiming(
                            word=word_info.word,
                            start_time=word_info.start_time.total_seconds(),
                            end_time=word_info.end_time.total_seconds(),
                            confidence=word_info.confidence if hasattr(word_info, 'confidence') else None,
                            speaker_tag=word_info.speaker_tag if self.enable_speaker_diarization else None
                        ))
            
            # Calculate average confidence
            avg_confidence = total_confidence / num_alternatives if num_alternatives > 0 else 0.0
            
            # Detect language from first result if available
            detected_language = None
            if hasattr(response.results[0], 'language_code'):
                detected_language = self._map_google_language_to_standard(
                    response.results[0].language_code
                )
            
            return TranscriptionResult(
                text=" ".join(full_transcript),
                confidence=avg_confidence,
                language_detected=detected_language or language,
                duration=duration,
                words=all_words if all_words else [],
                metadata={
                    "model": self.model,
                    "speaker_diarization": self.enable_speaker_diarization,
                    "num_results": len(response.results)
                }
            )
            
        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(f"Google API error: {e}", self.name)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ProviderError(f"Transcription failed: {e}", self.name)
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe raw audio data"""
        # Create audio object
        audio = speech_v1.RecognitionAudio(content=audio_data)
        
        # Build config with format info
        config = self._build_recognition_config(
            language=language,
            language_hints=language_hints,
            audio_format=audio_format,
            sample_rate=sample_rate,
            **kwargs
        )
        
        try:
            # Perform transcription
            logger.info(f"Transcribing {len(audio_data)} bytes of {audio_format} audio")
            start_time = datetime.now()
            
            response = self.client.recognize(
                config=config,
                audio=audio
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Process results (same as file transcription)
            if not response.results:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected=language,
                    duration=duration
                )
            
            # Process response same as file transcription
            full_transcript = []
            all_words = []
            total_confidence = 0.0
            num_alternatives = 0
            
            for result in response.results:
                alternative = result.alternatives[0]
                full_transcript.append(alternative.transcript)
                
                if alternative.confidence:
                    total_confidence += alternative.confidence
                    num_alternatives += 1
                
                if self.enable_word_time_offsets and alternative.words:
                    for word_info in alternative.words:
                        all_words.append(WordTiming(
                            word=word_info.word,
                            start_time=word_info.start_time.total_seconds(),
                            end_time=word_info.end_time.total_seconds(),
                            confidence=word_info.confidence if hasattr(word_info, 'confidence') else None,
                            speaker_tag=word_info.speaker_tag if self.enable_speaker_diarization else None
                        ))
            
            avg_confidence = total_confidence / num_alternatives if num_alternatives > 0 else 0.0
            
            return TranscriptionResult(
                text=" ".join(full_transcript),
                confidence=avg_confidence,
                language_detected=language,
                duration=duration,
                words=all_words if all_words else []
            )
            
        except google_exceptions.GoogleAPIError as e:
            raise ProviderError(f"Google API error: {e}", self.name)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise ProviderError(f"Transcription failed: {e}", self.name)
    
    async def start_streaming(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Start a streaming transcription session"""
        import uuid
        session_id = str(uuid.uuid4())
        
        # Build streaming config
        config = self._build_recognition_config(
            language=language,
            language_hints=language_hints,
            audio_format=audio_format,
            sample_rate=sample_rate,
            streaming=True,
            **kwargs
        )
        
        # Create session
        self.streaming_sessions[session_id] = {
            "config": config,
            "audio_queue": asyncio.Queue(),
            "result_queue": asyncio.Queue(),
            "active": True,
            "task": None
        }
        
        # Start streaming task
        session = self.streaming_sessions[session_id]
        session["task"] = asyncio.create_task(
            self._streaming_recognize(session_id)
        )
        
        logger.info(f"Started Google streaming session: {session_id}")
        return session_id
    
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming session"""
        if session_id not in self.streaming_sessions:
            raise ProviderError(f"Invalid session ID: {session_id}", self.name)
        
        session = self.streaming_sessions[session_id]
        if not session["active"]:
            raise ProviderError(f"Session {session_id} is not active", self.name)
        
        await session["audio_queue"].put(audio_chunk)
    
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results"""
        if session_id not in self.streaming_sessions:
            raise ProviderError(f"Invalid session ID: {session_id}", self.name)
        
        session = self.streaming_sessions[session_id]
        
        while session["active"] or not session["result_queue"].empty():
            try:
                result = await asyncio.wait_for(
                    session["result_queue"].get(),
                    timeout=0.1
                )
                yield result
            except asyncio.TimeoutError:
                if not session["active"]:
                    break
    
    async def stop_streaming(self, session_id: str) -> None:
        """Stop streaming transcription session"""
        if session_id not in self.streaming_sessions:
            return
        
        session = self.streaming_sessions[session_id]
        session["active"] = False
        
        # Cancel streaming task
        if session["task"] and not session["task"].done():
            session["task"].cancel()
            try:
                await session["task"]
            except asyncio.CancelledError:
                pass
        
        # Clean up
        del self.streaming_sessions[session_id]
        logger.info(f"Stopped Google streaming session: {session_id}")
    
    def _build_recognition_config(
        self,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        streaming: bool = False,
        **kwargs
    ) -> speech_v1.RecognitionConfig:
        """Build Google Speech recognition config"""
        # Map language code
        language_code = self._map_language_to_google(language) if language else "en-US"
        
        # Map audio encoding
        encoding_map = {
            "wav": enums.RecognitionConfig.AudioEncoding.LINEAR16,
            "flac": enums.RecognitionConfig.AudioEncoding.FLAC,
            "mp3": enums.RecognitionConfig.AudioEncoding.MP3,
            "opus": enums.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "mulaw": enums.RecognitionConfig.AudioEncoding.MULAW,
        }
        encoding = encoding_map.get(audio_format.lower(), enums.RecognitionConfig.AudioEncoding.LINEAR16)
        
        # Build config
        config_dict = {
            "encoding": encoding,
            "sample_rate_hertz": sample_rate,
            "language_code": language_code,
            "enable_automatic_punctuation": self.enable_automatic_punctuation,
            "enable_word_time_offsets": self.enable_word_time_offsets and not streaming,
            "profanity_filter": self.profanity_filter,
            "model": self.model,
            "use_enhanced": True,  # Use enhanced models when available
        }
        
        # Add alternative languages
        if language_hints:
            config_dict["alternative_language_codes"] = [
                self._map_language_to_google(lang) for lang in language_hints
            ]
        
        # Add speaker diarization
        if self.enable_speaker_diarization and not streaming:
            config_dict["enable_speaker_diarization"] = True
            config_dict["diarization_speaker_count"] = self.diarization_speaker_count
        
        # Add any custom options
        for key, value in kwargs.items():
            if key in ["max_alternatives", "speech_contexts", "metadata"]:
                config_dict[key] = value
        
        return speech_v1.RecognitionConfig(**config_dict)
    
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
    
    async def _streaming_recognize(self, session_id: str) -> None:
        """Handle streaming recognition"""
        session = self.streaming_sessions[session_id]
        config = session["config"]
        
        try:
            # Create request generator
            def request_generator():
                # First request with config
                yield speech_v1.StreamingRecognizeRequest(
                    streaming_config=speech_v1.StreamingRecognitionConfig(
                        config=config,
                        interim_results=True,
                    )
                )
                
                # Stream audio chunks
                while session["active"]:
                    try:
                        # Non-blocking get with timeout
                        audio_chunk = asyncio.run_coroutine_threadsafe(
                            asyncio.wait_for(session["audio_queue"].get(), timeout=0.1),
                            asyncio.get_event_loop()
                        ).result()
                        
                        yield speech_v1.StreamingRecognizeRequest(
                            audio_content=audio_chunk
                        )
                    except:
                        if not session["active"]:
                            break
            
            # Perform streaming recognition
            responses = self.client.streaming_recognize(request_generator())
            
            # Process responses
            for response in responses:
                if not session["active"]:
                    break
                
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        # Create streaming result
                        streaming_result = StreamingResult(
                            session_id=session_id,
                            is_final=result.is_final,
                            text=alternative.transcript,
                            confidence=alternative.confidence if hasattr(alternative, 'confidence') else 0.0
                        )
                        
                        # Add to result queue
                        await session["result_queue"].put(streaming_result)
                        
        except Exception as e:
            logger.error(f"Streaming recognition error: {e}")
            # Put error result
            error_result = StreamingResult(
                session_id=session_id,
                is_final=True,
                text="",
                confidence=0.0,
                metadata={"error": str(e)}
            )
            await session["result_queue"].put(error_result)
        finally:
            session["active"] = False
    
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """
        Estimate cost for transcription
        
        Google Cloud Speech pricing (as of 2024):
        - First 60 minutes free per month
        - Standard models: $0.006 per 15 seconds
        - Enhanced models: $0.009 per 15 seconds
        """
        # Using enhanced model pricing
        cost_per_15_seconds = 0.009
        chunks = duration_seconds / 15.0
        return chunks * cost_per_15_seconds
    
    async def test_connection(self) -> bool:
        """Test connection to Google Cloud Speech"""
        try:
            # Try a simple recognition request with empty audio
            config = speech_v1.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )
            
            # Create minimal audio (1 second of silence)
            import struct
            silence = struct.pack('<' + 'h' * 16000, *([0] * 16000))
            audio = speech_v1.RecognitionAudio(content=silence)
            
            # This should work even with silence
            self.client.recognize(config=config, audio=audio)
            return True
            
        except Exception as e:
            logger.error(f"Google Cloud Speech connection test failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Stop all streaming sessions
        session_ids = list(self.streaming_sessions.keys())
        for session_id in session_ids:
            await self.stop_streaming(session_id)
        
        # Close client connection
        if hasattr(self.client, 'transport') and hasattr(self.client.transport, 'close'):
            self.client.transport.close()