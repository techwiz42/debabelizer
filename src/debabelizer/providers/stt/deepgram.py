"""
Deepgram Speech-to-Text Provider for Debabelizer

Features:
- High-accuracy speech recognition using Nova-2 model
- Real-time streaming transcription via WebSocket
- Multiple language support (40+ languages)
- Word-level timestamps and confidence scores
- Automatic language detection
- Ultra-low latency (<300ms)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import json
from pathlib import Path
import uuid
from datetime import datetime

try:
    from deepgram import (
        DeepgramClient,
        PrerecordedOptions,
        LiveOptions,
        LiveTranscriptionEvents
    )
except ImportError:
    raise ImportError(
        "Deepgram SDK is required for Deepgram STT provider. "
        "Install it with: pip install deepgram-sdk"
    )

from ..base import STTProvider, TranscriptionResult, StreamingResult, WordTiming
from ..base.exceptions import (
    ProviderError, AuthenticationError, UnsupportedLanguageError,
    UnsupportedFormatError, ConnectionError as ProviderConnectionError
)

logger = logging.getLogger(__name__)


class DeepgramSTTProvider(STTProvider):
    """Deepgram Speech-to-Text Provider"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # Handle initialization parameters
        if config:
            api_key = api_key or config.get("api_key")
            self.model = config.get("model", "nova-2")
            self.language = config.get("language", "en-US")
        else:
            self.model = "nova-2"
            self.language = "en-US"
            
        if not api_key:
            raise AuthenticationError("Deepgram API key is required")
            
        # Don't pass config to super() if it contains api_key
        config_without_key = config.copy() if config else {}
        if config_without_key and 'api_key' in config_without_key:
            config_without_key.pop('api_key')
        super().__init__(api_key, **config_without_key)
        
        # Initialize Deepgram client
        self.client = DeepgramClient(api_key)
        
        # Supported languages by Deepgram
        self._supported_languages = [
            "en", "en-US", "en-GB", "en-AU", "en-IN", "en-NZ", "en-CA",
            "es", "es-ES", "es-419", "fr", "fr-CA", "de", "it", "pt", "pt-BR",
            "ru", "hi", "ja", "ko", "zh", "zh-CN", "zh-TW", "nl", "sv", "da",
            "no", "fi", "pl", "tr", "ar", "th", "vi", "uk", "cs", "sk", "hu",
            "ro", "bg", "hr", "sl", "et", "lv", "lt", "mt", "ga", "cy", "is"
        ]
        
        # Supported audio formats
        self._supported_formats = [
            "wav", "mp3", "mp4", "m4a", "flac", "ogg", "opus", "webm"
        ]
        
        # Active streaming sessions
        self.sessions = {}
        
    @property
    def name(self) -> str:
        return "deepgram"
        
    @property
    def supported_languages(self) -> List[str]:
        return self._supported_languages.copy()
        
    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats.copy()
        
    @property
    def supports_streaming(self) -> bool:
        return True
        
    @property
    def supports_language_detection(self) -> bool:
        return True
        
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file"""
        try:
            # Read audio file
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                
            # Use transcribe_audio method
            return await self.transcribe_audio(
                audio_data,
                audio_format=Path(file_path).suffix.lstrip('.'),
                language=language,
                language_hints=language_hints,
                **kwargs
            )
            
        except FileNotFoundError:
            raise ProviderError(f"Audio file not found: {file_path}")
            
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
        try:
            # Prepare options
            detect_language = False
            if language:
                if language.lower() == "auto":
                    detect_language = True
                    lang_code = None
                else:
                    lang_code = self._normalize_language(language)
            else:
                lang_code = self._normalize_language(self.language)
                
            # Set up transcription options
            deepgram_options = PrerecordedOptions(
                model=kwargs.get("model", self.model),
                language=lang_code if lang_code else None,
                detect_language=detect_language,
                punctuate=kwargs.get("punctuate", True),
                utterances=kwargs.get("utterances", False),
                diarize=kwargs.get("diarize", False),
                smart_format=kwargs.get("smart_format", True)
            )
            
            # Get transcription
            response = await asyncio.to_thread(
                self.client.listen.prerecorded.v("1").transcribe_file,
                {"buffer": audio_data},
                deepgram_options
            )
            
            # Parse response
            if not response or not response.results:
                raise ProviderError("Empty response from Deepgram")
                
            # Get first channel's best alternative
            channel = response.results.channels[0]
            if not channel.alternatives:
                raise ProviderError("No transcription alternatives found")
                
            best_alt = channel.alternatives[0]
            
            # Extract words with timestamps
            words = []
            if hasattr(best_alt, 'words') and best_alt.words:
                for word in best_alt.words:
                    words.append(WordTiming(
                        word=word.word,
                        start_time=word.start,
                        end_time=word.end,
                        confidence=word.confidence
                    ))
                    
            # Determine language
            detected_lang = None
            if hasattr(channel, 'detected_language'):
                detected_lang = channel.detected_language
            elif detect_language and hasattr(response, 'metadata') and hasattr(response.metadata, 'detected_language'):
                detected_lang = response.metadata.detected_language
                
            # Get duration
            duration = None
            if hasattr(response, 'metadata') and hasattr(response.metadata, 'duration'):
                duration = response.metadata.duration
            elif words:
                # Calculate from word timestamps
                duration = words[-1].end_time
                
            return TranscriptionResult(
                text=best_alt.transcript,
                confidence=best_alt.confidence,
                language_detected=detected_lang or lang_code or self.language,
                duration=duration or 0.0,
                words=words,
                metadata={"provider": "deepgram", "model": self.model}
            )
            
        except Exception as e:
            logger.error(f"Deepgram transcription error: {str(e)}")
            raise ProviderError(f"Transcription failed: {str(e)}")
            
    async def start_streaming(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Start a streaming transcription session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Prepare language
            if language and language.lower() != "auto":
                lang_code = self._normalize_language(language)
            else:
                lang_code = self._normalize_language(self.language)
                
            # Set up live transcription options
            live_options = LiveOptions(
                model=kwargs.get("model", self.model),
                language=lang_code,
                punctuate=kwargs.get("punctuate", True),
                interim_results=kwargs.get("interim_results", True),
                utterance_end_ms=kwargs.get("utterance_end_ms", 1000),
                vad_events=kwargs.get("vad_events", False),
                smart_format=kwargs.get("smart_format", True),
                encoding="linear16" if audio_format in ["wav", "pcm"] else audio_format,
                sample_rate=sample_rate
            )
            
            # Create live transcription connection
            connection = self.client.listen.live.v("1")
            
            # Store session info
            self.sessions[session_id] = {
                "connection": connection,
                "results_queue": asyncio.Queue(),
                "is_active": True,
                "options": live_options
            }
            
            # Event handlers
            def on_message(self_conn, result, **kwargs):
                try:
                    if result and hasattr(result, 'channel'):
                        channel = result.channel
                        if hasattr(channel, 'alternatives') and channel.alternatives:
                            best_alt = channel.alternatives[0]
                            
                            # Create streaming result
                            streaming_result = StreamingResult(
                                session_id=session_id,
                                is_final=result.is_final if hasattr(result, 'is_final') else True,
                                text=best_alt.transcript,
                                confidence=best_alt.confidence,
                                timestamp=datetime.now()
                            )
                            
                            # Add to queue
                            asyncio.create_task(
                                self.sessions[session_id]["results_queue"].put(streaming_result)
                            )
                except Exception as e:
                    logger.error(f"Error processing Deepgram message: {e}")
                    
            def on_error(self_conn, error, **kwargs):
                logger.error(f"Deepgram streaming error: {error}")
                if session_id in self.sessions:
                    self.sessions[session_id]["is_active"] = False
                    
            # Register event handlers
            connection.on(LiveTranscriptionEvents.Transcript, on_message)
            connection.on(LiveTranscriptionEvents.Error, on_error)
            
            # Start connection
            await asyncio.to_thread(connection.start, live_options)
            
            logger.info(f"Started Deepgram streaming session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start Deepgram streaming: {str(e)}")
            raise ProviderError(f"Failed to start streaming: {str(e)}")
            
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming session"""
        if session_id not in self.sessions:
            raise ProviderError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        if not session["is_active"]:
            raise ProviderError(f"Session {session_id} is not active")
            
        try:
            connection = session["connection"]
            await asyncio.to_thread(connection.send, audio_chunk)
        except Exception as e:
            logger.error(f"Error streaming audio to Deepgram: {e}")
            raise ProviderError(f"Failed to stream audio: {str(e)}")
            
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results"""
        if session_id not in self.sessions:
            raise ProviderError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        results_queue = session["results_queue"]
        
        while session["is_active"]:
            try:
                # Get result with timeout
                result = await asyncio.wait_for(results_queue.get(), timeout=30.0)
                yield result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for results in session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error getting streaming results: {e}")
                break
                
    async def stop_streaming(self, session_id: str) -> None:
        """Stop streaming session and cleanup"""
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        session["is_active"] = False
        
        try:
            connection = session["connection"]
            await asyncio.to_thread(connection.finish)
        except Exception as e:
            logger.error(f"Error stopping Deepgram session: {e}")
        finally:
            # Clean up session
            del self.sessions[session_id]
            logger.info(f"Stopped Deepgram streaming session: {session_id}")
            
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        **options
    ) -> AsyncGenerator[StreamingResult, None]:
        """Transcribe streaming audio (convenience method)"""
        # Start streaming session
        session_id = await self.start_streaming(
            language=language,
            **options
        )
        
        try:
            # Stream audio in background
            async def stream_audio_task():
                try:
                    async for chunk in audio_stream:
                        if chunk:
                            await self.stream_audio(session_id, chunk)
                except Exception as e:
                    logger.error(f"Error in audio streaming task: {e}")
                finally:
                    await self.stop_streaming(session_id)
                    
            # Start audio streaming task
            audio_task = asyncio.create_task(stream_audio_task())
            
            # Yield results
            async for result in self.get_streaming_results(session_id):
                yield result
                
        finally:
            # Ensure cleanup
            await self.stop_streaming(session_id)
            
    def _normalize_language(self, language: str) -> str:
        """Normalize language code for Deepgram"""
        # Handle auto-detection
        if language.lower() == "auto":
            return None  # Will be handled by detect_language=True
            
        # Map common language codes to Deepgram format
        language_map = {
            "en": "en-US",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "zh": "zh",
            "ja": "ja",
            "ko": "ko",
            "pt": "pt",
            "ru": "ru",
            "it": "it",
            "nl": "nl",
            "pl": "pl",
            "tr": "tr",
            "ar": "ar",
            "hi": "hi"
        }
        
        # Check if it's already a full locale
        if language in self._supported_languages:
            return language
            
        # Try to map short code
        short_code = language.split("-")[0].lower()
        if short_code in language_map:
            return language_map[short_code]
            
        # Check if the language is supported
        if language.lower() not in [lang.lower() for lang in self._supported_languages]:
            raise UnsupportedLanguageError(f"Language '{language}' is not supported by Deepgram")
            
        return language