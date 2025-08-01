"""
Deepgram Speech-to-Text Provider for Debabelizer

Features:
- High-accuracy speech recognition
- Real-time streaming transcription
- Multiple language support
- Nova-2 model with enhanced accuracy
- WebSocket streaming for low-latency
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
import json

from ..base import STTProvider, TranscriptionResult, StreamingResult
from ..base.exceptions import (
    ProviderError, AuthenticationError, UnsupportedLanguageError,
    UnsupportedFormatError, ConnectionError as ProviderConnectionError
)

logger = logging.getLogger(__name__)


class DeepgramSTTProvider(STTProvider):
    """Deepgram Speech-to-Text Provider"""
    
    def __init__(self, api_key: str, **config):
        super().__init__(api_key, **config)
        self.base_url = "https://api.deepgram.com/v1"
        self.default_model = config.get("model", "nova-2")
        self.default_language = config.get("language", "en-US")
        
        # Supported languages by Deepgram
        self._supported_languages = [
            "en", "en-US", "en-GB", "en-AU", "en-IN", "en-NZ", "en-CA",
            "es", "es-ES", "es-419", "fr", "fr-CA", "de", "it", "pt", "pt-BR",
            "ru", "hi", "ja", "ko", "zh", "zh-CN", "zh-TW", "nl", "sv", "da",
            "no", "fi", "pl", "tr", "ar", "th", "vi", "uk", "cs", "sk", "hu",
            "ro", "bg", "hr", "sl", "et", "lv", "lt", "mt", "ga", "cy", "is"
        ]
        
    @property
    def name(self) -> str:
        return "deepgram"
        
    @property
    def supported_languages(self) -> List[str]:
        return self._supported_languages.copy()
        
    @property
    def supports_streaming(self) -> bool:
        return True
        
    @property
    def supports_language_detection(self) -> bool:
        return True
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def _prepare_options(self, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare Deepgram API options for prerecorded transcription"""
        # Extract detect_language but don't pass it to Deepgram API
        detect_language = kwargs.get("detect_language", False)
        
        # Define valid PrerecordedOptions constructor parameters
        # Based on Deepgram SDK v3+ PrerecordedOptions class
        valid_params = {
            "model", "language", "version", "tier", "punctuate", "profanity_filter",
            "redact", "diarize", "ner", "multichannel", "alternatives", "numerals",
            "smart_format", "utterances", "utt_split", "dictation", "measurements",
            "filler_words", "summarize", "detect_language", "paragraphs", 
            "sentiment", "intents", "topics", "extra"
        }
        
        options = {
            "model": kwargs.get("model", self.default_model),
            "language": language or self.default_language,
            "punctuate": kwargs.get("punctuate", True),
            "diarize": kwargs.get("diarize", False),
            "utterances": kwargs.get("utterances", True),
            "smart_format": kwargs.get("smart_format", True),
            "profanity_filter": kwargs.get("profanity_filter", False),
            "redact": kwargs.get("redact", []),
            "alternatives": kwargs.get("alternatives", 1),
            "numerals": kwargs.get("numerals", True)
        }
        
        # Add other valid parameters if provided
        for param in ["search", "keywords", "paragraphs", "sentiment", "intents", "topics"]:
            if param in kwargs:
                if param in valid_params:
                    options[param] = kwargs[param]
        
        # Add word-level timestamps if requested (default True)
        if kwargs.get("words", True):
            options["words"] = True
            
        # Filter out any parameters not supported by PrerecordedOptions
        filtered_options = {k: v for k, v in options.items() if k in valid_params}
            
        return filtered_options
        
    def _prepare_streaming_options(self, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare Deepgram API options for streaming transcription"""
        # Extract detect_language but handle it separately
        detect_language = kwargs.get("detect_language", False)
        
        options = {
            "model": kwargs.get("model", self.default_model),
            "punctuate": kwargs.get("punctuate", True),
            "smart_format": kwargs.get("smart_format", True),
            "profanity_filter": kwargs.get("profanity_filter", False),
            "numerals": kwargs.get("numerals", True),
            "interim_results": kwargs.get("interim_results", True),
            "utterance_end_ms": kwargs.get("utterance_end_ms", 1500),
            "vad_events": kwargs.get("vad_events", True),
            "encoding": "linear16",
            "sample_rate": kwargs.get("sample_rate", 16000),
            "channels": kwargs.get("channels", 1)
        }
        
        # Handle language and auto-detection
        # Deepgram automatically detects language when no language is specified
        if language and language != "auto":
            options["language"] = language
        elif not language or language == "auto" or detect_language:
            # For auto-detection, don't set language
            # Deepgram will automatically detect the language
            pass
        else:
            options["language"] = self.default_language
        
        return options
        
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe an audio file using Deepgram"""
        
        try:
            # Dynamic import to avoid requiring deepgram-sdk if not used
            from deepgram import DeepgramClient, PrerecordedOptions, FileSource
            
            # Initialize client
            deepgram = DeepgramClient(self.api_key)
            
            # Prepare options
            base_options = self._prepare_options(language, **kwargs)
            
            # Handle language detection - Deepgram detects automatically when no language is set
            if not language or language == "auto" or kwargs.get("detect_language", False):
                # Remove language from options for auto-detection
                base_options.pop("language", None)
                
            options = PrerecordedOptions(**base_options)
            
            # Read audio file
            with open(file_path, "rb") as audio_file:
                buffer_data = audio_file.read()
                
            payload: FileSource = {
                "buffer": buffer_data,
            }
            
            # Make request
            response = await asyncio.to_thread(
                deepgram.listen.prerecorded.v("1").transcribe_file,
                payload, options
            )
            
            # Parse response
            if not response.results or not response.results.channels:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected="unknown",
                    duration=0.0
                )
                
            channel = response.results.channels[0]
            if not channel.alternatives:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected="unknown", 
                    duration=0.0
                )
                
            alternative = channel.alternatives[0]
            
            # Extract word-level details
            words = []
            if hasattr(alternative, 'words') and alternative.words:
                words = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.confidence
                    }
                    for word in alternative.words
                ]
            
            # Get detected language
            detected_language = language or self.default_language
            if hasattr(response.results, 'summary') and hasattr(response.results.summary, 'language'):
                detected_language = response.results.summary.language
                
            # Calculate duration
            duration = 0.0
            if words:
                duration = words[-1]["end"]
            elif hasattr(response.results, 'summary') and hasattr(response.results.summary, 'duration'):
                duration = response.results.summary.duration
                
            return TranscriptionResult(
                text=alternative.transcript,
                confidence=alternative.confidence if hasattr(alternative, 'confidence') else 1.0,
                language_detected=detected_language,
                duration=duration,
                words=words,
                metadata={
                    "model": base_options["model"],
                    "language": detected_language,
                    "word_count": len(words),
                    "deepgram_request_id": getattr(response, 'metadata', {}).get('request_id')
                }
            )
            
        except ImportError:
            raise ProviderError(
                "Deepgram SDK not installed. Install with: pip install deepgram-sdk>=3.0.0",
                "deepgram"
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise AuthenticationError("Invalid Deepgram API key", "deepgram")
            elif "400" in str(e):
                raise UnsupportedFormatError(f"Unsupported audio format: {e}", "deepgram")
            elif "language" in str(e).lower():
                raise UnsupportedLanguageError(f"Unsupported language: {language}", "deepgram")
            else:
                raise ProviderError(f"Deepgram transcription failed: {e}", "deepgram")
                
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe raw audio data using Deepgram"""
        
        try:
            # Dynamic import to avoid requiring deepgram-sdk if not used
            from deepgram import DeepgramClient, PrerecordedOptions, FileSource
            
            # Initialize client
            deepgram = DeepgramClient(self.api_key)
            
            # Prepare options
            base_options = self._prepare_options(language, **kwargs)
            
            # Add audio format specific options
            if audio_format.lower() in ["pcm", "raw", "linear16"]:
                base_options["encoding"] = "linear16"
                base_options["sample_rate"] = sample_rate
                base_options["channels"] = kwargs.get("channels", 1)
                
            options = PrerecordedOptions(**base_options)
            
            # Prepare payload
            payload: FileSource = {
                "buffer": audio_data,
            }
            
            # Make request
            response = await asyncio.to_thread(
                deepgram.listen.prerecorded.v("1").transcribe_file,
                payload, options
            )
            
            # Parse response (same logic as transcribe_file)
            if not response.results or not response.results.channels:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected="unknown",
                    duration=0.0
                )
                
            channel = response.results.channels[0]
            if not channel.alternatives:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language_detected="unknown",
                    duration=0.0
                )
                
            alternative = channel.alternatives[0]
            
            # Extract word-level details
            words = []
            if hasattr(alternative, 'words') and alternative.words:
                words = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": word.confidence
                    }
                    for word in alternative.words
                ]
                
            # Get detected language
            detected_language = language or self.default_language
            if hasattr(response.results, 'summary') and hasattr(response.results.summary, 'language'):
                detected_language = response.results.summary.language
                
            # Calculate duration
            duration = 0.0
            if words:
                duration = words[-1]["end"]
            elif hasattr(response.results, 'summary') and hasattr(response.results.summary, 'duration'):
                duration = response.results.summary.duration
                
            return TranscriptionResult(
                text=alternative.transcript,
                confidence=alternative.confidence if hasattr(alternative, 'confidence') else 1.0,
                language_detected=detected_language,
                duration=duration,
                words=words,
                metadata={
                    "model": base_options["model"],
                    "language": detected_language,
                    "audio_format": audio_format,
                    "sample_rate": sample_rate,
                    "word_count": len(words)
                }
            )
            
        except ImportError:
            raise ProviderError(
                "Deepgram SDK not installed. Install with: pip install deepgram-sdk>=3.0.0",
                "deepgram"
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise AuthenticationError("Invalid Deepgram API key", "deepgram")
            elif "400" in str(e):
                raise UnsupportedFormatError(f"Unsupported audio format: {e}", "deepgram")
            else:
                raise ProviderError(f"Deepgram transcription failed: {e}", "deepgram")
                
                
    async def start_streaming(
        self,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Start a true WebSocket streaming transcription session"""
        
        try:
            # Dynamic import
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
            
            # Generate session ID
            import uuid
            session_id = str(uuid.uuid4())
            
            # Initialize client
            deepgram = DeepgramClient(self.api_key)
            
            # Prepare options (use streaming-specific options)
            # Extract channels to avoid duplicate parameter error
            channels = kwargs.pop("channels", 1)
            base_options = self._prepare_streaming_options(
                language, 
                sample_rate=sample_rate,
                channels=channels,
                **kwargs
            )
            
            # Override encoding if needed
            if audio_format.lower() not in ["pcm", "raw", "linear16"]:
                base_options["encoding"] = audio_format
            
            options = LiveOptions(**base_options)
            
            # Create WebSocket connection
            dg_connection = deepgram.listen.websocket.v("1")
            
            # Store connection for this session
            if not hasattr(self, '_streaming_sessions'):
                self._streaming_sessions = {}
                
            self._streaming_sessions[session_id] = {
                "connection": dg_connection,
                "options": options,
                "results": asyncio.Queue(),
                "connected": False,
                "error": None
            }
            
            # Setup event handlers with proper async handling
            # Capture the provider instance to avoid 'self' confusion with Deepgram connection
            provider_instance = self
            
            def on_open(self, open, **kwargs):
                logger.info(f"Deepgram WebSocket opened for session {session_id}")
                if hasattr(provider_instance, '_streaming_sessions') and session_id in provider_instance._streaming_sessions:
                    provider_instance._streaming_sessions[session_id]["connected"] = True
            
            def on_message(self, result, **kwargs):
                if hasattr(provider_instance, '_streaming_sessions') and session_id in provider_instance._streaming_sessions:
                    # Use call_soon_threadsafe to schedule the put operation in the main event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.call_soon_threadsafe(
                                lambda: asyncio.create_task(provider_instance._streaming_sessions[session_id]["results"].put(result))
                            )
                    except RuntimeError:
                        # No event loop running, store directly (synchronous fallback)
                        try:
                            provider_instance._streaming_sessions[session_id]["results"].put_nowait(result)
                        except asyncio.QueueFull:
                            pass  # Skip if queue is full
                    
            def on_error(self, error, **kwargs):
                logger.error(f"Deepgram streaming error for session {session_id}: {error}")
                if hasattr(provider_instance, '_streaming_sessions') and session_id in provider_instance._streaming_sessions:
                    provider_instance._streaming_sessions[session_id]["error"] = error
                    # Use call_soon_threadsafe to schedule the put operation in the main event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.call_soon_threadsafe(
                                lambda: asyncio.create_task(provider_instance._streaming_sessions[session_id]["results"].put({"error": str(error)}))
                            )
                    except RuntimeError:
                        # No event loop running, store directly (synchronous fallback)
                        try:
                            provider_instance._streaming_sessions[session_id]["results"].put_nowait({"error": str(error)})
                        except asyncio.QueueFull:
                            pass  # Skip if queue is full
            
            def on_close(self, close, **kwargs):
                logger.info(f"Deepgram WebSocket closed for session {session_id}")
                if hasattr(provider_instance, '_streaming_sessions') and session_id in provider_instance._streaming_sessions:
                    provider_instance._streaming_sessions[session_id]["connected"] = False
            
            # Register handlers
            dg_connection.on(LiveTranscriptionEvents.Open, on_open)
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            dg_connection.on(LiveTranscriptionEvents.Close, on_close)
            
            # Start connection
            if dg_connection.start(options):
                # Wait for connection to establish - poll until connected or timeout
                for attempt in range(50):  # Wait up to 5 seconds (50 * 0.1s)
                    await asyncio.sleep(0.1)
                    if session_id in self._streaming_sessions and self._streaming_sessions[session_id]["connected"]:
                        logger.info(f"Started Deepgram streaming session: {session_id}")
                        
                        # Send keepalive message immediately after connection to prevent timeout
                        try:
                            keepalive_message = json.dumps({"type": "KeepAlive"})
                            self._streaming_sessions[session_id]["connection"].send(keepalive_message)
                            logger.info(f"Sent keepalive message to session {session_id}")
                        except Exception as e:
                            logger.warning(f"Failed to send keepalive to session {session_id}: {e}")
                        
                        return session_id
                    if session_id in self._streaming_sessions and self._streaming_sessions[session_id]["error"]:
                        error = self._streaming_sessions[session_id]["error"]
                        raise ProviderConnectionError(f"Deepgram connection error: {error}", "deepgram")
                
                # Timeout waiting for connection
                raise ProviderConnectionError("Timeout waiting for Deepgram connection", "deepgram")
            else:
                raise ProviderConnectionError("Failed to start Deepgram streaming session", "deepgram")
                
        except ImportError:
            raise ProviderError(
                "Deepgram SDK not installed. Install with: pip install deepgram-sdk>=3.0.0",
                "deepgram"
            )
        except Exception as e:
            if "401" in str(e) or "unauthorized" in str(e).lower():
                raise AuthenticationError("Invalid Deepgram API key", "deepgram")
            else:
                raise ProviderError(f"Failed to start streaming: {e}", "deepgram")
                
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming session"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            raise ProviderError(f"Streaming session {session_id} not found", "deepgram")
            
        session = self._streaming_sessions[session_id]
        
        if not session["connected"]:
            raise ProviderError(f"Streaming session {session_id} not connected", "deepgram")
            
        try:
            # Send audio data
            session["connection"].send(audio_chunk)
            
        except Exception as e:
            raise ProviderError(f"Failed to send audio to session {session_id}: {e}", "deepgram")
    
    async def send_keepalive(self, session_id: str) -> None:
        """Send keepalive message to prevent Deepgram timeout"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            raise ProviderError(f"Streaming session {session_id} not found", "deepgram")
            
        session = self._streaming_sessions[session_id]
        
        if not session["connected"]:
            return
            
        try:
            # Send keepalive message as JSON
            keepalive_message = json.dumps({"type": "KeepAlive"})
            session["connection"].send(keepalive_message)
            
        except Exception as e:
            logger.error(f"Failed to send keepalive message to session {session_id}: {e}")
    
    async def finalize_streaming(self, session_id: str) -> None:
        """Send finalize message to get final transcription results"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            raise ProviderError(f"Streaming session {session_id} not found", "deepgram")
            
        session = self._streaming_sessions[session_id]
        
        if not session["connected"]:
            return
            
        try:
            # Send finalize message as JSON
            finalize_message = json.dumps({"type": "Finalize"})
            session["connection"].send(finalize_message)
            
        except Exception as e:
            logger.error(f"Failed to send finalize message to session {session_id}: {e}")
            
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results from true WebSocket connection"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            raise ProviderError(f"Streaming session {session_id} not found", "deepgram")
            
        session = self._streaming_sessions[session_id]
        results_queue = session["results"]
        
        try:
            while session["connected"] or not results_queue.empty():
                try:
                    # Wait for results with timeout
                    result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                    
                    # Handle error messages
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Received error in streaming results: {result['error']}")
                        continue
                    
                    # Handle VAD events
                    if hasattr(result, 'type'):
                        if result.type == 'SpeechStarted':
                            # Speech activity detected
                            streaming_result = StreamingResult(
                                session_id=session_id,
                                is_final=False,
                                text="[SPEECH_STARTED]",
                                confidence=1.0,
                                timestamp=None,
                                processing_time_ms=0
                            )
                            yield streaming_result
                            continue
                        elif result.type == 'UtteranceEnd':
                            # End of utterance detected
                            streaming_result = StreamingResult(
                                session_id=session_id,
                                is_final=True,
                                text="[UTTERANCE_END]",
                                confidence=1.0,
                                timestamp=None,
                                processing_time_ms=0
                            )
                            yield streaming_result
                            continue
                    
                    # Parse transcription result
                    if hasattr(result, 'channel') and result.channel and hasattr(result.channel, 'alternatives') and result.channel.alternatives:
                        alternative = result.channel.alternatives[0]
                        
                        # Extract metadata
                        metadata = {
                            "is_final": getattr(result, 'is_final', False),
                            "speech_final": getattr(result, 'speech_final', False)
                        }
                        
                        # Add duration if available
                        if hasattr(result, 'duration'):
                            metadata["duration"] = result.duration
                            
                        # Add start time if available
                        if hasattr(result, 'start'):
                            metadata["start"] = result.start
                        
                        # Extract words if available
                        if hasattr(alternative, 'words') and alternative.words:
                            metadata["words"] = [
                                {
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end,
                                    "confidence": getattr(word, 'confidence', 1.0)
                                }
                                for word in alternative.words
                            ]
                        
                        # Create streaming result
                        streaming_result = StreamingResult(
                            session_id=session_id,
                            is_final=metadata["is_final"],
                            text=alternative.transcript,
                            confidence=getattr(alternative, 'confidence', 1.0),
                            timestamp=None,  # Will be set by session manager
                            processing_time_ms=0
                        )
                        
                        yield streaming_result
                        
                except asyncio.TimeoutError:
                    # Check if we should continue waiting
                    if not session["connected"] and results_queue.empty():
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error processing streaming result: {e}")
                    break
                    
        except Exception as e:
            raise ProviderError(f"Failed to get streaming results: {e}", "deepgram")
            
    async def stop_streaming(self, session_id: str) -> None:
        """Stop streaming session and cleanup"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            return  # Session already cleaned up or never existed
            
        session = self._streaming_sessions[session_id]
        
        try:
            # Send finalize message before closing
            if session["connected"]:
                await self.finalize_streaming(session_id)
                await asyncio.sleep(0.1)  # Give time for final results
                
            # Close connection
            if session["connection"]:
                session["connection"].finish()
                session["connected"] = False
                
            # Clean up session
            del self._streaming_sessions[session_id]
            logger.info(f"Stopped Deepgram streaming session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error stopping streaming session {session_id}: {e}")
            # Still clean up the session
            if session_id in self._streaming_sessions:
                del self._streaming_sessions[session_id]
                
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """Estimate cost for Deepgram transcription"""
        # Deepgram pricing: ~$0.0059 per minute for Nova-2 model
        minutes = duration_seconds / 60
        return minutes * 0.0059
        
    async def test_connection(self) -> bool:
        """Test Deepgram API connection"""
        try:
            # Test with a small audio buffer
            test_audio = b'\x00' * 1024  # 1KB of silence
            result = await self.transcribe_audio(
                test_audio, 
                audio_format="pcm",
                sample_rate=16000
            )
            return True
        except Exception:
            return False
            
    async def cleanup(self) -> None:
        """Cleanup all streaming sessions"""
        if hasattr(self, '_streaming_sessions'):
            session_ids = list(self._streaming_sessions.keys())
            for session_id in session_ids:
                await self.stop_streaming(session_id)