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
        """Prepare Deepgram API options"""
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
            "numerals": kwargs.get("numerals", True),
            "search": kwargs.get("search", []),
            "keywords": kwargs.get("keywords", []),
            "detect_language": kwargs.get("detect_language", False)
        }
        
        # Add confidence scores
        if kwargs.get("confidence", True):
            options["confidence"] = True
            
        # Add word-level timestamps
        if kwargs.get("words", True):
            options["words"] = True
            
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
            
            # Handle language detection
            if language_hints and not language:
                base_options["detect_language"] = True
                
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
            if audio_format.lower() in ["pcm", "raw"]:
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
        """Start a streaming transcription session"""
        
        try:
            # Dynamic import
            from deepgram import DeepgramClient, LiveOptions
            
            # Generate session ID
            import uuid
            session_id = str(uuid.uuid4())
            
            # Initialize client
            deepgram = DeepgramClient(self.api_key)
            
            # Prepare options
            base_options = self._prepare_options(language, **kwargs)
            
            # Add streaming-specific options
            base_options.update({
                "encoding": "linear16" if audio_format.lower() in ["pcm", "raw"] else audio_format,
                "sample_rate": sample_rate,
                "channels": kwargs.get("channels", 1),
                "interim_results": kwargs.get("interim_results", True),
                "endpointing": kwargs.get("endpointing", 300),  # 300ms silence
                "vad_events": kwargs.get("vad_events", True)
            })
            
            options = LiveOptions(**base_options)
            
            # Create connection
            dg_connection = deepgram.listen.asyncwebsocket.v("1")
            
            # Store connection for this session
            if not hasattr(self, '_streaming_sessions'):
                self._streaming_sessions = {}
                
            self._streaming_sessions[session_id] = {
                "connection": dg_connection,
                "options": options,
                "results": asyncio.Queue(),
                "connected": False
            }
            
            # Setup event handlers
            async def on_message(result, **kwargs):
                if session_id in self._streaming_sessions:
                    await self._streaming_sessions[session_id]["results"].put(result)
                    
            async def on_error(error, **kwargs):
                logger.error(f"Deepgram streaming error for session {session_id}: {error}")
                
            # Connect handlers
            dg_connection.on("message", on_message)
            dg_connection.on("error", on_error)
            
            # Start connection
            if await dg_connection.start(options):
                self._streaming_sessions[session_id]["connected"] = True
                logger.info(f"Started Deepgram streaming session: {session_id}")
                return session_id
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
            await session["connection"].send(audio_chunk)
            
        except Exception as e:
            raise ProviderError(f"Failed to send audio to session {session_id}: {e}", "deepgram")
            
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results"""
        
        if not hasattr(self, '_streaming_sessions') or session_id not in self._streaming_sessions:
            raise ProviderError(f"Streaming session {session_id} not found", "deepgram")
            
        session = self._streaming_sessions[session_id]
        results_queue = session["results"]
        
        try:
            while session["connected"]:
                try:
                    # Wait for results with timeout
                    result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                    
                    # Parse Deepgram result
                    if hasattr(result, 'channel') and result.channel.alternatives:
                        alternative = result.channel.alternatives[0]
                        
                        # Create streaming result
                        streaming_result = StreamingResult(
                            session_id=session_id,
                            is_final=getattr(result, 'is_final', False),
                            text=alternative.transcript,
                            confidence=getattr(alternative, 'confidence', 1.0),
                            timestamp=None,  # Will be set by session manager
                            processing_time_ms=0
                        )
                        
                        yield streaming_result
                        
                except asyncio.TimeoutError:
                    # Continue waiting for more results
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
            # Close connection
            if session["connected"]:
                await session["connection"].finish()
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