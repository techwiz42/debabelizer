"""
Soniox Speech-to-Text Provider for Debabelizer

Features:
- Real-time streaming transcription
- Automatic language detection for 60+ languages  
- Native telephony support (mulaw)
- Token-level streaming with configurable latency
"""

import asyncio
import json
import logging
import uuid
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
import websockets
from datetime import datetime

from ..base import STTProvider, TranscriptionResult, StreamingResult
from ..base.exceptions import (
    ProviderError, AuthenticationError, ConnectionError, 
    StreamingError, UnsupportedFormatError
)
from ...utils.audio import AudioConverter

logger = logging.getLogger(__name__)


class SonioxSTTProvider(STTProvider):
    """Soniox Speech-to-Text Provider"""
    
    # Soniox supported languages (60+ languages)
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
        "ar", "hi", "th", "vi", "tl", "nl", "pl", "tr", "sv", "da",
        "no", "fi", "cs", "sk", "hu", "ro", "bg", "hr", "sr", "sl",
        "et", "lv", "lt", "mt", "ga", "cy", "eu", "ca", "gl", "is",
        "mk", "sq", "bs", "me", "az", "kk", "ky", "uz", "tg", "mn",
        "ka", "hy", "he", "fa", "ur", "bn", "ta", "te", "ml", "kn",
        "gu", "pa", "or", "as", "mr", "ne", "si", "my", "km", "lo"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = "stt-rt-preview", **config):
        # Remove api_key from config to avoid duplicate argument error
        config_copy = config.copy()
        config_copy.pop('api_key', None)  # Remove if present
        super().__init__(api_key=api_key, **config_copy)
        self.model = model
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._receive_tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}  # Per-session locks
        self.audio_converter = AudioConverter()
        
    @property
    def name(self) -> str:
        return "soniox"
        
    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()
        
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
        """Transcribe an audio file using Soniox"""
        
        # Read audio file
        try:
            with open(file_path, 'rb') as f:
                # Skip WAV header if present (basic detection)
                header = f.read(44)
                audio_data = f.read()
                
                # If it's a WAV file, we already skipped the header
                if not header.startswith(b'RIFF'):
                    # Not a WAV file, include the "header" as audio data
                    audio_data = header + audio_data
                    
        except Exception as e:
            raise ProviderError(f"Failed to read audio file: {e}", "soniox")
            
        # Use streaming for file transcription (more robust)
        session_id = await self.start_streaming(
            audio_format="pcm_s16le",  # Assume 16-bit PCM for files
            sample_rate=16000,
            language=language,
            language_hints=language_hints,
            **kwargs
        )
        
        try:
            # Send audio data in chunks
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self.stream_audio(session_id, chunk)
                
            # Signal that no more audio is expected
            if session_id in self.sessions:
                async with self.sessions[session_id]["lock"]:
                    self.sessions[session_id]["has_pending_audio"] = False
            
            # Collect results
            final_result = None
            async for result in self.get_streaming_results(session_id):
                if result.result.is_final:
                    final_result = result.result
                    
            return final_result or TranscriptionResult(
                text="", 
                language_detected="unknown", 
                confidence=0.0
            )
            
        finally:
            await self.stop_streaming(session_id)
            
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
        
        # Map common formats to Soniox formats
        format_mapping = {
            "wav": "pcm_s16le",
            "pcm": "pcm_s16le", 
            "mulaw": "mulaw",
            "mp3": "mp3",
            "flac": "flac"
        }
        
        soniox_format = format_mapping.get(audio_format.lower())
        if not soniox_format:
            raise UnsupportedFormatError(
                f"Audio format '{audio_format}' not supported", 
                "soniox"
            )
            
        # Use streaming for audio transcription
        session_id = await self.start_streaming(
            audio_format=soniox_format,
            sample_rate=sample_rate,
            language=language,
            language_hints=language_hints,
            **kwargs
        )
        
        try:
            await self.stream_audio(session_id, audio_data)
            
            # Signal that no more audio is expected
            if session_id in self.sessions:
                async with self.sessions[session_id]["lock"]:
                    self.sessions[session_id]["has_pending_audio"] = False
            
            # Collect final result
            final_result = None
            async for result in self.get_streaming_results(session_id):
                if result.result.is_final:
                    final_result = result.result
                    
            return final_result or TranscriptionResult(
                text="", 
                language_detected="unknown", 
                confidence=0.0
            )
            
        finally:
            await self.stop_streaming(session_id)
            
    async def start_streaming(
        self,
        audio_format: str = "mulaw",
        sample_rate: int = 8000,
        language: Optional[str] = None,
        language_hints: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Start a streaming transcription session"""
        
        # Handle format conversion for provider-agnostic usage
        native_soniox_formats = ["pcm_s16le", "mulaw", "mp3", "flac"]
        format_mapping = {
            "wav": "pcm_s16le",
            "pcm": "pcm_s16le", 
            "mulaw": "mulaw",
            "mp3": "mp3",
            "flac": "flac"
        }
        
        # Check if we need format conversion
        needs_conversion = False
        soniox_format = format_mapping.get(audio_format.lower(), audio_format.lower())
        
        if soniox_format not in native_soniox_formats:
            # Check if we can convert this format
            if audio_format.lower() in ["webm", "ogg", "m4a", "aac"]:
                if self.audio_converter.ffmpeg_available:
                    logger.info(f"Will convert {audio_format} to PCM for Soniox")
                    soniox_format = "pcm_s16le"
                    needs_conversion = True
                else:
                    raise UnsupportedFormatError(
                        f"Audio format '{audio_format}' requires FFmpeg for conversion, but FFmpeg is not available", 
                        "soniox"
                    )
            else:
                raise UnsupportedFormatError(
                    f"Audio format '{audio_format}' not supported by Soniox. Supported formats: wav, pcm, mulaw, mp3, flac, webm (with FFmpeg)", 
                    "soniox"
                )
        
        session_id = str(uuid.uuid4())
        
        try:
            # Set up authorization header
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            logger.info(f"🔌 Starting Soniox streaming session: {session_id}")
            
            # Connect to Soniox WebSocket endpoint with stable timeouts
            websocket = await websockets.connect(
                "wss://stt-rt.soniox.com/transcribe-websocket",
                additional_headers=headers,
                ping_interval=None,  # Disable ping/pong - let server handle keepalive
                ping_timeout=None,   # Disable ping timeout
                close_timeout=30     # Longer close timeout
            )
            
            # Send Soniox configuration 
            config = {
                "api_key": self.api_key,
                "audio_format": soniox_format,
                "sample_rate": sample_rate,
                "num_channels": 1,  # Mono audio
                "model": self.model,
                "enable_language_identification": True,
                "include_nonfinal": True
            }
            
            # Add language hints if provided
            if language_hints:
                config["language_hints"] = language_hints
            elif language:
                config["language_hints"] = [language]
                
            await websocket.send(json.dumps(config))
            
            # Wait for initial response to confirm connection with longer timeout
            try:
                initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)  # Increased from 5.0
                initial_data = json.loads(initial_message)
                
                # Check for immediate errors
                if "error" in initial_data:
                    error_msg = initial_data.get("error", "Unknown error")
                    raise ProviderError(f"Soniox configuration error: {error_msg}", "soniox")
                    
                logger.info(f"✅ Soniox session {session_id} configured successfully: {initial_data}")
                
            except asyncio.TimeoutError:
                logger.warning(f"No initial response from Soniox for session {session_id} within 10s, proceeding anyway")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid initial response from Soniox for session {session_id}: {e}")
            except Exception as e:
                logger.error(f"Error getting initial response from Soniox for session {session_id}: {e}")
            
            # Store session info with lock
            session_lock = asyncio.Lock()
            self._session_locks[session_id] = session_lock
            
            self.sessions[session_id] = {
                "websocket": websocket,
                "results_queue": asyncio.Queue(),
                "is_active": True,
                "is_connected": True,
                "config": config,
                "needs_conversion": needs_conversion,
                "input_format": audio_format.lower(),
                "soniox_format": soniox_format,
                "reconnect_attempts": 0,
                "max_reconnect_attempts": 3,
                "lock": session_lock,  # Store reference to lock in session
                "last_audio_time": None,  # Track when audio was last sent
                "has_pending_audio": False  # Track if we expect more audio
            }
            
            # Start receive task
            self._receive_tasks[session_id] = asyncio.create_task(
                self._receive_loop(session_id)
            )
            
            logger.info(f"✅ Soniox streaming session started: {session_id}")
            return session_id
            
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise AuthenticationError(f"Invalid API key: {e}", "soniox")
            raise ConnectionError(f"Failed to connect to Soniox: {e}", "soniox")
            
    async def stream_audio(self, session_id: str, audio_chunk: bytes) -> None:
        """Send audio chunk to streaming session"""
        
        if session_id not in self.sessions:
            raise StreamingError(
                f"Session {session_id} not found. Available sessions: {list(self.sessions.keys())}", 
                "soniox"
            )
            
        session = self.sessions[session_id]
        session_lock = session["lock"]
        
        # Check session state under lock
        async with session_lock:
            if not session["is_active"]:
                raise StreamingError(
                    f"Session {session_id} has been terminated", 
                    "soniox"
                )
                
            # If not connected, try to reconnect
            if not session["is_connected"]:
                if session_id not in self._reconnect_tasks or self._reconnect_tasks[session_id].done():
                    logger.info(f"Starting reconnection for session {session_id}")
                    self._reconnect_tasks[session_id] = asyncio.create_task(
                        self._reconnect_session(session_id)
                    )
        
        # Wait for reconnection to complete (outside lock to avoid deadlock)
        if not session["is_connected"]:
            for _ in range(10):  # Wait up to 1 second
                await asyncio.sleep(0.1)
                async with session_lock:
                    if session["is_connected"]:
                        break
            
            async with session_lock:
                if not session["is_connected"]:
                    logger.warning(f"Session {session_id} reconnection still in progress, will retry audio send")
                    # Don't raise error immediately, let reconnection continue
                    return
                websocket = session["websocket"]
        else:
            async with session_lock:
                websocket = session["websocket"]
            
        try:
            # Convert audio if needed
            audio_to_send = audio_chunk
            if session.get("needs_conversion", False):
                logger.debug(f"Converting {len(audio_chunk)} bytes from {session['input_format']} to {session['soniox_format']}")
                audio_to_send = await self.audio_converter.convert_audio(
                    audio_chunk,
                    session["input_format"],
                    session["soniox_format"].replace("_s16le", ""),  # Convert pcm_s16le to pcm
                    sample_rate=16000,
                    channels=1
                )
                logger.debug(f"Converted to {len(audio_to_send)} bytes")
            
            logger.debug(f"Sending {len(audio_to_send)} bytes to Soniox session {session_id}")
            
            # Update session activity tracking
            async with session_lock:
                session["last_audio_time"] = time.time()
                session["has_pending_audio"] = True
            
            await websocket.send(audio_to_send)
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed while sending audio to session {session_id}: {e}")
            async with session_lock:
                session["is_connected"] = False
                # Mark for reconnection but don't raise error
                if session_id not in self._reconnect_tasks or self._reconnect_tasks[session_id].done():
                    self._reconnect_tasks[session_id] = asyncio.create_task(
                        self._reconnect_session(session_id)
                    )
        except Exception as e:
            logger.error(f"Failed to send audio to Soniox session {session_id}: {e}")
            # Only raise error for critical failures, not connection issues
            if isinstance(e, (websockets.exceptions.WebSocketException, ConnectionError, OSError)):
                logger.warning(f"Network error sending audio, will attempt reconnection")
                async with session_lock:
                    session["is_connected"] = False
            else:
                raise StreamingError(f"Failed to send audio: {e}", "soniox")
            
    async def get_streaming_results(
        self, 
        session_id: str
    ) -> AsyncGenerator[StreamingResult, None]:
        """Get streaming transcription results"""
        
        if session_id not in self.sessions:
            raise StreamingError(f"Session {session_id} not found", "soniox")
            
        session = self.sessions[session_id]
        results_queue = session["results_queue"]
        
        while session["is_active"]:
            try:
                # Wait for result with timeout
                result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                yield result
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error getting streaming results: {e}")
                break
                
    async def stop_streaming(self, session_id: str) -> None:
        """Stop streaming session and cleanup"""
        
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        session_lock = session["lock"]
        
        # Mark session as inactive under lock
        async with session_lock:
            session["is_active"] = False
            session["is_connected"] = False
            session["has_pending_audio"] = False  # Clear pending audio flag
        
        # Cancel receive task
        if session_id in self._receive_tasks:
            self._receive_tasks[session_id].cancel()
            try:
                await self._receive_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self._receive_tasks[session_id]
            
        # Cancel reconnect task
        if session_id in self._reconnect_tasks:
            self._reconnect_tasks[session_id].cancel()
            try:
                await self._reconnect_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self._reconnect_tasks[session_id]
            
        # Close WebSocket
        try:
            await session["websocket"].close()
        except Exception:
            pass
            
        # Clean up session and locks
        del self.sessions[session_id]
        if session_id in self._session_locks:
            del self._session_locks[session_id]
        logger.info(f"🔌 Stopped Soniox streaming session: {session_id}")
        
    async def _reconnect_session(self, session_id: str) -> bool:
        """Attempt to reconnect a session with exponential backoff"""
        
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        session_lock = session["lock"]
        
        for attempt in range(session["max_reconnect_attempts"]):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{session['max_reconnect_attempts']} for session {session_id}")
                
                # Close old websocket if still open
                try:
                    await session["websocket"].close()
                except:
                    pass
                
                # Wait with exponential backoff
                if attempt > 0:
                    wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                    await asyncio.sleep(wait_time)
                
                # Create new WebSocket connection
                headers = {"Authorization": f"Bearer {self.api_key}"}
                websocket = await websockets.connect(
                    "wss://stt-rt.soniox.com/transcribe-websocket",
                    additional_headers=headers,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=30
                )
                
                # Send configuration
                await websocket.send(json.dumps(session["config"]))
                
                # Wait for initial response
                try:
                    initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    initial_data = json.loads(initial_message)
                    
                    if "error" in initial_data:
                        logger.error(f"Soniox configuration error on reconnect: {initial_data['error']}")
                        continue
                        
                except asyncio.TimeoutError:
                    logger.warning(f"No initial response on reconnect for session {session_id}, proceeding")
                
                # Update session under lock
                async with session_lock:
                    session["websocket"] = websocket
                    session["is_connected"] = True
                    session["reconnect_attempts"] = attempt + 1
                
                # Don't restart receive loop here - let the existing loop continue
                # The loop should detect the new websocket and continue
                
                logger.info(f"✅ Successfully reconnected session {session_id}")
                return True
                
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed for session {session_id}: {e}")
                continue
        
        logger.error(f"Failed to reconnect session {session_id} after {session['max_reconnect_attempts']} attempts")
        async with session_lock:
            session["is_connected"] = False
        return False
    
    async def _receive_loop(self, session_id: str):
        """Internal loop to receive and process messages from Soniox"""
        
        session = self.sessions[session_id]
        session_lock = session["lock"]
        results_queue = session["results_queue"]
        
        logger.debug(f"Starting receive loop for Soniox session {session_id}")
        
        try:
            while session["is_active"]:
                # Always get fresh websocket reference under lock
                async with session_lock:
                    if not session["is_active"]:
                        break
                    websocket = session["websocket"]
                    is_connected = session["is_connected"]
                # Skip if not connected
                if not is_connected:
                    await asyncio.sleep(0.1)
                    continue
                    
                try:
                    # Use timeout for recv to allow periodic health checks
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    logger.debug(f"Received message from Soniox: {message[:200]}...")
                    data = json.loads(message)
                    
                    # Handle Soniox response format
                    if "tokens" in data:
                        tokens = data.get("tokens", [])
                        
                        if tokens:
                            # Extract language and common metadata once
                            detected_language = "unknown"
                            for token in tokens:
                                if isinstance(token, dict) and "language" in token:
                                    detected_language = token["language"]
                                    break
                            
                            # Process each token individually for word-level streaming
                            for token in tokens:
                                if isinstance(token, dict) and "text" in token:
                                    token_text = token["text"]
                                    token_confidence = token.get("confidence", 0.0)
                                    token_is_final = token.get("is_final", False)
                                    
                                    # Create streaming result for each token/word
                                    streaming_result = StreamingResult(
                                        session_id=session_id,
                                        is_final=token_is_final,
                                        text=token_text,
                                        confidence=token_confidence,
                                        timestamp=datetime.now(),
                                        processing_time_ms=data.get("final_audio_proc_ms", 0)
                                    )
                                    
                                    await results_queue.put(streaming_result)
                                    
                                elif isinstance(token, str):
                                    # Handle string tokens
                                    streaming_result = StreamingResult(
                                        session_id=session_id,
                                        is_final=False,  # String tokens are usually interim
                                        text=token,
                                        confidence=0.0,
                                        timestamp=datetime.now(),
                                        processing_time_ms=data.get("final_audio_proc_ms", 0)
                                    )
                                    
                                    await results_queue.put(streaming_result)
                            
                            # Also send complete utterance result for backward compatibility
                            text_parts = []
                            for token in tokens:
                                if isinstance(token, dict) and "text" in token:
                                    text_parts.append(token["text"])
                                elif isinstance(token, str):
                                    text_parts.append(token)
                            
                            if text_parts:
                                full_text = "".join(text_parts)
                                
                                # Calculate average confidence
                                confidences = []
                                for token in tokens:
                                    if isinstance(token, dict) and "confidence" in token:
                                        confidences.append(token["confidence"])
                                
                                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                                
                                # Determine if final (Soniox uses is_final field in tokens)
                                is_final = any(
                                    token.get("is_final", False) 
                                    for token in tokens 
                                    if isinstance(token, dict)
                                )
                                
                                # Send complete utterance as a separate result (for backward compatibility)
                                complete_result = StreamingResult(
                                    session_id=session_id,
                                    is_final=is_final,
                                    text=full_text,
                                    confidence=avg_confidence,
                                    timestamp=datetime.now(),
                                    processing_time_ms=data.get("final_audio_proc_ms", 0)
                                )
                                
                                await results_queue.put(complete_result)
                                
                    elif "error" in data:
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"❌ Soniox error: {error_msg}")
                        # Don't raise immediately - try to continue
                        continue
                    
                except (websockets.exceptions.ConnectionClosed, 
                        websockets.exceptions.ConnectionClosedError, 
                        websockets.exceptions.ConnectionClosedOK) as e:
                    logger.warning(f"Soniox connection closed for session {session_id}: {e}")
                    
                    # Update connection state under lock
                    async with session_lock:
                        session["is_connected"] = False
                    
                    # For normal close (1000), only reconnect if there's recent audio activity
                    if hasattr(e, 'code') and e.code == 1000:
                        current_time = time.time()
                        should_reconnect = False
                        
                        async with session_lock:
                            last_audio_time = session.get("last_audio_time")
                            has_pending_audio = session.get("has_pending_audio", False)
                            
                            # Only reconnect if audio was sent recently (within 60 seconds) or we expect more audio
                            if last_audio_time and (current_time - last_audio_time) < 60:
                                should_reconnect = True
                                logger.info(f"Normal close detected for session {session_id} with recent audio activity, attempting reconnection")
                            elif has_pending_audio:
                                should_reconnect = True
                                logger.info(f"Normal close detected for session {session_id} with pending audio, attempting reconnection")
                            else:
                                logger.info(f"Normal close detected for session {session_id} with no recent audio activity, ending session gracefully")
                        
                        if should_reconnect:
                            # Start reconnection task if not already running
                            async with session_lock:
                                if session_id not in self._reconnect_tasks or self._reconnect_tasks[session_id].done():
                                    self._reconnect_tasks[session_id] = asyncio.create_task(
                                        self._reconnect_session(session_id)
                                    )
                            
                            # Wait for reconnection to complete
                            for _ in range(30):  # Wait up to 3 seconds
                                await asyncio.sleep(0.1)
                                async with session_lock:
                                    if session["is_connected"]:
                                        logger.info(f"Reconnected session {session_id}, resuming receive loop")
                                        break
                            else:
                                logger.error(f"Reconnection failed for session {session_id}, exiting receive loop")
                                break
                            continue
                        else:
                            # No recent activity, end session gracefully
                            async with session_lock:
                                session["is_active"] = False
                            break
                    else:
                        # For abnormal closes, exit the receive loop
                        break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Soniox message for session {session_id}: {e}")
                    # Continue processing other messages instead of breaking
                    continue
                except asyncio.TimeoutError:
                    # Timeout is normal - just continue the loop to allow health check
                    logger.debug(f"Soniox receive timeout for session {session_id} (normal keepalive)")
                    continue
                except Exception as e:
                    logger.error(f"Error in Soniox receive loop for session {session_id}: {e}", exc_info=True)
                    
                    # Classify error types for appropriate handling
                    if isinstance(e, (websockets.exceptions.WebSocketException, ConnectionError, OSError)):
                        logger.warning(f"Network/connection error in session {session_id}, marking as disconnected")
                        session["is_connected"] = False
                        # Try to reconnect for network errors
                        if session_id not in self._reconnect_tasks or self._reconnect_tasks[session_id].done():
                            self._reconnect_tasks[session_id] = asyncio.create_task(
                                self._reconnect_session(session_id)
                            )
                        break
                    elif isinstance(e, (PermissionError, AuthenticationError)):
                        logger.error(f"Authentication error in session {session_id}, terminating session")
                        session["is_active"] = False
                        break
                    else:
                        # For other errors (parsing, etc.), log but continue
                        logger.warning(f"Non-critical error in session {session_id}, continuing: {e}")
                        continue
        except Exception as e:
            logger.error(f"Fatal error in Soniox receive loop for session {session_id}: {e}", exc_info=True)
            session["is_connected"] = False
        finally:
            logger.info(f"Receive loop ended for session {session_id}")
            # Only mark session as inactive if explicitly stopped, not on connection issues
            if session.get("is_connected", True):  # If we're still "connected", this was a clean stop
                session["is_active"] = False
                
    def get_cost_estimate(self, duration_seconds: float) -> float:
        """Estimate cost for Soniox transcription"""
        # Soniox pricing: approximately $0.0013 per minute
        minutes = duration_seconds / 60
        return minutes * 0.0013
        
    async def cleanup(self) -> None:
        """Cleanup all sessions and resources"""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.stop_streaming(session_id)
        self.is_connected = False