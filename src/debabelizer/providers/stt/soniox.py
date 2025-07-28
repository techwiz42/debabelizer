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
    
    def __init__(self, api_key: str, model: str = "stt-rt-preview", **config):
        super().__init__(api_key, **config)
        self.model = model
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._receive_tasks: Dict[str, asyncio.Task] = {}
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
            
            # Connect to Soniox WebSocket endpoint
            websocket = await websockets.connect(
                "wss://stt-rt.soniox.com/transcribe-websocket",
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
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
            
            # Wait for initial response to confirm connection
            try:
                initial_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                initial_data = json.loads(initial_message)
                
                # Check for immediate errors
                if "error" in initial_data:
                    error_msg = initial_data.get("error", "Unknown error")
                    raise ProviderError(f"Soniox configuration error: {error_msg}", "soniox")
                    
                logger.debug(f"Soniox initial response: {initial_data}")
                
            except asyncio.TimeoutError:
                logger.warning(f"No initial response from Soniox for session {session_id}, proceeding anyway")
            except json.JSONDecodeError:
                logger.warning(f"Invalid initial response from Soniox for session {session_id}")
            
            # Store session info
            self.sessions[session_id] = {
                "websocket": websocket,
                "results_queue": asyncio.Queue(),
                "is_active": True,
                "config": config,
                "needs_conversion": needs_conversion,
                "input_format": audio_format.lower(),
                "soniox_format": soniox_format
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
        if not session["is_active"]:
            # Check if the receive task is still running
            receive_task = self._receive_tasks.get(session_id)
            task_status = "unknown"
            if receive_task:
                if receive_task.done():
                    try:
                        receive_task.result()
                        task_status = "completed normally"
                    except Exception as e:
                        task_status = f"completed with error: {e}"
                else:
                    task_status = "still running"
            
            raise StreamingError(
                f"Session {session_id} is not active. Receive task status: {task_status}", 
                "soniox"
            )
            
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
            await session["websocket"].send(audio_to_send)
        except Exception as e:
            logger.error(f"Failed to send audio to Soniox session {session_id}: {e}")
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
        session["is_active"] = False
        
        # Cancel receive task
        if session_id in self._receive_tasks:
            self._receive_tasks[session_id].cancel()
            try:
                await self._receive_tasks[session_id]
            except asyncio.CancelledError:
                pass
            del self._receive_tasks[session_id]
            
        # Close WebSocket
        try:
            await session["websocket"].close()
        except Exception:
            pass
            
        # Remove session
        del self.sessions[session_id]
        logger.info(f"🔌 Stopped Soniox streaming session: {session_id}")
        
    async def _receive_loop(self, session_id: str):
        """Internal loop to receive and process messages from Soniox"""
        
        session = self.sessions[session_id]
        websocket = session["websocket"]
        results_queue = session["results_queue"]
        
        logger.debug(f"Starting receive loop for Soniox session {session_id}")
        
        while session["is_active"]:
            try:
                message = await websocket.recv()
                logger.debug(f"Received message from Soniox: {message[:200]}...")
                data = json.loads(message)
                
                # Handle Soniox response format
                if "tokens" in data:
                    tokens = data.get("tokens", [])
                    
                    if tokens:
                        # Extract text from tokens
                        text_parts = []
                        for token in tokens:
                            if isinstance(token, dict) and "text" in token:
                                text_parts.append(token["text"])
                            elif isinstance(token, str):
                                text_parts.append(token)
                        
                        if text_parts:
                            text = "".join(text_parts)
                            
                            # Extract language from tokens
                            detected_language = "unknown"
                            if tokens:
                                for token in tokens:
                                    if isinstance(token, dict) and "language" in token:
                                        detected_language = token["language"]
                                        break
                            
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
                            
                            # Create transcription result
                            result = TranscriptionResult(
                                text=text,
                                language_detected=detected_language,
                                confidence=avg_confidence,
                                is_final=is_final,
                                tokens=tokens,
                                metadata={
                                    "processing_time_ms": data.get("final_audio_proc_ms", 0),
                                    "total_audio_proc_ms": data.get("total_audio_proc_ms", 0)
                                }
                            )
                            
                            # Create streaming result
                            streaming_result = StreamingResult(
                                result=result,
                                session_id=session_id,
                                timestamp=datetime.now(),
                                processing_time_ms=data.get("final_audio_proc_ms", 0)
                            )
                            
                            await results_queue.put(streaming_result)
                            
                elif "error" in data:
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"❌ Soniox error: {error_msg}")
                    raise ProviderError(error_msg, "soniox")
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Soniox connection closed for session {session_id}: {e}")
                session["is_active"] = False
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Soniox message for session {session_id}: {e}")
                # Continue processing other messages
            except Exception as e:
                logger.error(f"Error in Soniox receive loop for session {session_id}: {e}", exc_info=True)
                # Don't immediately mark as inactive unless it's a critical error
                if isinstance(e, (ConnectionError, OSError)):
                    session["is_active"] = False
                    break
                
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