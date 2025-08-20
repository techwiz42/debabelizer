"""
Audio processing utilities for Debabelizer
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)


class AudioConverter:
    """Audio format conversion and processing utilities"""
    
    def __init__(self):
        """Initialize audio converter"""
        self.ffmpeg_available = self._check_ffmpeg()
        
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            logger.debug("FFmpeg is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not available - format conversion limited")
            return False
            
    async def convert_audio(
        self,
        input_data: bytes,
        input_format: str,
        output_format: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None
    ) -> bytes:
        """
        Convert audio data between formats
        
        Args:
            input_data: Input audio data
            input_format: Input format (wav, mp3, etc.)
            output_format: Output format
            sample_rate: Target sample rate (optional)
            channels: Target number of channels (optional)
            
        Returns:
            Converted audio data
        """
        if not self.ffmpeg_available:
            if input_format == output_format:
                return input_data
            raise RuntimeError("FFmpeg not available for format conversion")
            
        # Build FFmpeg command
        cmd = ['ffmpeg', '-f', input_format, '-i', 'pipe:0']
        
        if sample_rate:
            cmd.extend(['-ar', str(sample_rate)])
            
        if channels:
            cmd.extend(['-ac', str(channels)])
            
        cmd.extend(['-f', output_format, 'pipe:1'])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input_data)
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")
                
            logger.debug(
                f"Converted {len(input_data)} bytes from {input_format} "
                f"to {output_format} ({len(stdout)} bytes output)"
            )
            
            return stdout
            
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {e}")
            
    async def convert_to_telephony_format(self, audio_data: bytes, input_format: str) -> bytes:
        """
        Convert audio to telephony format (mulaw, 8kHz, mono)
        
        Args:
            audio_data: Input audio data
            input_format: Input format
            
        Returns:
            Audio data in mulaw format
        """
        return await self.convert_audio(
            audio_data,
            input_format,
            "mulaw",
            sample_rate=8000,
            channels=1
        )
        
    async def convert_from_telephony_format(
        self, 
        audio_data: bytes, 
        output_format: str = "wav"
    ) -> bytes:
        """
        Convert from telephony format to standard format
        
        Args:
            audio_data: Mulaw audio data
            output_format: Target format
            
        Returns:
            Converted audio data
        """
        return await self.convert_audio(
            audio_data,
            "mulaw",
            output_format,
            sample_rate=16000,  # Upsample to 16kHz for better quality
            channels=1
        )
        
    def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get detailed audio file information using FFprobe
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        if not self.ffmpeg_available:
            return {"error": "FFmpeg not available"}
            
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', str(file_path)
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            
            import json
            metadata = json.loads(result.stdout)
            
            # Extract audio stream info
            audio_streams = [
                stream for stream in metadata.get('streams', [])
                if stream.get('codec_type') == 'audio'
            ]
            
            if not audio_streams:
                return {"error": "No audio streams found"}
                
            audio_stream = audio_streams[0]  # Use first audio stream
            format_info = metadata.get('format', {})
            
            return {
                "codec": audio_stream.get('codec_name'),
                "sample_rate": int(audio_stream.get('sample_rate', 0)),
                "channels": int(audio_stream.get('channels', 0)),
                "duration": float(format_info.get('duration', 0)),
                "bitrate": int(format_info.get('bit_rate', 0)),
                "size": int(format_info.get('size', 0)),
                "format": format_info.get('format_name')
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            return {"error": f"Failed to get audio info: {e}"}
            
    async def extract_audio_chunk(
        self,
        file_path: Path,
        start_time: float,
        duration: float,
        output_format: str = "wav"
    ) -> bytes:
        """
        Extract a chunk of audio from a file
        
        Args:
            file_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            output_format: Output format
            
        Returns:
            Audio chunk data
        """
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg not available")
            
        cmd = [
            'ffmpeg', '-i', str(file_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-f', output_format,
            'pipe:1'
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Audio extraction failed: {error_msg}")
                
            return stdout
            
        except Exception as e:
            raise RuntimeError(f"Audio chunk extraction failed: {e}")
            
    def create_silence(
        self, 
        duration_seconds: float, 
        sample_rate: int = 16000,
        format_type: str = "wav"
    ) -> bytes:
        """
        Create silent audio data
        
        Args:
            duration_seconds: Duration of silence
            sample_rate: Sample rate in Hz
            format_type: Audio format
            
        Returns:
            Silent audio data
        """
        if format_type == "pcm":
            # Create PCM silence (16-bit)
            samples = int(duration_seconds * sample_rate)
            silence_data = b'\x00\x00' * samples  # 16-bit silence
            return silence_data
        elif format_type == "mulaw":
            # Create mulaw silence
            samples = int(duration_seconds * sample_rate)
            silence_data = b'\x7f' * samples  # mulaw silence value
            return silence_data
        else:
            # For other formats, use FFmpeg
            if not self.ffmpeg_available:
                raise RuntimeError("FFmpeg required for non-PCM silence generation")
                
            # This would require async, so for now return basic PCM
            samples = int(duration_seconds * sample_rate)
            return b'\x00\x00' * samples
            
    def detect_silence(
        self, 
        audio_data: bytes, 
        threshold: float = 0.01,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Detect silent segments in audio (basic implementation)
        
        Args:
            audio_data: Audio data (assumed 16-bit PCM)
            threshold: Silence threshold (0.0 to 1.0)
            min_duration: Minimum silence duration to detect
            
        Returns:
            List of (start_time, end_time) tuples for silent segments
        """
        # Basic silence detection for 16-bit PCM
        # This is a simplified implementation
        
        silent_segments = []
        chunk_size = 2  # 16-bit = 2 bytes per sample
        sample_rate = 16000  # Assume 16kHz
        
        current_silence_start = None
        
        for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
            # Get 16-bit sample value
            sample = int.from_bytes(
                audio_data[i:i + chunk_size], 
                byteorder='little', 
                signed=True
            )
            
            # Normalize to 0.0-1.0 range
            normalized = abs(sample) / 32768.0
            
            current_time = i / (chunk_size * sample_rate)
            
            if normalized < threshold:
                # This sample is silent
                if current_silence_start is None:
                    current_silence_start = current_time
            else:
                # This sample is not silent
                if current_silence_start is not None:
                    silence_duration = current_time - current_silence_start
                    if silence_duration >= min_duration:
                        silent_segments.append((current_silence_start, current_time))
                    current_silence_start = None
                    
        # Handle silence that extends to end of audio
        if current_silence_start is not None:
            end_time = len(audio_data) / (chunk_size * sample_rate)
            silence_duration = end_time - current_silence_start
            if silence_duration >= min_duration:
                silent_segments.append((current_silence_start, end_time))
                
        return silent_segments