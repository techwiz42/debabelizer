"""
Unit tests for Debabelizer utility modules

Tests for audio processing, format detection, and session management utilities.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, Mock
from pathlib import Path
import sys
import tempfile
import os
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer.utils.audio import AudioConverter
from debabelizer.utils.formats import (
    detect_audio_format, get_sample_rate_from_wav, format_duration,
    estimate_audio_duration, validate_audio_file
)
from debabelizer.core.session import SessionManager, SessionInfo


class TestAudioConverter:
    """Test cases for AudioConverter utility"""

    def test_initialization(self):
        """Test AudioConverter initialization"""
        converter = AudioConverter()
        assert hasattr(converter, 'ffmpeg_available')
        assert isinstance(converter.ffmpeg_available, bool)

    @patch('subprocess.run')
    def test_ffmpeg_availability_check_success(self, mock_run):
        """Test FFmpeg availability check when FFmpeg is available"""
        mock_run.return_value = Mock(returncode=0)
        
        converter = AudioConverter()
        assert converter.ffmpeg_available is True

    @patch('subprocess.run')
    def test_ffmpeg_availability_check_failure(self, mock_run):
        """Test FFmpeg availability check when FFmpeg is not available"""
        mock_run.side_effect = FileNotFoundError()
        
        converter = AudioConverter()
        assert converter.ffmpeg_available is False

    @pytest.mark.asyncio
    async def test_convert_audio_without_ffmpeg(self):
        """Test audio conversion when FFmpeg is not available"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=False):
            converter = AudioConverter()
            
            # Same format should work
            result = await converter.convert_audio(
                b"audio_data", "wav", "wav"
            )
            assert result == b"audio_data"
            
            # Different formats should fail
            with pytest.raises(RuntimeError, match="FFmpeg not available"):
                await converter.convert_audio(
                    b"audio_data", "wav", "mp3"
                )

    @pytest.mark.asyncio
    async def test_convert_audio_with_ffmpeg_success(self):
        """Test successful audio conversion with FFmpeg"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"converted_audio", b"")
            mock_process.returncode = 0
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                result = await converter.convert_audio(
                    b"input_audio", "wav", "mp3"
                )
                
                assert result == b"converted_audio"

    @pytest.mark.asyncio
    async def test_convert_audio_ffmpeg_error(self):
        """Test audio conversion when FFmpeg fails"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"FFmpeg error message")
            mock_process.returncode = 1
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
                    await converter.convert_audio(
                        b"input_audio", "wav", "mp3"
                    )

    @pytest.mark.asyncio
    async def test_convert_to_telephony_format(self):
        """Test conversion to telephony format (mulaw, 8kHz, mono)"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"mulaw_audio", b"")
            mock_process.returncode = 0
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                result = await converter.convert_to_telephony_format(
                    b"input_audio", "wav"
                )
                
                assert result == b"mulaw_audio"
                
                # Verify correct FFmpeg parameters were used
                call_args = mock_exec.call_args[0]
                assert "-ar" in call_args
                assert "8000" in call_args
                assert "-ac" in call_args
                assert "1" in call_args
                assert "-f" in call_args
                assert "mulaw" in call_args

    @pytest.mark.asyncio
    async def test_convert_from_telephony_format(self):
        """Test conversion from telephony format"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"wav_audio", b"")
            mock_process.returncode = 0
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                result = await converter.convert_from_telephony_format(
                    b"mulaw_audio", "wav"
                )
                
                assert result == b"wav_audio"

    def test_get_audio_info_success(self):
        """Test getting audio file information"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_result = Mock()
            mock_result.stdout = json.dumps({
                "streams": [{
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "16000",
                    "channels": "1"
                }],
                "format": {
                    "duration": "5.0",
                    "bit_rate": "256000",
                    "size": "160000",
                    "format_name": "wav"
                }
            })
            mock_result.returncode = 0
            
            with patch('subprocess.run', return_value=mock_result):
                info = converter.get_audio_info(Path("test.wav"))
                
                assert info["codec"] == "pcm_s16le"
                assert info["sample_rate"] == 16000
                assert info["channels"] == 1
                assert info["duration"] == 5.0
                assert info["bitrate"] == 256000
                assert info["size"] == 160000
                assert info["format"] == "wav"

    def test_get_audio_info_no_ffmpeg(self):
        """Test getting audio info when FFmpeg is not available"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=False):
            converter = AudioConverter()
            
            info = converter.get_audio_info(Path("test.wav"))
            assert "error" in info
            assert "FFmpeg not available" in info["error"]

    @pytest.mark.asyncio
    async def test_extract_audio_chunk(self):
        """Test extracting audio chunk from file"""
        with patch.object(AudioConverter, '_check_ffmpeg', return_value=True):
            converter = AudioConverter()
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"audio_chunk", b"")
            mock_process.returncode = 0
            
            with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
                result = await converter.extract_audio_chunk(
                    Path("test.wav"), start_time=1.0, duration=2.0
                )
                
                assert result == b"audio_chunk"
                
                # Verify correct parameters
                call_args = mock_exec.call_args[0]
                assert "-ss" in call_args
                assert "1.0" in call_args
                assert "-t" in call_args
                assert "2.0" in call_args

    def test_create_silence_pcm(self):
        """Test creating PCM silence"""
        converter = AudioConverter()
        
        # 1 second of silence at 16kHz = 16000 samples * 2 bytes = 32000 bytes
        silence = converter.create_silence(1.0, sample_rate=16000, format_type="pcm")
        
        expected_size = 16000 * 2  # 16000 samples * 2 bytes per sample
        assert len(silence) == expected_size
        assert silence == b'\x00\x00' * 16000

    def test_create_silence_mulaw(self):
        """Test creating mulaw silence"""
        converter = AudioConverter()
        
        # 0.5 seconds of silence at 8kHz = 4000 samples
        silence = converter.create_silence(0.5, sample_rate=8000, format_type="mulaw")
        
        expected_size = 4000  # 4000 samples * 1 byte per sample
        assert len(silence) == expected_size
        assert silence == b'\x7f' * 4000

    def test_detect_silence_basic(self):
        """Test basic silence detection"""
        converter = AudioConverter()
        
        # Create audio with silence in the middle
        # 16-bit samples: loud, quiet, loud
        loud_sample = (32000).to_bytes(2, byteorder='little', signed=True)
        quiet_sample = (100).to_bytes(2, byteorder='little', signed=True)
        
        # 1000 loud + 1000 quiet + 1000 loud samples
        audio_data = loud_sample * 1000 + quiet_sample * 1000 + loud_sample * 1000
        
        silent_segments = converter.detect_silence(
            audio_data, threshold=0.01, min_duration=0.01
        )
        
        # Should detect the quiet middle section
        assert len(silent_segments) >= 1


class TestFormatUtils:
    """Test cases for format detection utilities"""

    def test_detect_audio_format_by_extension(self):
        """Test audio format detection by file extension"""
        test_cases = [
            ("test.wav", "wav"),
            ("test.mp3", "mp3"),
            ("test.flac", "flac"),
            ("test.ogg", "ogg"),
            ("test.m4a", "m4a"),
            ("test.aac", "aac"),
            ("test.webm", "webm"),
            ("test.mulaw", "mulaw"),
            ("test.pcm", "pcm"),
        ]
        
        for file_path, expected_format in test_cases:
            detected_format = detect_audio_format(Path(file_path))
            assert detected_format == expected_format

    def test_detect_audio_format_by_header(self):
        """Test audio format detection by file header"""
        test_cases = [
            (b'RIFF\x00\x00\x00\x00WAVE', "wav"),
            (b'ID3\x03\x00\x00\x00', "mp3"),
            (b'\xff\xfbID3', "mp3"),  # Alternative MP3 header
            (b'fLaC\x00\x00\x00\x22', "flac"),
            (b'OggS\x00\x02\x00\x00', "ogg"),
        ]
        
        for header_bytes, expected_format in test_cases:
            with tempfile.NamedTemporaryFile(suffix=".unknown") as tmp_file:
                tmp_file.write(header_bytes + b'\x00' * 100)  # Pad with zeros
                tmp_file.flush()
                
                detected_format = detect_audio_format(Path(tmp_file.name))
                assert detected_format == expected_format

    def test_detect_audio_format_fallback(self):
        """Test audio format detection fallback to WAV"""
        with tempfile.NamedTemporaryFile(suffix=".unknown") as tmp_file:
            tmp_file.write(b'UNKNOWN_HEADER_DATA')
            tmp_file.flush()
            
            detected_format = detect_audio_format(Path(tmp_file.name))
            assert detected_format == "wav"  # Should fallback to WAV

    def test_get_sample_rate_from_wav(self):
        """Test extracting sample rate from WAV file header"""
        # Create a minimal WAV header with 16000 Hz sample rate
        wav_header = (
            b'RIFF'
            + (44).to_bytes(4, byteorder='little')  # File size
            + b'WAVE'
            + b'fmt '
            + (16).to_bytes(4, byteorder='little')  # Format chunk size
            + (1).to_bytes(2, byteorder='little')   # Audio format (PCM)
            + (1).to_bytes(2, byteorder='little')   # Number of channels
            + (16000).to_bytes(4, byteorder='little')  # Sample rate
            + (32000).to_bytes(4, byteorder='little')  # Byte rate
            + (2).to_bytes(2, byteorder='little')   # Block align
            + (16).to_bytes(2, byteorder='little')  # Bits per sample
            + b'data'
            + (0).to_bytes(4, byteorder='little')   # Data chunk size
        )
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            tmp_file.write(wav_header)
            tmp_file.flush()
            
            sample_rate = get_sample_rate_from_wav(Path(tmp_file.name))
            assert sample_rate == 16000

    def test_get_sample_rate_from_invalid_wav(self):
        """Test sample rate extraction from invalid WAV file"""
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            tmp_file.write(b'INVALID_WAV_DATA')
            tmp_file.flush()
            
            sample_rate = get_sample_rate_from_wav(Path(tmp_file.name))
            assert sample_rate is None

    def test_format_duration(self):
        """Test duration formatting"""
        test_cases = [
            (30.5, "30.5s"),
            (90.0, "1m 30.0s"),
            (3661.5, "1h 1m 1.5s"),
            (7200.0, "2h 0m 0.0s"),
        ]
        
        for seconds, expected in test_cases:
            formatted = format_duration(seconds)
            assert formatted == expected

    def test_estimate_audio_duration(self):
        """Test audio duration estimation"""
        # Create a mock WAV file with known parameters
        file_size = 32044  # 44 byte header + 32000 bytes of audio data
        
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            # Write minimal WAV header
            wav_header = (
                b'RIFF'
                + (file_size - 8).to_bytes(4, byteorder='little')
                + b'WAVE'
                + b'fmt '
                + (16).to_bytes(4, byteorder='little')
                + (1).to_bytes(2, byteorder='little')
                + (1).to_bytes(2, byteorder='little')
                + (16000).to_bytes(4, byteorder='little')  # 16kHz sample rate
                + (32000).to_bytes(4, byteorder='little')
                + (2).to_bytes(2, byteorder='little')
                + (16).to_bytes(2, byteorder='little')
                + b'data'
                + (32000).to_bytes(4, byteorder='little')
            )
            
            tmp_file.write(wav_header + b'\x00' * 32000)  # Add audio data
            tmp_file.flush()
            
            duration = estimate_audio_duration(Path(tmp_file.name))
            # 32000 bytes / 2 bytes per sample / 16000 Hz = 1.0 second
            assert abs(duration - 1.0) < 0.1

    def test_validate_audio_file(self):
        """Test audio file validation"""
        # Test with valid file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 1000)
            tmp_file.flush()
            
            result = validate_audio_file(Path(tmp_file.name))
            
            assert result["exists"] is True
            assert result["size_bytes"] > 0
            assert result["format"] == "wav"
            assert result["is_valid"] is True
            
            # Clean up
            os.unlink(tmp_file.name)
        
        # Test with non-existent file
        result = validate_audio_file(Path("nonexistent.wav"))
        assert result["exists"] is False
        assert result["is_valid"] is False
        assert "File does not exist" in result["errors"]


class TestSessionManager:
    """Test cases for SessionManager"""

    @pytest.fixture
    def session_manager(self):
        """Create a SessionManager for testing"""
        return SessionManager(cleanup_interval=1)  # Short interval for testing

    def test_initialization(self, session_manager):
        """Test SessionManager initialization"""
        assert len(session_manager.sessions) == 0
        assert session_manager.cleanup_interval == 1
        assert session_manager._running is False

    def test_create_session(self, session_manager):
        """Test creating a new session"""
        session_info = session_manager.create_session(
            session_id="test_session_123",
            session_type="stt",
            provider="soniox",
            metadata={"test_key": "test_value"}
        )
        
        assert isinstance(session_info, SessionInfo)
        assert session_info.session_id == "test_session_123"
        assert session_info.session_type == "stt"
        assert session_info.provider == "soniox"
        assert session_info.metadata["test_key"] == "test_value"
        
        # Verify session is stored
        assert "test_session_123" in session_manager.sessions
        # Note: _running may be False if no event loop is available during testing

    def test_get_session(self, session_manager):
        """Test retrieving session information"""
        # Create a session
        session_manager.create_session("test_session", "tts", "elevenlabs")
        
        # Retrieve it
        session_info = session_manager.get_session("test_session")
        assert session_info is not None
        assert session_info.session_id == "test_session"
        
        # Test non-existent session
        assert session_manager.get_session("nonexistent") is None

    def test_update_session_activity(self, session_manager):
        """Test updating session activity timestamp"""
        # Create a session
        session_info = session_manager.create_session("test_session", "stt", "soniox")
        original_activity = session_info.last_activity
        
        # Wait a bit and update activity
        import time
        time.sleep(0.01)
        session_manager.update_session_activity("test_session")
        
        # Activity should be updated
        updated_info = session_manager.get_session("test_session")
        assert updated_info.last_activity > original_activity

    def test_end_session(self, session_manager):
        """Test ending a session"""
        # Create a session
        session_manager.create_session("test_session", "stt", "soniox")
        assert "test_session" in session_manager.sessions
        
        # End the session
        result = session_manager.end_session("test_session")
        assert result is True
        assert "test_session" not in session_manager.sessions
        
        # Test ending non-existent session
        result = session_manager.end_session("nonexistent")
        assert result is False

    def test_get_active_sessions(self, session_manager):
        """Test getting list of active sessions"""
        # Initially empty
        assert len(session_manager.get_active_sessions()) == 0
        
        # Create multiple sessions
        session_manager.create_session("session_1", "stt", "soniox")
        session_manager.create_session("session_2", "tts", "elevenlabs")
        session_manager.create_session("session_3", "stt", "deepgram")
        
        active_sessions = session_manager.get_active_sessions()
        assert len(active_sessions) == 3
        
        session_ids = [s.session_id for s in active_sessions]
        assert "session_1" in session_ids
        assert "session_2" in session_ids
        assert "session_3" in session_ids

    def test_get_sessions_by_type(self, session_manager):
        """Test filtering sessions by type"""
        # Create sessions of different types
        session_manager.create_session("stt_1", "stt", "soniox")
        session_manager.create_session("stt_2", "stt", "deepgram")
        session_manager.create_session("tts_1", "tts", "elevenlabs")
        
        stt_sessions = session_manager.get_sessions_by_type("stt")
        tts_sessions = session_manager.get_sessions_by_type("tts")
        
        assert len(stt_sessions) == 2
        assert len(tts_sessions) == 1
        
        stt_ids = [s.session_id for s in stt_sessions]
        assert "stt_1" in stt_ids
        assert "stt_2" in stt_ids

    def test_get_sessions_by_provider(self, session_manager):
        """Test filtering sessions by provider"""
        # Create sessions with different providers
        session_manager.create_session("session_1", "stt", "soniox")
        session_manager.create_session("session_2", "stt", "soniox")
        session_manager.create_session("session_3", "tts", "elevenlabs")
        
        soniox_sessions = session_manager.get_sessions_by_provider("soniox")
        elevenlabs_sessions = session_manager.get_sessions_by_provider("elevenlabs")
        
        assert len(soniox_sessions) == 2
        assert len(elevenlabs_sessions) == 1

    def test_get_session_stats(self, session_manager):
        """Test getting session statistics"""
        # Test with no sessions
        stats = session_manager.get_session_stats()
        assert stats["total_sessions"] == 0
        assert stats["by_type"] == {}
        assert stats["by_provider"] == {}
        
        # Create some sessions
        session_manager.create_session("session_1", "stt", "soniox")
        session_manager.create_session("session_2", "stt", "deepgram")
        session_manager.create_session("session_3", "tts", "elevenlabs")
        
        stats = session_manager.get_session_stats()
        assert stats["total_sessions"] == 3
        assert stats["by_type"]["stt"] == 2
        assert stats["by_type"]["tts"] == 1
        assert stats["by_provider"]["soniox"] == 1
        assert stats["by_provider"]["deepgram"] == 1
        assert stats["by_provider"]["elevenlabs"] == 1
        
        # Should have oldest and newest session info
        assert "oldest_session" in stats
        assert "newest_session" in stats
        assert "id" in stats["oldest_session"]
        assert "age_seconds" in stats["oldest_session"]

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self, session_manager):
        """Test automatic cleanup of stale sessions"""
        # Create a session and manually set old activity time
        session_info = session_manager.create_session("stale_session", "stt", "soniox")
        
        # Make the session appear stale (older than 30 minutes)
        session_info.last_activity = datetime.now() - timedelta(minutes=35)
        
        # Run cleanup
        await session_manager._cleanup_stale_sessions()
        
        # Session should be removed
        assert "stale_session" not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_cleanup_all_sessions(self, session_manager):
        """Test cleaning up all sessions"""
        # Create multiple sessions
        session_manager.create_session("session_1", "stt", "soniox")
        session_manager.create_session("session_2", "tts", "elevenlabs")
        
        assert len(session_manager.sessions) == 2
        assert session_manager._running is True
        
        # Cleanup all
        await session_manager.cleanup_all_sessions()
        
        assert len(session_manager.sessions) == 0
        assert session_manager._running is False

    def test_session_manager_len(self, session_manager):
        """Test SessionManager __len__ method"""
        assert len(session_manager) == 0
        
        session_manager.create_session("session_1", "stt", "soniox")
        assert len(session_manager) == 1
        
        session_manager.create_session("session_2", "tts", "elevenlabs")
        assert len(session_manager) == 2
        
        session_manager.end_session("session_1")
        assert len(session_manager) == 1