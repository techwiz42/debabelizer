"""
Pytest configuration for Debabelizer tests

Provides common fixtures and test configuration.
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import sys

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from debabelizer import DebabelizerConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide a test configuration with mock API keys"""
    return DebabelizerConfig({
        "soniox": {
            "api_key": "test_soniox_key",
            "model": "stt-rt-preview"
        },
        "deepgram": {
            "api_key": "test_deepgram_key", 
            "model": "nova-2",
            "language": "en-US"
        },
        "elevenlabs": {
            "api_key": "test_elevenlabs_key",
            "default_voice_id": "test_voice_id",
            "model": "eleven_turbo_v2_5"
        },
        "openai": {
            "api_key": "test_openai_key",
            "model": "whisper-1"
        },
        "azure": {
            "api_key": "test_azure_key",
            "region": "test_region"
        }
    })


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        # Create a minimal WAV file header (0.2 seconds of audio for OpenAI compatibility)
        data_size = 6400  # 0.2 seconds * 16000 Hz * 2 bytes = 6400 bytes
        file_size = 36 + data_size  # Header is 44 bytes, data is remaining
        wav_header = (
            b'RIFF'
            + file_size.to_bytes(4, byteorder='little')
            + b'WAVE'
            + b'fmt '
            + (16).to_bytes(4, byteorder='little')
            + (1).to_bytes(2, byteorder='little')   # PCM
            + (1).to_bytes(2, byteorder='little')   # Mono
            + (16000).to_bytes(4, byteorder='little')  # 16kHz
            + (32000).to_bytes(4, byteorder='little')  # Byte rate
            + (2).to_bytes(2, byteorder='little')   # Block align
            + (16).to_bytes(2, byteorder='little')  # 16-bit
            + b'data'
            + data_size.to_bytes(4, byteorder='little')  # Data size
        )
        
        tmp_file.write(wav_header + b'\x00' * data_size)  # Add 0.2 seconds of silence
        tmp_file.flush()
        
        yield Path(tmp_file.name)
        
        # Cleanup
        try:
            os.unlink(tmp_file.name)
        except FileNotFoundError:
            pass


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(params=["en", "es", "de", "ru", "zh", "tl", "ko", "vi"])
def test_language(request):
    """Parametrized fixture for the eight test languages"""
    language_map = {
        "en": "english",
        "es": "spanish",  
        "de": "german",
        "ru": "russian",
        "zh": "chinese",
        "tl": "tagalog",
        "ko": "korean",
        "vi": "vietnamese"
    }
    
    return {
        "code": request.param,
        "name": language_map[request.param]
    }


@pytest.fixture
def mock_audio_files(temp_directory, test_language):
    """Create mock audio files for each test language"""
    audio_files = {}
    
    for lang_code, lang_name in [
        ("en", "english"), ("es", "spanish"), ("de", "german"), ("ru", "russian"),
        ("zh", "chinese"), ("tl", "tagalog"), ("ko", "korean"), ("vi", "vietnamese")
    ]:
        file_path = temp_directory / f"{lang_name}_sample.wav"
        
        # Create minimal WAV file
        wav_header = (
            b'RIFF' + (1044).to_bytes(4, byteorder='little') + b'WAVE'
            + b'fmt ' + (16).to_bytes(4, byteorder='little')
            + (1).to_bytes(2, byteorder='little')   # PCM
            + (1).to_bytes(2, byteorder='little')   # Mono
            + (16000).to_bytes(4, byteorder='little')  # 16kHz
            + (32000).to_bytes(4, byteorder='little')
            + (2).to_bytes(2, byteorder='little')
            + (16).to_bytes(2, byteorder='little')
            + b'data' + (1000).to_bytes(4, byteorder='little')
        )
        
        with open(file_path, 'wb') as f:
            f.write(wav_header + b'\x00' * 1000)
            
        audio_files[lang_code] = file_path
    
    return audio_files


class TestLanguages:
    """Constants for the eight test languages from Thanotopolis"""
    
    LANGUAGES = {
        "en": "english",    # English
        "es": "spanish",    # Spanish  
        "de": "german",     # German
        "ru": "russian",    # Russian
        "zh": "chinese",    # Chinese
        "tl": "tagalog",    # Tagalog (Filipino)
        "ko": "korean",     # Korean
        "vi": "vietnamese"  # Vietnamese
    }
    
    @classmethod
    def get_language_codes(cls):
        """Get list of language codes"""
        return list(cls.LANGUAGES.keys())
    
    @classmethod
    def get_language_names(cls):
        """Get list of language names"""
        return list(cls.LANGUAGES.values())
    
    @classmethod
    def get_language_pairs(cls):
        """Get list of (code, name) tuples"""
        return list(cls.LANGUAGES.items())


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network connectivity"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
            
        # Mark slow tests
        if "concurrent" in item.name or "streaming" in item.name or "long_text" in item.name:
            item.add_marker(pytest.mark.slow)
            
        # Mark network tests
        if "api" in item.name.lower() or "provider" in item.name.lower():
            item.add_marker(pytest.mark.network)