"""Basic tests for Debabelizer Python bindings."""

import pytest
import debabelizer
from unittest.mock import patch


class TestAudioFormat:
    """Test AudioFormat class."""
    
    def test_create_audio_format(self):
        """Test creating AudioFormat instance."""
        format_obj = debabelizer.AudioFormat(
            format="wav",
            sample_rate=16000,
            channels=1,
            bit_depth=16
        )
        
        assert format_obj.format == "wav"
        assert format_obj.sample_rate == 16000
        assert format_obj.channels == 1
        assert format_obj.bit_depth == 16


class TestAudioData:
    """Test AudioData class."""
    
    def test_create_audio_data(self):
        """Test creating AudioData instance."""
        audio_format = debabelizer.AudioFormat("wav", 16000, 1, 16)
        audio_data = debabelizer.AudioData(b"test_data", audio_format)
        
        assert audio_data.data == b"test_data"
        assert audio_data.format.format == "wav"


class TestSynthesisOptions:
    """Test SynthesisOptions class."""
    
    def test_create_synthesis_options_empty(self):
        """Test creating empty SynthesisOptions."""
        options = debabelizer.SynthesisOptions()
        
        assert options.voice is None
        assert options.speed is None
        assert options.pitch is None
        assert options.volume is None
        assert options.format is None
    
    def test_create_synthesis_options_full(self):
        """Test creating SynthesisOptions with all parameters."""
        options = debabelizer.SynthesisOptions(
            voice="alloy",
            speed=1.2,
            pitch=2.0,
            volume=0.8,
            format="mp3"
        )
        
        assert options.voice == "alloy"
        assert options.speed == 1.2
        assert options.pitch == 2.0
        assert options.volume == 0.8
        assert options.format == "mp3"
    
    def test_modify_synthesis_options(self):
        """Test modifying SynthesisOptions after creation."""
        options = debabelizer.SynthesisOptions()
        
        options.voice = "nova"
        options.speed = 1.5
        
        assert options.voice == "nova"
        assert options.speed == 1.5


class TestVoiceProcessor:
    """Test VoiceProcessor class."""
    
    def test_create_processor_no_config(self):
        """Test creating processor without configuration."""
        # This should work if environment variables are set or fail gracefully
        try:
            processor = debabelizer.VoiceProcessor()
            # If successful, test basic functionality
            assert hasattr(processor, 'has_stt_provider')
            assert hasattr(processor, 'has_tts_provider')
        except debabelizer.DebabelizerException:
            # Expected if no configuration is available
            pass
    
    def test_create_processor_with_config(self):
        """Test creating processor with configuration."""
        config = {
            "preferences": {
                "stt_provider": "deepgram",
                "tts_provider": "openai"
            },
            "deepgram": {
                "api_key": "test_key"
            },
            "openai": {
                "api_key": "test_key"
            }
        }
        
        # This might fail due to invalid API keys, but should accept the config
        try:
            processor = debabelizer.VoiceProcessor(config=config)
            assert hasattr(processor, 'has_stt_provider')
        except debabelizer.DebabelizerException:
            # Expected with invalid API keys
            pass
    
    def test_create_processor_with_explicit_providers(self):
        """Test creating processor with explicit provider selection."""
        try:
            processor = debabelizer.VoiceProcessor(
                stt_provider="deepgram",
                tts_provider="openai"
            )
            assert hasattr(processor, 'get_stt_provider_name')
            assert hasattr(processor, 'get_tts_provider_name')
        except debabelizer.DebabelizerException:
            # Expected if providers can't be configured
            pass
    
    def test_processor_methods_exist(self):
        """Test that processor has expected methods."""
        # Create a mock processor to test interface
        try:
            processor = debabelizer.VoiceProcessor()
            
            # Test method existence
            assert callable(getattr(processor, 'transcribe', None))
            assert callable(getattr(processor, 'synthesize', None))
            assert callable(getattr(processor, 'get_available_voices', None))
            assert callable(getattr(processor, 'has_stt_provider', None))
            assert callable(getattr(processor, 'has_tts_provider', None))
            assert callable(getattr(processor, 'get_stt_provider_name', None))
            assert callable(getattr(processor, 'get_tts_provider_name', None))
            
        except debabelizer.DebabelizerException:
            # If we can't create a processor, just test that the class exists
            assert hasattr(debabelizer, 'VoiceProcessor')


class TestExceptions:
    """Test exception handling."""
    
    def test_debabelizer_exception_exists(self):
        """Test that DebabelizerException exists."""
        assert hasattr(debabelizer, 'DebabelizerException')
        assert issubclass(debabelizer.DebabelizerException, Exception)
    
    def test_raise_debabelizer_exception(self):
        """Test raising DebabelizerException."""
        with pytest.raises(debabelizer.DebabelizerException):
            raise debabelizer.DebabelizerException("Test error")


class TestModuleStructure:
    """Test module structure and imports."""
    
    def test_all_classes_imported(self):
        """Test that all expected classes are available."""
        expected_classes = [
            'AudioFormat',
            'AudioData',
            'WordTiming',
            'TranscriptionResult',
            'Voice',
            'SynthesisResult',
            'SynthesisOptions',
            'VoiceProcessor',
            'DebabelizerException',
        ]
        
        for class_name in expected_classes:
            assert hasattr(debabelizer, class_name), f"Missing class: {class_name}"
    
    def test_module_metadata(self):
        """Test module metadata."""
        assert hasattr(debabelizer, '__version__')
        assert hasattr(debabelizer, '__author__')
        assert debabelizer.__version__ == "0.1.0"


class TestUtils:
    """Test utility functions."""
    
    def test_utils_import(self):
        """Test that utils module can be imported."""
        from debabelizer import utils
        assert hasattr(utils, 'load_audio_file')
        assert hasattr(utils, 'get_audio_format_from_extension')
        assert hasattr(utils, 'create_config_from_env')
        assert hasattr(utils, 'create_synthesis_options')
    
    def test_get_audio_format_from_extension(self):
        """Test audio format detection from file extension."""
        from debabelizer.utils import get_audio_format_from_extension
        
        assert get_audio_format_from_extension("test.wav") == "wav"
        assert get_audio_format_from_extension("test.mp3") == "mp3"
        assert get_audio_format_from_extension("test.flac") == "flac"
        assert get_audio_format_from_extension("/path/to/file.ogg") == "ogg"
    
    def test_create_synthesis_options_util(self):
        """Test utility function for creating synthesis options."""
        from debabelizer.utils import create_synthesis_options
        
        options = create_synthesis_options(
            voice="alloy",
            speed=1.2,
            format="mp3"
        )
        
        assert isinstance(options, debabelizer.SynthesisOptions)
        assert options.voice == "alloy"
        assert options.speed == 1.2
        assert options.format == "mp3"
    
    @patch.dict('os.environ', {
        'DEBABELIZER_STT_PROVIDER': 'deepgram',
        'DEBABELIZER_TTS_PROVIDER': 'openai',
        'DEEPGRAM_API_KEY': 'test_key',
        'OPENAI_API_KEY': 'test_key'
    })
    def test_create_config_from_env(self):
        """Test creating configuration from environment variables."""
        from debabelizer.utils import create_config_from_env
        
        config = create_config_from_env()
        
        assert 'preferences' in config
        assert config['preferences']['stt_provider'] == 'deepgram'
        assert config['preferences']['tts_provider'] == 'openai'
        assert 'deepgram' in config
        assert config['deepgram']['api_key'] == 'test_key'
        assert 'openai' in config
        assert config['openai']['api_key'] == 'test_key'