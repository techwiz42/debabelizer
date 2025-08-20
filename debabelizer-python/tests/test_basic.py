"""
Basic tests for the new Python wrapper around the Rust debabelizer
"""
import pytest
import debabelizer


def test_imports():
    """Test that all expected classes can be imported"""
    # Test the main classes are available
    assert hasattr(debabelizer, 'VoiceProcessor')
    assert hasattr(debabelizer, 'DebabelizerConfig')
    assert hasattr(debabelizer, 'AudioFormat')
    assert hasattr(debabelizer, 'Voice')
    assert hasattr(debabelizer, 'TranscriptionResult')
    assert hasattr(debabelizer, 'SynthesisResult')
    assert hasattr(debabelizer, 'StreamingResult')
    assert hasattr(debabelizer, 'WordTiming')
    
    # Test exception classes are available
    assert hasattr(debabelizer, 'ProviderError')
    assert hasattr(debabelizer, 'AuthenticationError')
    assert hasattr(debabelizer, 'RateLimitError')
    assert hasattr(debabelizer, 'ConfigurationError')
    
    # Test convenience function is available
    assert hasattr(debabelizer, 'create_processor')


def test_audio_format_creation():
    """Test AudioFormat creation matches original API"""
    # Test basic creation
    fmt = debabelizer.AudioFormat("wav", 16000)
    assert fmt.format == "wav"
    assert fmt.sample_rate == 16000
    assert fmt.channels == 1
    assert fmt.bit_depth == 16

    # Test with all parameters
    fmt2 = debabelizer.AudioFormat("mp3", 44100, 2, 24)
    assert fmt2.format == "mp3"
    assert fmt2.sample_rate == 44100
    assert fmt2.channels == 2
    assert fmt2.bit_depth == 24


def test_word_timing_creation():
    """Test WordTiming creation matches original API"""
    word = debabelizer.WordTiming("hello", 0.5, 1.0, 0.95)
    assert word.word == "hello"
    assert word.start_time == 0.5
    assert word.end_time == 1.0
    assert abs(word.confidence - 0.95) < 0.001  # Allow for floating point precision


def test_voice_creation():
    """Test Voice creation matches original API"""
    voice = debabelizer.Voice(
        voice_id="test-voice",
        name="Test Voice",
        language="en-US",
        gender="female",
        description="A test voice"
    )
    assert voice.voice_id == "test-voice"
    assert voice.name == "Test Voice"
    assert voice.language == "en-US"
    assert voice.description == "A test voice"


def test_config_creation():
    """Test DebabelizerConfig creation"""
    # Test empty config
    config = debabelizer.DebabelizerConfig()
    assert config is not None
    
    # Test config with dict (would need to be implemented properly)
    config_dict = {
        "soniox": {"api_key": "test-key"},
        "preferences": {"stt_provider": "soniox"}
    }
    config2 = debabelizer.DebabelizerConfig(config_dict)
    assert config2 is not None


def test_voice_processor_creation():
    """Test VoiceProcessor creation matches original API"""
    # Test basic creation
    processor = debabelizer.VoiceProcessor()
    assert processor is not None
    
    # Test creation with providers
    processor2 = debabelizer.VoiceProcessor(
        stt_provider="soniox",
        tts_provider="elevenlabs"
    )
    assert processor2 is not None
    
    # Test creation with config
    config = debabelizer.DebabelizerConfig()
    processor3 = debabelizer.VoiceProcessor(config=config)
    assert processor3 is not None


def test_create_processor_convenience_function():
    """Test the create_processor convenience function"""
    processor = debabelizer.create_processor(
        stt_provider="soniox",
        tts_provider="elevenlabs"
    )
    assert processor is not None
    assert isinstance(processor, debabelizer.VoiceProcessor)


def test_voice_processor_methods_exist():
    """Test that VoiceProcessor has all expected methods from original API"""
    processor = debabelizer.VoiceProcessor()
    
    # Core transcription methods
    assert hasattr(processor, 'transcribe_file')
    assert hasattr(processor, 'transcribe_audio')
    assert hasattr(processor, 'transcribe_chunk')
    
    # Core synthesis methods
    assert hasattr(processor, 'synthesize')
    assert hasattr(processor, 'synthesize_text')
    
    # Streaming methods
    assert hasattr(processor, 'start_streaming_transcription')
    assert hasattr(processor, 'stream_audio')
    assert hasattr(processor, 'stop_streaming_transcription')
    
    # Provider management
    assert hasattr(processor, 'set_stt_provider')
    assert hasattr(processor, 'set_tts_provider')
    assert hasattr(processor, 'get_available_voices')
    
    # Utility methods
    assert hasattr(processor, 'get_usage_stats')
    assert hasattr(processor, 'reset_usage_stats')
    assert hasattr(processor, 'test_providers')
    assert hasattr(processor, 'cleanup')
    
    # Properties
    assert hasattr(processor, 'stt_provider_name')
    assert hasattr(processor, 'tts_provider_name')


def test_api_compatibility_with_original():
    """Test that the API is compatible with the original Python module"""
    # This test verifies that all the main exports match the original __all__ list
    expected_exports = [
        "VoiceProcessor",
        "DebabelizerConfig", 
        "STTProvider",
        "TTSProvider",
        "TranscriptionResult",
        "SynthesisResult", 
        "StreamingResult",
        "Voice",
        "AudioFormat",
        "ProviderError",
        "WordTiming",
        "AuthenticationError",
        "RateLimitError", 
        "ConfigurationError",
        "create_processor",
    ]
    
    for export in expected_exports:
        assert hasattr(debabelizer, export), f"Missing export: {export}"
    
    # Test that __all__ contains the expected exports
    if hasattr(debabelizer, '__all__'):
        for export in expected_exports:
            assert export in debabelizer.__all__, f"Export {export} not in __all__"


if __name__ == "__main__":
    pytest.main([__file__])