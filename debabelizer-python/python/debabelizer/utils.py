"""Utility functions for working with Debabelizer"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_audio_file(file_path: str) -> bytes:
    """Load audio file as bytes for processing
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Audio data as bytes
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file cannot be read
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if not path.is_file():
        raise IOError(f"Path is not a file: {file_path}")
        
    return path.read_bytes()


def get_audio_format_from_extension(file_path: str) -> str:
    """Determine audio format from file extension
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Audio format string (e.g., 'wav', 'mp3', 'flac')
    """
    ext = Path(file_path).suffix.lower().lstrip('.')
    
    # Map common extensions to format names
    format_map = {
        'wav': 'wav',
        'mp3': 'mp3', 
        'flac': 'flac',
        'ogg': 'ogg',
        'opus': 'opus',
        'm4a': 'aac',
        'aac': 'aac',
        'webm': 'webm',
    }
    
    return format_map.get(ext, ext)


def create_config_from_env() -> Dict[str, Any]:
    """Create configuration dictionary from environment variables
    
    Returns:
        Configuration dictionary for VoiceProcessor
    """
    config = {}
    
    # Core preferences
    if stt_provider := os.getenv('DEBABELIZER_STT_PROVIDER'):
        config.setdefault('preferences', {})['stt_provider'] = stt_provider
        
    if tts_provider := os.getenv('DEBABELIZER_TTS_PROVIDER'):
        config.setdefault('preferences', {})['tts_provider'] = tts_provider
        
    if auto_select := os.getenv('DEBABELIZER_AUTO_SELECT'):
        config.setdefault('preferences', {})['auto_select'] = auto_select.lower() == 'true'
        
    if optimize_for := os.getenv('DEBABELIZER_OPTIMIZE_FOR'):
        config.setdefault('preferences', {})['optimize_for'] = optimize_for
    
    # Provider configurations
    provider_configs = {
        'soniox': {
            'api_key': os.getenv('SONIOX_API_KEY'),
            'model': os.getenv('SONIOX_MODEL', 'telephony'),
            'include_speaker': os.getenv('SONIOX_INCLUDE_SPEAKER', 'false').lower() == 'true',
        },
        'elevenlabs': {
            'api_key': os.getenv('ELEVENLABS_API_KEY'),
            'voice_id': os.getenv('ELEVENLABS_VOICE_ID', '21m00Tcm4TlvDq8ikWAM'),
            'model': os.getenv('ELEVENLABS_MODEL', 'eleven_monolingual_v1'),
        },
        'openai': {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'tts_model': os.getenv('OPENAI_TTS_MODEL', 'tts-1'),
            'tts_voice': os.getenv('OPENAI_TTS_VOICE', 'alloy'),
        },
        'deepgram': {
            'api_key': os.getenv('DEEPGRAM_API_KEY'),
            'model': os.getenv('DEEPGRAM_MODEL', 'nova-2'),
            'language': os.getenv('DEEPGRAM_LANGUAGE', 'en-US'),
        },
        'google': {
            'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            'project_id': os.getenv('GOOGLE_PROJECT_ID'),
            'api_key': os.getenv('GOOGLE_API_KEY'),
        },
        'azure': {
            'api_key': os.getenv('AZURE_SPEECH_KEY'),
            'region': os.getenv('AZURE_SPEECH_REGION'),
        },
    }
    
    # Only include provider configs that have required credentials
    for provider, provider_config in provider_configs.items():
        if any(v for v in provider_config.values() if v):
            config[provider] = {k: v for k, v in provider_config.items() if v is not None}
    
    return config


def create_synthesis_options(
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    pitch: Optional[float] = None,
    volume: Optional[float] = None,
    format: Optional[str] = None,
) -> "SynthesisOptions":
    """Create SynthesisOptions with the given parameters
    
    Args:
        voice: Voice ID or name to use
        speed: Speaking speed (0.25-4.0, 1.0 = normal)
        pitch: Pitch adjustment (-20.0 to 20.0 semitones, 0.0 = normal)
        volume: Volume level (0.0-1.0, 1.0 = normal)
        format: Audio format ('mp3', 'wav', 'opus', etc.)
        
    Returns:
        SynthesisOptions instance
    """
    from . import SynthesisOptions
    return SynthesisOptions(
        voice=voice,
        speed=speed,
        pitch=pitch,
        volume=volume,
        format=format,
    )