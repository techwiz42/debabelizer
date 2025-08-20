"""
Audio format detection and utilities
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def detect_audio_format(file_path: Path) -> str:
    """
    Detect audio format from file extension or header
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Audio format string
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    # Map file extensions to format names
    format_mapping = {
        ".wav": "wav",
        ".mp3": "mp3", 
        ".flac": "flac",
        ".ogg": "ogg",
        ".m4a": "m4a",
        ".aac": "aac",
        ".webm": "webm",
        ".mulaw": "mulaw",
        ".pcm": "pcm"
    }
    
    detected_format = format_mapping.get(extension)
    if detected_format:
        logger.debug(f"Detected format from extension: {detected_format}")
        return detected_format
        
    # Try to detect from file header if extension is unknown
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
            
        if header.startswith(b'RIFF') and b'WAVE' in header:
            logger.debug("Detected WAV format from header")
            return "wav"
        elif header.startswith(b'ID3') or b'ID3' in header or header.startswith(b'\xff\xfb'):
            logger.debug("Detected MP3 format from header")
            return "mp3"
        elif header.startswith(b'fLaC'):
            logger.debug("Detected FLAC format from header")
            return "flac"
        elif header.startswith(b'OggS'):
            logger.debug("Detected OGG format from header")
            return "ogg"
            
    except Exception as e:
        logger.warning(f"Could not read file header: {e}")
        
    # Default fallback
    logger.warning(f"Could not detect format for {file_path}, defaulting to WAV")
    return "wav"


def get_sample_rate_from_wav(file_path: Path) -> Optional[int]:
    """
    Extract sample rate from WAV file header
    
    Args:
        file_path: Path to WAV file
        
    Returns:
        Sample rate in Hz or None if not determinable
    """
    try:
        with open(file_path, 'rb') as f:
            # Skip to sample rate field in WAV header
            f.seek(24)  # Sample rate is at byte 24-27
            sample_rate_bytes = f.read(4)
            
            if len(sample_rate_bytes) == 4:
                # Convert little-endian bytes to integer
                sample_rate = int.from_bytes(sample_rate_bytes, byteorder='little')
                logger.debug(f"Detected sample rate: {sample_rate} Hz")
                return sample_rate
                
    except Exception as e:
        logger.warning(f"Could not extract sample rate from {file_path}: {e}")
        
    return None


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def estimate_audio_duration(file_path: Path, sample_rate: Optional[int] = None) -> Optional[float]:
    """
    Estimate audio duration from file size (rough approximation)
    
    Args:
        file_path: Path to audio file
        sample_rate: Sample rate in Hz (will try to detect if not provided)
        
    Returns:
        Estimated duration in seconds or None
    """
    try:
        file_size = file_path.stat().st_size
        format_type = detect_audio_format(file_path)
        
        if format_type == "wav":
            # For WAV files, we can be more accurate
            if sample_rate is None:
                sample_rate = get_sample_rate_from_wav(file_path)
                
            if sample_rate:
                # Assume 16-bit mono audio (2 bytes per sample)
                # Subtract 44 bytes for WAV header
                audio_bytes = file_size - 44
                samples = audio_bytes / 2
                duration = samples / sample_rate
                return duration
                
        # For other formats, use rough estimates
        elif format_type == "mp3":
            # MP3: roughly 1MB per minute at 128kbps
            duration = (file_size / (1024 * 1024)) * 60
            return duration
        elif format_type == "flac":
            # FLAC: roughly 20-30MB per minute (lossless)
            duration = (file_size / (25 * 1024 * 1024)) * 60  
            return duration
            
    except Exception as e:
        logger.warning(f"Could not estimate duration for {file_path}: {e}")
        
    return None


def validate_audio_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate audio file and return metadata
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with file metadata and validation results
    """
    file_path = Path(file_path)
    
    result = {
        "exists": file_path.exists(),
        "size_bytes": 0,
        "format": None,
        "sample_rate": None,
        "estimated_duration": None,
        "is_valid": False,
        "errors": []
    }
    
    if not result["exists"]:
        result["errors"].append("File does not exist")
        return result
        
    try:
        result["size_bytes"] = file_path.stat().st_size
        
        if result["size_bytes"] == 0:
            result["errors"].append("File is empty")
            return result
            
        result["format"] = detect_audio_format(file_path)
        
        if result["format"] == "wav":
            result["sample_rate"] = get_sample_rate_from_wav(file_path)
            
        result["estimated_duration"] = estimate_audio_duration(
            file_path, result["sample_rate"]
        )
        
        # Basic validation
        if result["size_bytes"] < 100:  # Less than 100 bytes
            result["errors"].append("File suspiciously small")
        elif result["size_bytes"] > 100 * 1024 * 1024:  # More than 100MB
            result["errors"].append("File suspiciously large")
            
        result["is_valid"] = len(result["errors"]) == 0
        
    except Exception as e:
        result["errors"].append(f"Validation error: {e}")
        
    return result