"""
Provider-specific exceptions for Debabelizer
"""


class ProviderError(Exception):
    """Base exception for all provider errors"""
    
    def __init__(self, message: str, provider: str = "unknown", code: str = None, **kwargs):
        super().__init__(message)
        self.provider = provider
        self.code = code
        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __str__(self):
        return self.args[0] if self.args else ""


class AuthenticationError(ProviderError):
    """Invalid API key or authentication failure"""
    pass


class ConfigurationError(ProviderError):
    """Provider configuration error"""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded"""
    
    def __init__(self, message: str, provider: str = "unknown", retry_after: int = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """Usage quota exceeded"""
    pass


class UnsupportedFormatError(ProviderError):
    """Audio format not supported by provider"""
    pass


class UnsupportedLanguageError(ProviderError):
    """Language not supported by provider"""
    pass


class ConnectionError(ProviderError):
    """Network or connection error"""
    pass


class StreamingError(ProviderError):
    """Error in streaming session"""
    pass


class VoiceNotFoundError(ProviderError):
    """Requested voice not available"""
    pass


class TextTooLongError(ProviderError):
    """Text exceeds provider limits"""
    
    def __init__(self, message: str, provider: str = "unknown", max_length: int = None):
        super().__init__(message, provider)
        self.max_length = max_length