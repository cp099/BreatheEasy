
# File: src/exceptions.py
"""
Custom exception classes for the BreatheEasy application.

This module defines a hierarchy of custom exceptions to allow for more specific
and robust error handling throughout the application, distinguishing between
file errors, API issues, modeling problems, and configuration errors.
"""

class BreatheEasyError(Exception):
    """Base class for exceptions in this application."""
    pass

# --- Data and File Errors ---

class DataFileNotFoundError(BreatheEasyError, FileNotFoundError):
    """Raised when the primary data file cannot be found."""
    pass

class ConfigFileNotFoundError(BreatheEasyError, FileNotFoundError):
    """Raised when the configuration file cannot be found."""
    pass

class ModelFileNotFoundError(BreatheEasyError, FileNotFoundError):
    """Raised when a specific model file cannot be found."""
    pass

# --- API Errors ---

class APIError(BreatheEasyError):
    """Base class for external API related errors."""
    def __init__(self, message="API error occurred", status_code=None, service="Unknown"):
        self.status_code = status_code
        self.service = service
        super().__init__(f"{service} API Error: {message}" + (f" (Status: {status_code})" if status_code else ""))

class APIKeyError(APIError):
    """Raised when an API key is missing, invalid, or unauthorized."""
    def __init__(self, message="Invalid or missing API key", service="Unknown"):
        super().__init__(message, status_code=401, service=service) 

class APIRateLimitError(APIError):
    """Raised when an API rate limit is exceeded."""
    def __init__(self, message="API rate limit exceeded", service="Unknown"):
        super().__init__(message, status_code=429, service=service) 

class APINotFoundError(APIError):
    """Raised when a requested resource (e.g., city) is not found by the API."""
    def __init__(self, message="Resource not found by API", service="Unknown"):
        super().__init__(message, status_code=404, service=service) 

class APITimeoutError(APIError, TimeoutError):
     """Raised when a request to an API times out."""
     def __init__(self, message="API request timed out", service="Unknown"):
        super().__init__(message, service=service)

# --- Modeling Errors ---

class ModelLoadError(BreatheEasyError):
    """Raised when a model file cannot be loaded or deserialized."""
    pass

class PredictionError(BreatheEasyError):
    """Raised during the prediction generation phase."""
    pass

# --- Configuration Errors ---

class ConfigError(BreatheEasyError):
    """Raised for general configuration loading/parsing issues."""
    pass