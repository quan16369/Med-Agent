"""
Error Handling and Custom Exceptions
"""


class MedAssistError(Exception):
    """Base exception for all MedAssist errors"""
    pass


class ConfigurationError(MedAssistError):
    """Configuration-related errors"""
    pass


class DatabaseError(MedAssistError):
    """Database connection and query errors"""
    pass


class ModelError(MedAssistError):
    """Model loading and inference errors"""
    pass


class RetrievalError(MedAssistError):
    """Knowledge graph retrieval errors"""
    pass


class ValidationError(MedAssistError):
    """Data validation errors"""
    pass


class TimeoutError(MedAssistError):
    """Operation timeout errors"""
    pass


class ResourceNotFoundError(MedAssistError):
    """Resource not found errors"""
    pass


def handle_error(error: Exception, context: str = "") -> str:
    """
    Handle errors gracefully and return user-friendly message
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error happened
    
    Returns:
        User-friendly error message
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Log the full error
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    # Return user-friendly message
    if isinstance(error, DatabaseError):
        return "Database connection issue. Please try again later."
    elif isinstance(error, ModelError):
        return "Model processing error. Using fallback system."
    elif isinstance(error, RetrievalError):
        return "Unable to retrieve information. Please rephrase your question."
    elif isinstance(error, TimeoutError):
        return "Request timed out. Please try again."
    elif isinstance(error, ValidationError):
        return f"Invalid input: {str(error)}"
    else:
        return "An unexpected error occurred. Please contact support."


class ErrorHandler:
    """Context manager for error handling"""
    
    def __init__(self, context: str = "", raise_on_error: bool = False):
        self.context = context
        self.raise_on_error = raise_on_error
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            message = handle_error(exc_val, self.context)
            
            if self.raise_on_error:
                return False  # Re-raise the exception
            else:
                return True  # Suppress the exception
        return False
