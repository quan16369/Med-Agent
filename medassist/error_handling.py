"""
Production-ready exception handling and error recovery
Implements retry logic, circuit breaker, and graceful degradation
"""

import logging
import time
import functools
from typing import Callable, Type, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Prevents cascading failures by stopping requests to failing service
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: Attempting recovery (HALF_OPEN)")
            else:
                raise Exception(f"Circuit breaker OPEN (failures={self.failure_count})")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker: Recovery successful (CLOSED)")
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker: OPEN (failures={self.failure_count})")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts: {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
            return None
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator
    
    Args:
        seconds: Maximum execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            # Set the signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel the alarm
            
            return result
        
        return wrapper
    return decorator


def fallback(fallback_func: Callable):
    """
    Fallback decorator - use alternative function if main fails
    
    Args:
        fallback_func: Function to call if main function fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed: {e}. Using fallback.",
                    exc_info=True
                )
                return fallback_func(*args, **kwargs)
        
        return wrapper
    return decorator


class GracefulDegradation:
    """
    Graceful degradation manager
    
    Allows system to continue operating with reduced functionality
    """
    
    def __init__(self):
        self.degraded_features = set()
        self.feature_errors = {}
    
    def mark_degraded(self, feature: str, error: Exception):
        """Mark a feature as degraded"""
        self.degraded_features.add(feature)
        self.feature_errors[feature] = {
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.warning(f"Feature degraded: {feature} - {error}")
    
    def restore_feature(self, feature: str):
        """Restore a degraded feature"""
        if feature in self.degraded_features:
            self.degraded_features.remove(feature)
            if feature in self.feature_errors:
                del self.feature_errors[feature]
            logger.info(f"Feature restored: {feature}")
    
    def is_degraded(self, feature: str) -> bool:
        """Check if a feature is degraded"""
        return feature in self.degraded_features
    
    def get_status(self) -> dict:
        """Get degradation status"""
        return {
            "degraded_features": list(self.degraded_features),
            "errors": self.feature_errors
        }


class ErrorRecovery:
    """
    Centralized error recovery strategies
    """
    
    @staticmethod
    def recover_from_model_error(error: Exception) -> dict:
        """Recover from model inference error"""
        logger.error(f"Model error: {error}")
        
        return {
            "success": False,
            "error": "Model inference failed",
            "fallback": "Using rule-based system",
            "recommendations": [
                "Please consult a medical professional",
                "This is a preliminary assessment only"
            ]
        }
    
    @staticmethod
    def recover_from_database_error(error: Exception) -> dict:
        """Recover from database error"""
        logger.error(f"Database error: {error}")
        
        return {
            "success": False,
            "error": "Knowledge base unavailable",
            "fallback": "Using cached data",
            "message": "Limited information available"
        }
    
    @staticmethod
    def recover_from_cloud_error(error: Exception) -> dict:
        """Recover from cloud service error"""
        logger.error(f"Cloud error: {error}")
        
        return {
            "success": True,
            "mode": "offline",
            "message": "Operating in offline mode",
            "limitations": [
                "Local model only",
                "Cached knowledge base",
                "No real-time updates"
            ]
        }


class ErrorHandler:
    """
    Production error handler with proper logging and recovery
    """
    
    def __init__(self):
        self.degradation = GracefulDegradation()
        self.circuit_breakers = {}
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: str = "error",
        recovery_func: Optional[Callable] = None
    ) -> Any:
        """
        Handle error with proper logging and optional recovery
        
        Args:
            error: The exception that occurred
            context: Context description
            severity: Error severity (debug, info, warning, error, critical)
            recovery_func: Optional recovery function
        """
        # Log error with context
        log_func = getattr(logger, severity, logger.error)
        log_func(f"Error in {context}: {error}", exc_info=True)
        
        # Attempt recovery if function provided
        if recovery_func:
            try:
                return recovery_func(error)
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {context}: {recovery_error}")
        
        # Re-raise if no recovery
        raise error


# Global error handler instance
error_handler = ErrorHandler()


if __name__ == "__main__":
    # Demo
    print("Error Handling Demo")
    print("="*60)
    
    # Retry decorator
    @retry(max_attempts=3, delay=0.5)
    def flaky_function(success_on_attempt: int):
        """Simulate flaky function"""
        flaky_function.attempt = getattr(flaky_function, 'attempt', 0) + 1
        
        if flaky_function.attempt < success_on_attempt:
            raise Exception(f"Attempt {flaky_function.attempt} failed")
        
        return f"Success on attempt {flaky_function.attempt}"
    
    # Test retry
    try:
        result = flaky_function(success_on_attempt=2)
        print(f"\nRetry test: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Circuit breaker
    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=2)
    
    def failing_service():
        raise Exception("Service unavailable")
    
    print(f"\nCircuit breaker test:")
    for i in range(5):
        try:
            circuit.call(failing_service)
        except Exception as e:
            print(f"  Attempt {i+1}: {str(e)[:50]}")
    
    # Graceful degradation
    degradation = GracefulDegradation()
    degradation.mark_degraded("cloud_sync", Exception("Connection timeout"))
    degradation.mark_degraded("telemetry", Exception("API unavailable"))
    
    print(f"\nDegradation status:")
    status = degradation.get_status()
    print(f"  Degraded features: {status['degraded_features']}")
