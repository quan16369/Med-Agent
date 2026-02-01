"""
Production-grade logging setup
Structured logging with proper formatting and rotation
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add context fields
        for key in ["clinic_id", "patient_id", "case_id", "user_id"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        
        return json.dumps(log_data)


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self, clinic_id: str = None):
        super().__init__()
        self.clinic_id = clinic_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        if self.clinic_id:
            record.clinic_id = self.clinic_id
        return True


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: str = "./logs/medassist.log",
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
    clinic_id: str = None
):
    """
    Setup production-grade logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
        log_file: Path to log file
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        clinic_id: Clinic identifier for context
    """
    
    # Create logs directory if not exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add context filter
    if clinic_id:
        context_filter = ContextFilter(clinic_id=clinic_id)
        console_handler.addFilter(context_filter)
        file_handler.addFilter(context_filter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized: level={log_level}, format={log_format}, file={log_file}")


class AuditLogger:
    """Audit logger for security-sensitive operations"""
    
    def __init__(self, log_file: str = "./logs/audit.log"):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Separate audit log file
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=20  # Keep more audit logs
        )
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Log access attempt"""
        self.logger.info(
            f"Access: {action}",
            extra={
                "extra_fields": {
                    "event_type": "access",
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "success": success,
                    "details": details or {}
                }
            }
        )
    
    def log_case_access(
        self,
        user_id: str,
        case_id: str,
        action: str
    ):
        """Log patient case access"""
        self.logger.info(
            f"Case access: {action}",
            extra={
                "extra_fields": {
                    "event_type": "case_access",
                    "user_id": user_id,
                    "case_id": case_id,
                    "action": action
                }
            }
        )
    
    def log_configuration_change(
        self,
        user_id: str,
        setting: str,
        old_value: Any,
        new_value: Any
    ):
        """Log configuration change"""
        self.logger.info(
            f"Config change: {setting}",
            extra={
                "extra_fields": {
                    "event_type": "config_change",
                    "user_id": user_id,
                    "setting": setting,
                    "old_value": str(old_value),
                    "new_value": str(new_value)
                }
            }
        )
    
    def log_auth_event(
        self,
        user_id: str,
        event: str,
        success: bool,
        ip_address: str = None
    ):
        """Log authentication event"""
        self.logger.info(
            f"Auth: {event}",
            extra={
                "extra_fields": {
                    "event_type": "authentication",
                    "user_id": user_id,
                    "event": event,
                    "success": success,
                    "ip_address": ip_address
                }
            }
        )


if __name__ == "__main__":
    # Demo
    print("Logging Setup Demo")
    print("="*60)
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_format="json",
        log_file="./logs/demo.log",
        clinic_id="demo_clinic_001"
    )
    
    # Test logging
    logger = logging.getLogger(__name__)
    logger.info("System initialized")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    try:
        1 / 0
    except Exception as e:
        logger.exception("Exception occurred")
    
    # Test audit logging
    audit = AuditLogger(log_file="./logs/demo_audit.log")
    audit.log_access("user123", "view_case", "case_456", True)
    audit.log_auth_event("user123", "login", True, "192.168.1.1")
    
    print("\nLogs written to ./logs/")
