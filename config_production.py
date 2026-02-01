"""
Production-ready configuration management
Supports multiple environments with proper validation
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "./data/medical_kb/knowledge.db"
    enable_wal: bool = True  # Write-Ahead Logging for better concurrency
    cache_size: int = 10000  # SQLite cache size in pages
    timeout: int = 30  # Busy timeout in seconds
    backup_dir: str = "./backups/db"
    auto_vacuum: bool = True


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "google/medgemma-2b"
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    cache_dir: str = "./models/cache"
    fallback_model: str = "mock"  # Fallback if main model fails


@dataclass
class CloudConfig:
    """Cloud services configuration"""
    api_endpoint: str = "https://api.medassist.health"
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    enable_telemetry: bool = True
    enable_cloud_inference: bool = True
    max_cache_size_mb: int = 100


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_encryption: bool = True
    encryption_key_path: str = "./config/encryption.key"
    enable_auth: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    enable_audit_log: bool = True
    audit_log_path: str = "./logs/audit.log"


@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 60
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    thread_pool_size: int = 4
    process_pool_size: int = 2


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 60
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    log_file: str = "./logs/medassist.log"
    max_log_size_mb: int = 100
    log_backup_count: int = 10


@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: str = "production"  # development, staging, production
    clinic_id: str = "default_clinic"
    region: str = "global"
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()
    cloud: CloudConfig = CloudConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Feature flags
    rural_mode: bool = False
    offline_mode: bool = False
    enable_rag: bool = True
    enable_multi_agent: bool = True
    enable_adaptive_models: bool = True
    
    # Deployment
    deployment_type: str = "standalone"  # standalone, cluster, edge
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate database path
        db_dir = Path(self.database.path).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory: {e}")
        
        # Validate model cache
        model_cache = Path(self.model.cache_dir)
        if not model_cache.exists():
            try:
                model_cache.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create model cache directory: {e}")
        
        # Validate cloud config
        if self.cloud.enable_cloud_inference and not self.cloud.api_key:
            if self.environment == "production":
                errors.append("Cloud API key required in production mode")
        
        # Validate log directory
        log_dir = Path(self.monitoring.log_file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory: {e}")
        
        # Validate security
        if self.security.enable_encryption:
            key_path = Path(self.security.encryption_key_path)
            if not key_path.exists():
                logger.warning(f"Encryption key not found: {key_path}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True
    
    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Environment
        config.environment = os.getenv("MEDASSIST_ENV", "production")
        config.clinic_id = os.getenv("CLINIC_ID", "default_clinic")
        config.region = os.getenv("REGION", "global")
        
        # Database
        config.database.path = os.getenv("DB_PATH", config.database.path)
        config.database.cache_size = int(os.getenv("DB_CACHE_SIZE", config.database.cache_size))
        
        # Model
        config.model.name = os.getenv("MODEL_NAME", config.model.name)
        config.model.device = os.getenv("MODEL_DEVICE", config.model.device)
        config.model.load_in_4bit = os.getenv("MODEL_4BIT", "true").lower() == "true"
        
        # Cloud
        config.cloud.api_key = os.getenv("CLOUD_API_KEY")
        config.cloud.api_endpoint = os.getenv("CLOUD_API_ENDPOINT", config.cloud.api_endpoint)
        config.cloud.enable_telemetry = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
        
        # Security
        config.security.enable_encryption = os.getenv("ENABLE_ENCRYPTION", "true").lower() == "true"
        config.security.enable_auth = os.getenv("ENABLE_AUTH", "true").lower() == "true"
        
        # Monitoring
        config.monitoring.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.monitoring.log_format = os.getenv("LOG_FORMAT", "json")
        
        # Feature flags
        config.rural_mode = os.getenv("RURAL_MODE", "false").lower() == "true"
        config.offline_mode = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> "ProductionConfig":
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Recursively update config from dict
        def update_from_dict(obj, d):
            for key, value in d.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if isinstance(attr, (DatabaseConfig, ModelConfig, CloudConfig, 
                                        SecurityConfig, PerformanceConfig, MonitoringConfig)):
                        update_from_dict(attr, value)
                    else:
                        setattr(obj, key, value)
        
        update_from_dict(config, data)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def dataclass_to_dict(obj):
            if isinstance(obj, (DatabaseConfig, ModelConfig, CloudConfig,
                               SecurityConfig, PerformanceConfig, MonitoringConfig)):
                return {k: v for k, v in obj.__dict__.items()}
            return obj
        
        return {
            "environment": self.environment,
            "clinic_id": self.clinic_id,
            "region": self.region,
            "database": dataclass_to_dict(self.database),
            "model": dataclass_to_dict(self.model),
            "cloud": dataclass_to_dict(self.cloud),
            "security": dataclass_to_dict(self.security),
            "performance": dataclass_to_dict(self.performance),
            "monitoring": dataclass_to_dict(self.monitoring),
            "rural_mode": self.rural_mode,
            "offline_mode": self.offline_mode,
            "deployment_type": self.deployment_type
        }
    
    def save(self, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Environment-specific configurations
DEVELOPMENT_CONFIG = ProductionConfig(
    environment="development",
    database=DatabaseConfig(path="./data/dev/knowledge.db"),
    model=ModelConfig(name="mock", device="cpu"),
    cloud=CloudConfig(enable_telemetry=False),
    security=SecurityConfig(enable_encryption=False, enable_auth=False),
    monitoring=MonitoringConfig(log_level="DEBUG", log_format="text")
)

STAGING_CONFIG = ProductionConfig(
    environment="staging",
    database=DatabaseConfig(path="./data/staging/knowledge.db"),
    model=ModelConfig(name="google/medgemma-2b", load_in_4bit=True),
    cloud=CloudConfig(enable_telemetry=True),
    security=SecurityConfig(enable_encryption=True, enable_auth=True),
    monitoring=MonitoringConfig(log_level="INFO", log_format="json")
)

PRODUCTION_CONFIG = ProductionConfig(
    environment="production",
    database=DatabaseConfig(
        path="/var/lib/medassist/knowledge.db",
        cache_size=20000,
        backup_dir="/var/backups/medassist"
    ),
    model=ModelConfig(
        name="google/medgemma-2b",
        load_in_4bit=True,
        cache_dir="/var/cache/medassist/models"
    ),
    cloud=CloudConfig(
        enable_telemetry=True,
        enable_cloud_inference=True,
        retry_attempts=5
    ),
    security=SecurityConfig(
        enable_encryption=True,
        enable_auth=True,
        enable_audit_log=True
    ),
    performance=PerformanceConfig(
        max_concurrent_requests=50,
        enable_caching=True,
        thread_pool_size=8
    ),
    monitoring=MonitoringConfig(
        enable_metrics=True,
        enable_health_checks=True,
        log_level="INFO",
        log_format="json",
        log_file="/var/log/medassist/medassist.log"
    )
)


def get_config(environment: Optional[str] = None) -> ProductionConfig:
    """Get configuration for specified environment"""
    env = environment or os.getenv("MEDASSIST_ENV", "development")
    
    if env == "development":
        return DEVELOPMENT_CONFIG
    elif env == "staging":
        return STAGING_CONFIG
    elif env == "production":
        return PRODUCTION_CONFIG
    else:
        # Try to load from environment or file
        config_file = os.getenv("MEDASSIST_CONFIG_FILE")
        if config_file and os.path.exists(config_file):
            return ProductionConfig.from_file(config_file)
        else:
            return ProductionConfig.from_env()


if __name__ == "__main__":
    # Demo
    print("Configuration Management Demo")
    print("="*60)
    
    # Development
    dev_config = get_config("development")
    print(f"\nDevelopment Config:")
    print(f"  Environment: {dev_config.environment}")
    print(f"  Model: {dev_config.model.name}")
    print(f"  Database: {dev_config.database.path}")
    print(f"  Log Level: {dev_config.monitoring.log_level}")
    
    # Production
    prod_config = get_config("production")
    print(f"\nProduction Config:")
    print(f"  Environment: {prod_config.environment}")
    print(f"  Model: {prod_config.model.name}")
    print(f"  Database: {prod_config.database.path}")
    print(f"  Encryption: {prod_config.security.enable_encryption}")
    print(f"  Metrics: {prod_config.monitoring.enable_metrics}")
    
    # Validation
    print(f"\nValidation:")
    print(f"  Development: {dev_config.validate()}")
    print(f"  Production: {prod_config.validate()}")
