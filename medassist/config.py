"""
Configuration Management
Environment-based settings for development, testing, and production
"""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database connection settings"""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    use_memory_fallback: bool = True
    connection_timeout: int = 30
    max_retry_attempts: int = 3


@dataclass
class ModelConfig:
    """Model settings"""
    biobert_model: str = "dmis-lab/biobert-base-cased-v1.1"
    device: str = "cpu"
    max_sequence_length: int = 512
    batch_size: int = 8
    use_cache: bool = True


@dataclass
class RetrievalConfig:
    """Knowledge graph retrieval settings"""
    max_hops: int = 3
    min_confidence: float = 0.5
    top_k_paths: int = 20
    enable_multi_hop: bool = True


@dataclass
class PubMedConfig:
    """PubMed API settings"""
    email: str = "user@example.com"
    api_key: Optional[str] = None
    max_results: int = 10
    min_year: int = 2015
    timeout: int = 10


@dataclass
class LoggingConfig:
    """Logging settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration"""
    environment: Environment
    database: DatabaseConfig
    model: ModelConfig
    retrieval: RetrievalConfig
    pubmed: PubMedConfig
    logging: LoggingConfig
    
    # Application settings
    debug: bool = False
    enable_telemetry: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 60


def load_config() -> Config:
    """Load configuration from environment variables"""
    
    env_name = os.getenv("ENVIRONMENT", "development")
    environment = Environment(env_name)
    
    # Database config
    database = DatabaseConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        use_memory_fallback=os.getenv("USE_MEMORY_FALLBACK", "true").lower() == "true",
        connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
        max_retry_attempts=int(os.getenv("DB_MAX_RETRIES", "3"))
    )
    
    # Model config
    device = os.getenv("DEVICE", "cpu")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ModelConfig(
        biobert_model=os.getenv("BIOBERT_MODEL", "dmis-lab/biobert-base-cased-v1.1"),
        device=device,
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
        batch_size=int(os.getenv("BATCH_SIZE", "8")),
        use_cache=os.getenv("USE_MODEL_CACHE", "true").lower() == "true"
    )
    
    # Retrieval config
    retrieval = RetrievalConfig(
        max_hops=int(os.getenv("MAX_HOPS", "3")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.5")),
        top_k_paths=int(os.getenv("TOP_K_PATHS", "20")),
        enable_multi_hop=os.getenv("ENABLE_MULTI_HOP", "true").lower() == "true"
    )
    
    # PubMed config
    pubmed = PubMedConfig(
        email=os.getenv("PUBMED_EMAIL", "user@example.com"),
        api_key=os.getenv("PUBMED_API_KEY"),
        max_results=int(os.getenv("PUBMED_MAX_RESULTS", "10")),
        min_year=int(os.getenv("PUBMED_MIN_YEAR", "2015")),
        timeout=int(os.getenv("PUBMED_TIMEOUT", "10"))
    )
    
    # Logging config
    log_level = os.getenv("LOG_LEVEL", "INFO")
    if environment == Environment.DEVELOPMENT:
        log_level = "DEBUG"
    elif environment == Environment.PRODUCTION:
        log_level = "WARNING"
    
    logging = LoggingConfig(
        level=log_level,
        format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        file_path=os.getenv("LOG_FILE"),
        max_bytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
        backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
    )
    
    # Application settings
    debug = environment == Environment.DEVELOPMENT
    
    return Config(
        environment=environment,
        database=database,
        model=model,
        retrieval=retrieval,
        pubmed=pubmed,
        logging=logging,
        debug=debug,
        enable_telemetry=os.getenv("ENABLE_TELEMETRY", "true").lower() == "true",
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60"))
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = load_config()
    return _config
