"""
Production deployment script.
Starts the FastAPI server with optimal settings.
"""

import uvicorn
from medassist.config import get_config
from medassist.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

if __name__ == "__main__":
    config = get_config()
    
    logger.info("Starting Medical Knowledge Assistant API")
    logger.info(f"Environment: {config.logging.environment}")
    logger.info(f"Device: {config.model.device}")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker due to model memory requirements
        timeout_keep_alive=30,
        limit_concurrency=10
    )
