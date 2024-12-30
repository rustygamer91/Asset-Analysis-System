# main.py

import uvicorn
from hybridprocessor import HybridProcessor
import logging
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger.add("debug.log", 
          format="{time} {level} {message}",
          level="DEBUG",
          rotation="10 MB")

logger.info("Starting application...")

# Create processor instance
processor = HybridProcessor()

if __name__ == "__main__":
    logger.info("Starting server in debug mode")
    uvicorn.run(
        processor.app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )