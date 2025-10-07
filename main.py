#!/usr/bin/env python3
"""
Main entry point for MedNER-DE Service.

This script provides a convenient way to start the MedNER-DE Service
with proper configuration and error handling.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from model_loader import ModelLoader, ModelLoadError
from ner_service import MedicalNERService
import uvicorn


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("medner-de.log"),
        ],
    )


async def initialize_service():
    """Initialize the MedNER-DE service."""
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize model loader
        model_loader = ModelLoader(config)
        logger.info("Model loader initialized")

        # Load models
        logger.info("Loading models...")
        model_status = await model_loader.initialize_models()
        logger.info(f"Models loaded: {model_status}")

        # Initialize NER service
        ner_service = MedicalNERService(config, model_loader)
        logger.info("NER service initialized")

        # Warm up models
        logger.info("Warming up models...")
        await model_loader.warmup_models()
        logger.info("Models warmed up successfully")

        return config, model_loader, ner_service

    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Please ensure spaCy German model is installed:")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        sys.exit(1)


async def run_service_async(
    config, host: str = "0.0.0.0", port: int = 8000, workers: int = 1
):
    """Run FastAPI service safely within an existing event loop."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MedNER-DE Service on {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log level: {config.log_level}")

    config_uvicorn = uvicorn.Config(
        "api.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=config.log_level.lower(),
        reload=False,
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


def run_service(config, host="0.0.0.0", port=8000, workers=1):
    """Detect if event loop already running and handle accordingly."""
    try:
        loop = asyncio.get_running_loop()
        # If we are in a running event loop (like Jupyter)
        loop.create_task(run_service_async(config, host, port, workers))
    except RuntimeError:
        # No running loop — safe to start normally
        asyncio.run(run_service_async(config, host, port, workers))
    except KeyboardInterrupt:
        print("Service stopped by user")
    except Exception as e:
        print(f"Service failed to start: {e}")
        sys.exit(1)


async def main():
    """Main function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting MedNER-DE Service")
    logger.info("=" * 50)

    try:
        # Initialize service
        config, model_loader, ner_service = await initialize_service()

        logger.info("✅ Service initialized successfully")
        logger.info(f"📊 Model status: {model_loader.get_model_status()}")
        logger.info(f"🔧 Configuration: {config.host}:{config.port}")

        # Start the service
        # run_service(config, config.host, config.port, config.workers)

        await run_service_async(config, config.host, config.port, config.workers)

    except KeyboardInterrupt:
        logger.info("👋 Service shutdown requested")
    except Exception as e:
        logger.error(f"❌ Service failed: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 MedNER-DE Service stopped")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 12):
        print("❌ Python 3.12+ is required")
        sys.exit(1)

    # Run the service
    asyncio.run(main())
