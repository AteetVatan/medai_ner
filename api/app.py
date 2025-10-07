"""
FastAPI application for MedNER-DE Service.

This module provides the REST API endpoints for the German medical NER service.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from config import load_config
from model_loader import ModelLoader, ModelLoadError
from ner_service import MedicalNERService, ExtractionResult, MedicalEntity

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
config = None
model_loader = None
ner_service = None


# Pydantic models for API
class TextInput(BaseModel):
    """Input model for single text extraction."""

    text: str = Field(
        ..., description="Text to extract entities from", min_length=1, max_length=50000
    )

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class BatchTextInput(BaseModel):
    """Input model for batch text extraction."""

    texts: List[str] = Field(
        ...,
        description="List of texts to extract entities from",
        min_items=1,
        max_items=100,
    )

    @validator("texts")
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return [text.strip() for text in v]


class EntityResponse(BaseModel):
    """Response model for a single entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_text: Optional[str] = None
    icd_code: Optional[str] = None
    icd_description: Optional[str] = None
    source_model: str
    category: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response model for entity extraction."""

    entities: List[EntityResponse]
    processing_time: float
    model_used: List[str]
    text_length: int
    timestamp: str


class BatchExtractionResponse(BaseModel):
    """Response model for batch entity extraction."""

    results: List[ExtractionResponse]
    total_processing_time: float
    batch_size: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: str
    models: Dict[str, bool]
    stats: Dict[str, Any]
    memory_usage: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="MedNER-DE Service",
    description="German Medical Named Entity Recognition Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    global config, model_loader, ner_service

    logger.info("Starting MedNER-DE Service...")

    try:
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: {config}")

        # Initialize model loader
        model_loader = ModelLoader(config)
        logger.info("Model loader initialized")

        # Load models
        model_status = await model_loader.initialize_models()
        logger.info(f"Models loaded: {model_status}")

        # Initialize NER service
        ner_service = MedicalNERService(config, model_loader)
        logger.info("NER service initialized")

        # Warm up models
        await model_loader.warmup_models()
        logger.info("Models warmed up")

        logger.info("MedNER-DE Service started successfully")

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global model_loader

    logger.info("Shutting down MedNER-DE Service...")

    if model_loader:
        model_loader.cleanup()

    logger.info("Service shutdown complete")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "MedNER-DE Service",
        "version": "1.0.0",
        "description": "German Medical Named Entity Recognition Service",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_entities(input_data: TextInput):
    """
    Extract medical entities from a single text.

    Args:
        input_data: Text input containing the text to process

    Returns:
        ExtractionResponse with detected entities and metadata
    """
    if not ner_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Extract entities
        result = await ner_service.extract_entities(input_data.text)

        # Convert entities to response format
        entities = [
            EntityResponse(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
                normalized_text=entity.normalized_text,
                icd_code=entity.icd_code,
                icd_description=entity.icd_description,
                source_model=entity.source_model,
                category=entity.category,
            )
            for entity in result.entities
        ]

        return ExtractionResponse(
            entities=entities,
            processing_time=result.processing_time,
            model_used=result.model_used,
            text_length=result.text_length,
            timestamp=result.timestamp.isoformat(),
        )

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract_batch", response_model=BatchExtractionResponse)
async def extract_entities_batch(input_data: BatchTextInput):
    """
    Extract medical entities from multiple texts in batch.

    Args:
        input_data: Batch text input containing list of texts to process

    Returns:
        BatchExtractionResponse with results for all texts
    """
    if not ner_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Check batch size limit
        if len(input_data.texts) > config.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(input_data.texts)} exceeds maximum {config.max_batch_size}",
            )

        # Extract entities for all texts
        results = await ner_service.extract_batch(input_data.texts)

        # Convert results to response format
        response_results = []
        total_processing_time = 0.0

        for result in results:
            entities = [
                EntityResponse(
                    text=entity.text,
                    label=entity.label,
                    start=entity.start,
                    end=entity.end,
                    confidence=entity.confidence,
                    normalized_text=entity.normalized_text,
                    icd_code=entity.icd_code,
                    icd_description=entity.icd_description,
                    source_model=entity.source_model,
                    category=entity.category,
                )
                for entity in result.entities
            ]

            response_results.append(
                ExtractionResponse(
                    entities=entities,
                    processing_time=result.processing_time,
                    model_used=result.model_used,
                    text_length=result.text_length,
                    timestamp=result.timestamp.isoformat(),
                )
            )

            total_processing_time += result.processing_time

        return BatchExtractionResponse(
            results=response_results,
            total_processing_time=total_processing_time,
            batch_size=len(input_data.texts),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch extraction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and status.

    Returns:
        HealthResponse with service status, model availability, and statistics
    """
    if not ner_service:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            models={},
            stats={},
            memory_usage={},
        )

    try:
        # Get health status
        health_data = await ner_service.health_check()

        return HealthResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            models=health_data["models"],
            stats=health_data["stats"],
            memory_usage=health_data["memory_usage"],
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            models={},
            stats={},
            memory_usage={},
        )


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """
    Get service statistics and performance metrics.

    Returns:
        Dictionary with service statistics
    """
    if not ner_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return ner_service.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/models", response_model=Dict[str, bool])
async def get_model_status():
    """
    Get status of all loaded models.

    Returns:
        Dictionary mapping model names to their load status
    """
    if not model_loader:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return model_loader.get_model_status()
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model status: {str(e)}"
        )


@app.post("/extract_with_stats", response_model=Dict[str, Any])
async def extract_entities_with_stats(input_data: TextInput):
    """
    Extract medical entities with detailed statistics.

    Args:
        input_data: Text input containing the text to process

    Returns:
        Dictionary with entities and detailed statistics
    """
    if not ner_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Extract entities
        result = await ner_service.extract_entities(input_data.text)

        # Get entity statistics
        stats = ner_service.get_entity_statistics(result.entities)

        # Format entities for display
        display_text = ner_service.format_entities_for_display(result.entities)

        return {
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "normalized_text": entity.normalized_text,
                    "icd_code": entity.icd_code,
                    "icd_description": entity.icd_description,
                    "source_model": entity.source_model,
                    "category": entity.category,
                }
                for entity in result.entities
            ],
            "statistics": stats,
            "display_text": display_text,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "text_length": result.text_length,
            "timestamp": result.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Entity extraction with stats failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Extraction with stats failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "api.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info"
    )
