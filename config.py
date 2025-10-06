"""
Configuration module for MedNER-DE Service.

This module contains all configuration settings for the German medical NER service,
including model URLs, paths, and service parameters using Pydantic for validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.env_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for individual models."""
    name: str
    url: str
    local_path: str = ""
    local_modal_path: str = ""
    local_file_path:str = ""
    local_file_name:str = ""
    zip_path:str = ""
    enabled: bool = True
    timeout: int = Field(default=300, ge=1, le=3600)  # 1 second to 1 hour
    memory_limit_mb: int = Field(default=2048, ge=512, le=16384)  # 512MB to 16GB

    @validator('local_path')
    def validate_local_path(cls, v):
        """Ensure local path is absolute."""
        return os.path.abspath(v)

    @validator('url')
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://', 'file://')):
            raise ValueError('URL must start with http://, https://, or file://')
        return v


class ServiceConfig(BaseSettings):
    """Main service configuration with environment variable support."""
    
    # Model configurations
    spacy_model_name: str = Field(default="spacy_de_core_news_md", env="SPACY_MODEL_NAME")
    spacy_model_url: str = Field(
        default="https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.7.0/de_core_news_md-3.7.0-py3-none-any.whl",
        env="SPACY_MODEL_URL"
    )
    spacy_enabled: bool = Field(default=True, env="SPACY_ENABLED")
    
    gernermed_model_name: str = Field(default="GERNERMEDpp", env="GERNERMED_MODEL_NAME")
    gernermed_model_url: str = Field(
        default="https://myweb.rz.uni-augsburg.de/~freijoha/GERNERMEDpp/GERNERMEDpp_GottBERT.zip",
        env="GERNERMED_URL"
    )
    gernermed_model_file_name: str = Field(default="GERNERMEDpp_GottBERT.zip", env="GERNERMED_MODEL_FILE_NAME")    
    gernermed_enabled: bool = Field(default=True, env="GERNERMED_ENABLED")
    
    germanbert_model_name: Optional[str] = Field(default=None, env="GERMANBERT_MODEL_NAME")
    germanbert_model_url: Optional[str] = Field(default=None, env="GERMANBERT_URL")
    germanbert_enabled: bool = Field(default=False, env="GERMANBERT_ENABLED")
    
           
    # Model paths
    models_dir: str = Field(default="./models", env="MODELS_DIR")
    models_file_dir: str = Field(default="./ner_archived", env="MODELS_FILE_DIR")    
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    
    
    # Service settings
    max_text_length: int = Field(default=10000, ge=100, le=100000, env="MAX_TEXT_LENGTH")
    max_batch_size: int = Field(default=100, ge=1, le=1000, env="MAX_BATCH_SIZE")
    extraction_timeout: int = Field(default=30, ge=5, le=300, env="EXTRACTION_TIMEOUT")
    
    # API settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, ge=1, le=65535, env="PORT")
    workers: int = Field(default=1, ge=1, le=32, env="WORKERS")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", env="LOG_LEVEL"
    )

       
    # Performance settings
    warmup_enabled: bool = Field(default=True, env="WARMUP_ENABLED")
    memory_limit_mb: int = Field(default=4096, ge=1024, le=32768, env="MEMORY_LIMIT_MB")
    gc_threshold: float = Field(default=0.8, ge=0.1, le=1.0, env="GC_THRESHOLD")
    batch_processing: bool = Field(default=True, env="BATCH_PROCESSING")
    
    # ICD configuration
    icd_lookup_enabled: bool = Field(default=True, env="ICD_LOOKUP_ENABLED")
    icd_cache_size: int = Field(default=1000, ge=100, le=10000, env="ICD_CACHE_SIZE")
    icd_timeout: int = Field(default=5, ge=1, le=60, env="ICD_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

    @validator('models_dir', 'cache_dir', 'models_file_dir')
    def validate_directories(cls, v):
        """Ensure directories are absolute paths."""
        return os.path.abspath(v)

    @validator('log_level')
    def validate_log_level(cls, v):
        """Convert string to uppercase for log level."""
        return v.upper()

    @root_validator
    def validate_germanbert_config(cls, values):
        """Validate GermanBERT configuration."""
        if values.get('germanbert_enabled') and not values.get('germanbert_model_url'):
            raise ValueError('GermanBERT URL must be provided when GermanBERT is enabled')
        return values

    def get_spacy_model_config(self) -> ModelConfig:
        """Get spaCy model configuration."""
        return ModelConfig(
            name=self.spacy_model_name,
            url=self.spacy_model_url,
            local_modal_path=os.path.join(self.models_dir, self.spacy_model_name),
            enabled=self.spacy_enabled
        )

    def get_gernermed_model_config(self) -> ModelConfig:
        """Get GERNERMED model configuration."""
        return ModelConfig(
            name=self.gernermed_model_name,
            url=self.gernermed_model_url,
            local_modal_path=os.path.join(self.models_dir, self.gernermed_model_name),
            local_file_path=os.path.join(self.models_file_dir, self.gernermed_model_name),
            zip_path=os.path.join(self.models_file_dir, self.gernermed_model_name, self.gernermed_model_file_name),
            local_file_name=self.gernermed_model_file_name,
            enabled=self.gernermed_enabled
        )

    def get_germanbert_model_config(self) -> Optional[ModelConfig]:
        """Get GermanBERT model configuration."""
        if not self.germanbert_enabled or not self.germanbert_model_url:
            return None
        
        return ModelConfig(
            name=self.germanbert_model_name or "GermanBERT",
            url=self.germanbert_model_url,
            local_modal_path=os.path.join(self.models_dir, self.germanbert_model_name or "GermanBERT"),
            enabled=self.germanbert_enabled
        )


def load_config() -> ServiceConfig:
    """Load configuration from environment variables and .env file."""
    return ServiceConfig()


def get_model_configs(service_config: ServiceConfig) -> Dict[str, ModelConfig]:
    """Get all model configurations as a dictionary."""
    configs = {
        "spacy": service_config.get_spacy_model_config(),
        "gernermed": service_config.get_gernermed_model_config()
    }
    
    germanbert_config = service_config.get_germanbert_model_config()
    if germanbert_config:
        configs["germanbert"] = germanbert_config
    
    return configs


# Entity mapping configuration
ENTITY_MAPPING = {
    # spaCy labels to normalized categories
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION", 
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "MISC": "MISC",
    
    # Medical entity categories
    "MED_DRUG": "MED_DRUG",
    "DIAGNOSIS": "DIAGNOSIS", 
    "ANATOMY": "ANATOMY",
    "PHYSIO_TREATMENT": "PHYSIO_TREATMENT",
    "MEDICAL_DEVICE": "MEDICAL_DEVICE",
    "SYMPTOM": "SYMPTOM",
    "DISEASE": "DISEASE",
    "MEDICAL_PROCEDURE": "MEDICAL_PROCEDURE"
}

# Default configuration instance
def get_default_config() -> ServiceConfig:
    """Get default configuration instance."""
    return ServiceConfig()

# Configuration validation
def validate_config(config: ServiceConfig) -> bool:
    """Validate configuration settings."""
    try:
        # Check if models directory exists or can be created
        os.makedirs(config.models_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # Validate port range
        if not (1 <= config.port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {config.port}")
        
        # Validate text length
        if not (100 <= config.max_text_length <= 100000):
            raise ValueError(f"Max text length must be between 100 and 100000, got {config.max_text_length}")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# Legacy constants for backward compatibility
ICD_CONFIG = {
    "lookup_enabled": True,
    "cache_size": 1000,
    "timeout": 5
}

PERFORMANCE_CONFIG = {
    "warmup_enabled": True,
    "memory_limit_mb": 4096,
    "gc_threshold": 0.8,
    "batch_processing": True
}
