"""
Model loader module for MedNER-DE Service.

This module handles downloading, unpacking, and loading of various NER models
including spaCy, GERNERMEDpp, and GermanBERT.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import aiohttp
import aiofiles
import ssl
import certifi

from config import ServiceConfig, ModelConfig

logger = logging.getLogger(__name__)


class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelLoader:
    """Handles downloading, unpacking, and loading of NER models."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.loaded_models: Dict[str, Any] = {}
        self.model_status: Dict[str, bool] = {}
        
    async def initialize_models(self) -> Dict[str, bool]:
        """
        Initialize all configured models.
        
        Returns:
            Dict mapping model names to their load status
        """
        logger.info("Starting model initialization...")
        
        # Create models directory
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Initialize models in parallel
        tasks = []
        
        if self.config.spacy_enabled:
            tasks.append(self._load_spacy_model())
            
        if self.config.gernermed_enabled:            
            tasks.append(self._load_gernermed_model())
            
        if self.config.germanbert_enabled:
            tasks.append(self._load_germanbert_model())
        
        # Wait for all model loading tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        model_names = ["spacy", "gernermed", "germanbert"]
        for i, result in enumerate(results):
            model_name = model_names[i] if i < len(model_names) else f"model_{i}"
            if isinstance(result, Exception):
                logger.error(f"Failed to load {model_name}: {result}")
                self.model_status[model_name] = False
            else:
                self.model_status[model_name] = result
        
        # Ensure at least spaCy is loaded
        if not self.model_status.get("spacy", False):
            raise ModelLoadError("Failed to load spaCy model - this is required for the service to function")
        
        logger.info(f"Model initialization complete. Status: {self.model_status}")
        return self.model_status
    
    async def _load_spacy_model(self) -> bool:
        """Load spaCy German model."""
        try:
            logger.info("Loading spaCy German model...")
            
            # Check if model is already installed
            try:
                import spacy
                nlp = spacy.load("de_core_news_md")
                self.loaded_models["spacy"] = nlp
                logger.info("spaCy model loaded successfully")
                return True
            except OSError:
                logger.info("spaCy model not found, downloading...")
            
            # Download and install spaCy model
            await self._download_and_install_spacy()
            
            # Load the model
            import spacy
            nlp = spacy.load("de_core_news_md")
            self.loaded_models["spacy"] = nlp
            logger.info("spaCy model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return False
    
    async def _download_and_install_spacy(self):
        """Download and install spaCy German model."""
        try:
            # Download the wheel file
            spacy_config = self.config.get_spacy_model_config()
            wheel_path = await self._download_file(
                spacy_config.url,
                os.path.join(self.config.cache_dir, "de_core_news_md.whl")
            )
            
            # Install the wheel
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                str(wheel_path), "--no-deps", "--force-reinstall"
            ], check=True)
            
            logger.info("spaCy model installed successfully")
            
        except Exception as e:
            logger.error(f"Failed to install spaCy model: {e}")
            raise ModelDownloadError(f"spaCy model installation failed: {e}")
    
    async def _load_gernermed_model(self) -> bool:
        """Load GERNERMEDpp model."""
        try:
            logger.info("Loading GERNERMEDpp model...")
            
            # Check if model directory exists
            gernermed_config = self.config.get_gernermed_model_config()
            model_path = Path(gernermed_config.local_modal_path)
            
            if not model_path.exists():
                logger.info("GERNERMEDpp model not found, downloading...")
                await self._download_and_extract_gernermed()
            
            # Load GERNERMEDpp model
            # Note: This is a placeholder - actual implementation would depend on GERNERMEDpp's API
            self.loaded_models["gernermed"] = {
                "model_path": str(model_path),
                "loaded": True
            }
            
            import spacy
            nlp = spacy.load(model_path)
            self.loaded_models["gernermed"] = nlp
            
            
            logger.info("GERNERMEDpp model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GERNERMEDpp model: {e}")
            return False
    
    async def _download_and_extract_gernermed(self):
        """Download and extract GERNERMEDpp model."""
        try:
            # Download the zip file
            gernermed_config = self.config.get_gernermed_model_config()
            
            #zip_path = gernermed_config.zip_path
            
            model_path = Path(gernermed_config.local_modal_path)
            if not model_path.exists():
                # Create models directory
                os.makedirs(model_path, exist_ok=True)
                
            local_zip_path = Path(gernermed_config.local_file_path)
            if not model_path.exists():
                # Create model download directory
                os.makedirs(model_path, exist_ok=True)
                        
            # zip_path = Path(gernermed_config.local_file_path)
            # if not model_path.exists():
                
            zip_path = await self._download_file_with_ssl(
                 gernermed_config.url,
                 os.path.join(local_zip_path, gernermed_config.local_file_name)
            )
            
            # Extract to models directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
                
            # remove the zip file
            os.remove(zip_path)
            
            logger.info("GERNERMEDpp model extracted successfully")
            
        except Exception as e:
            logger.error(f"Failed to extract GERNERMEDpp model: {e}")
            raise ModelDownloadError(f"GERNERMEDpp model extraction failed: {e}")
    
    async def _load_germanbert_model(self) -> bool:
        """Load GermanBERT model."""
        try:
            logger.info("Loading GermanBERT model...")
            
            # Check if model directory exists
            germanbert_config = self.config.get_germanbert_model_config()
            if not germanbert_config:
                raise ModelLoadError("GermanBERT model not configured")
                
            model_path = Path(germanbert_config.local_modal_path)
            if not model_path.exists():
                logger.info("GermanBERT model not found, downloading...")
                await self._download_germanbert()
            
            # Load GermanBERT model
            # Note: This is a placeholder - actual implementation would depend on the specific model
            self.loaded_models["germanbert"] = {
                "model_path": str(model_path),
                "loaded": True
            }
            
            logger.info("GermanBERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GermanBERT model: {e}")
            return False
    
    async def _download_germanbert(self):
        """Download GermanBERT model."""
        try:
            # Download the model
            germanbert_config = self.config.get_germanbert_model_config()
            if not germanbert_config:
                raise ModelLoadError("GermanBERT model not configured")
                
            model_path = await self._download_file(
                germanbert_config.url,
                os.path.join(self.config.cache_dir, "GermanBERT.zip")
            )
            
            # Extract if it's a zip file
            if model_path.endswith('.zip'):
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(germanbert_config.local_modal_path)
            
            logger.info("GermanBERT model downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download GermanBERT model: {e}")
            raise ModelDownloadError(f"GermanBERT model download failed: {e}")
    
    async def _download_file(self, url: str, destination: str) -> str:
        """
        Download a file from URL to destination.
        
        Args:
            url: URL to download from
            destination: Local path to save the file
            
        Returns:
            Path to the downloaded file
        """
        logger.info(f"Downloading {url} to {destination}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ModelDownloadError(f"Failed to download {url}: HTTP {response.status}")
                
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                
                # Download file
                async with aiofiles.open(destination, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        logger.info(f"Downloaded {url} to {destination}")
        return destination
    
    async def _download_file_with_ssl(self, url: str, dest_path: str) -> str:
        """Download file with verified SSL context, fallback, and progress indicator."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async def _stream_download(session, url, dest_path):
            async with session.get(url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(dest_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total > 0:
                            percent = downloaded / total * 100
                            # overwrite same line
                            sys.stdout.write(f"\rDownloading: {percent:6.2f}% ({downloaded/1e6:.2f} MB / {total/1e6:.2f} MB)")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(f"\rDownloaded: {downloaded/1e6:.2f} MB")
                            sys.stdout.flush()

                print("\nDownload complete.")

        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                await _stream_download(session, url, dest_path)
            return dest_path

        except aiohttp.ClientConnectorCertificateError:
            print(f"\n[WARN] SSL verification failed for {url}, retrying with ssl=False (trusted host only).")
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                await _stream_download(session, url, dest_path)
            return dest_path

        except Exception as e:
            print(f"[ERROR] Download failed for {url}: {e}")
            raise
        
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self.loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return self.model_status.get(model_name, False)
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models."""
        return self.model_status.copy()
    
    async def warmup_models(self):
        """Warm up models with sample text."""
        if len(self.loaded_models) == 0:
            return
        
        logger.info("Warming up models...")
        
        for model_name, model in self.loaded_models.items():        
            nlp = model
            sample_text = "Der Patient hat Diabetes und nimmt Metformin."
            doc = nlp(sample_text)
            logger.info(f"{model_name} warmup completed. Processed: {sample_text}")
        
        logger.info("Model warmup completed")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up model resources...")
        
        # Clear loaded models
        self.loaded_models.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Model cleanup completed")
