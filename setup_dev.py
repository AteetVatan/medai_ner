#!/usr/bin/env python3
"""
Development setup script for MedNER-DE Service.

This script helps set up the development environment and download required models.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 12):
        print("❌ Python 3.12+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")


def download_spacy_model() -> bool:
    """Download spaCy German model (de_core_news_md) via safest method."""
    print("📥 Downloading spaCy German model (de_core_news_md)…")
    
    # Strategy A: Use spaCy CLI (preferred if works)
    cli_cmd = [sys.executable, "-m", "spacy", "download", "de_core_news_md"]
    if run_command(cli_cmd, "spaCy CLI download"):
        return True
    
    # Strategy B: Direct pip install of model wheel (bypass CLI)
    wheel_url = (
        "https://github.com/explosion/spacy-models/releases/download/"
        "de_core_news_md-3.7.0/de_core_news_md-3.7.0-py3-none-any.whl"
    )
    pip_cmd = [sys.executable, "-m", "pip", "install", wheel_url]
    if run_command(pip_cmd, "pip install spaCy German model wheel"):
        return True
    
    print("[ERROR] All methods to download spaCy German model failed.")
    return False


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ["models", "cache", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_environment():
    """Set up environment variables."""
    print("🔧 Setting up environment...")
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# MedNER-DE Service Environment Variables

# Service configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=INFO

# Model configuration
SPACY_ENABLED=true
GERNERMED_ENABLED=true
GERMANBERT_ENABLED=false

# Service limits
MAX_TEXT_LENGTH=10000
MAX_BATCH_SIZE=100
EXTRACTION_TIMEOUT=30

# Model URLs (optional overrides)
# SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/de_core_news_md-3.7.0/de_core_news_md-3.7.0-py3-none-any.whl
# GERNERMED_URL=https://myweb.rz.uni-augsburg.de/~freijoha/GERNERMEDpp/GERNERMEDpp_GottBERT.zip
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    return run_command("python -m pytest tests/ -v", "Running test suite")


def main():
    """Main setup function."""
    print("🚀 MedNER-DE Service Development Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  Failed to download spaCy model - you may need to download it manually")
    
    # Setup environment
    setup_environment()
    
    # Run tests
    print("\n🧪 Running tests to verify setup...")
    if run_tests():
        print(" All tests passed!")
    else:
        print("  Some tests failed - check the output above")
    
    print("\n Development setup completed!")
    print("\nNext steps:")
    print("1. Start the service: python -m api.app")
    print("2. Test the service: python example_usage.py")
    print("3. View API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
