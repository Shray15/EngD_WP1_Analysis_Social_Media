#!/usr/bin/env python3
"""
Setup script for Social Media Analysis project.

This script helps set up the project environment and validates the installation.
Run this after installing requirements.txt to ensure everything is working correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install requirements from requirements.txt."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def download_spacy_model():
    """Download Dutch spaCy model if needed."""
    print("🔤 Setting up Dutch language model...")
    try:
        import spacy
        # Try to load the model
        try:
            nlp = spacy.load("nl_core_news_sm")
            print("✅ Dutch spaCy model already available!")
        except OSError:
            # Model not found, try to download
            print("📥 Downloading Dutch spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "nl_core_news_sm"])
            print("✅ Dutch spaCy model downloaded!")
        return True
    except ImportError:
        print("⚠️  spaCy not available. Dutch language processing may be limited.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to download spaCy model: {e}")
        print("You may need to download it manually: python -m spacy download nl_core_news_sm")
        return True

def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/raw", 
        "data/processed",
        "data/synthetic",
        "models",
        "models/checkpoints",
        "results",
        "plots",
        "logs",
        "tests"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created!")
    return True

def test_imports():
    """Test that all critical imports work."""
    print("🧪 Testing critical imports...")
    
    critical_imports = [
        "pandas",
        "numpy", 
        "sklearn",
        "torch",
        "transformers",
        "matplotlib",
        "seaborn"
    ]
    
    failed_imports = []
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All critical imports successful!")
    return True

def test_pytorch_gpu():
    """Test PyTorch GPU availability."""
    print("🔥 Checking PyTorch GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            print(f"✅ CUDA available! GPU: {device_name}")
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available. Using CPU for training.")
            print("   For GPU training, install CUDA-enabled PyTorch:")
            print("   Visit: https://pytorch.org/get-started/locally/")
    except ImportError:
        print("❌ PyTorch not available!")
        return False
    
    return True

def create_sample_config():
    """Create or update config with sample paths."""
    print("⚙️  Setting up configuration...")
    
    try:
        from config import check_environment
        print("✅ Configuration module loaded!")
        
        # Run environment check
        check_environment()
        
        print("\n📝 To customize data paths:")
        print("   1. Edit config.py")
        print("   2. Update DEFAULT_DATA_PATHS with your file locations")
        print("   3. Run: python config.py to verify settings")
        
        return True
    except ImportError as e:
        print(f"⚠️  Could not load config module: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic functionality tests...")
    
    try:
        # Test text preprocessing
        sys.path.append("intent_utils")
        from intent_train_test_preprocess import preprocess
        
        test_text = "Hello world! Visit https://example.com @user123"
        processed = preprocess(test_text)
        print(f"  ✅ Text preprocessing: '{test_text[:30]}...' → '{processed[:30]}...'")
        
        return True
    except Exception as e:
        print(f"  ⚠️  Basic tests failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 Social Media Analysis Project Setup")
    print("=" * 60)
    print()
    
    success = True
    
    # Run all setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements), 
        ("Create Directories", create_directories),
        ("Test Imports", test_imports),
        ("PyTorch GPU", test_pytorch_gpu),
        ("Download spaCy Model", download_spacy_model),
        ("Configuration", create_sample_config),
        ("Basic Tests", run_basic_tests)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 40)
        step_success = step_func()
        if not step_success:
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Update file paths in config.py for your data")
        print("2. Review the README.md for usage instructions")
        print("3. Start with the Jupyter notebooks in each analysis folder")
        print("4. For model training, run: python intent_detection/fine_tune_bertje.py")
    else:
        print("⚠️  Setup completed with some issues.")
        print("Please resolve the issues above before proceeding.")
    print("=" * 60)

if __name__ == "__main__":
    main()