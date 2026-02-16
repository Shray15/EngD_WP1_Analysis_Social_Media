# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md with project overview and setup instructions
- requirements.txt with all necessary dependencies
- .gitignore file configured for Python/ML projects
- config.py for centralized configuration management
- CONTRIBUTING.md with guidelines for contributors
- setup.py for automated project setup and validation
- Proper documentation headers for all utility modules

### Changed
- Refactored hardcoded file paths in fine_tune_bertje.py to use config.py
- Updated text preprocessing module with comprehensive documentation
- Improved project structure for better maintainability

### Fixed
- Removed hardcoded absolute paths that were system-specific
- Standardized column names and model configurations

## [1.0.0] - Initial Research Implementation

### Added
- Intent detection using BERT (Dutch), DeBERTa, and RoBERTa models
- Sentiment analysis pipeline with model evaluation
- Discourse analysis with temporal clustering (Q1 2018 - Q1 2023)
- Text preprocessing utilities for social media content
- K-means clustering for comment analysis
- Multinomial logistic regression for discourse modeling
- Synthetic data generation for model training
- Comprehensive Jupyter notebooks for each analysis component
- Visualization tools for temporal discourse patterns
- Similarity analysis for post relatedness assessment

### Models Implemented
- GRoNLP/bert-base-dutch-cased for Dutch intent classification
- microsoft/deberta-v3-base for advanced sequence classification  
- roberta-base for robust intent recognition
- Custom preprocessing pipeline for Dutch social media text

### Analysis Features
- Stratified k-fold cross-validation for model evaluation
- Quarterly discourse pattern analysis
- Statistical modeling of post features vs discourse types
- Clustering-based content similarity assessment
- Author name extraction and privacy protection utilities

---

## Development Notes

### Research Context
This project represents Work Package 1 of an Engineering Doctorate program focused on social media analysis. The implementation includes state-of-the-art NLP techniques adapted for Dutch language social media content.

### Technical Implementation
- PyTorch-based transformer fine-tuning with Hugging Face
- Scikit-learn integration for traditional ML approaches
- Advanced text preprocessing maintaining linguistic features
- GPU-optimized training with CUDA support
- Reproducible research practices with fixed random seeds

### Data Handling
- Privacy-preserving data processing utilities
- Synthetic data augmentation for improved model robustness
- Temporal data aggregation and analysis frameworks
- Cross-validation strategies for reliable performance estimation