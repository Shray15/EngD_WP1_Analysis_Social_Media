# Social Media Analysis for EngD Work Package 1

This repository contains the implementation and analysis code for Work Package 1 of an Engineering Doctorate (EngD) research project focused on social media analysis. The project encompasses sentiment analysis, intent detection, discourse analysis, and relatedness assessment of social media content.

## 📊 Project Overview

This research project analyzes social media data through multiple dimensions:
- **Sentiment Detection**: Classifying emotional tone of social media posts
- **Intent Recognition**: Identifying user intentions and motivations
- **Discourse Analysis**: Understanding conversation patterns and themes over time
- **Relatedness Assessment**: Measuring similarity between posts and content

## 🏗️ Repository Structure

```
├── Combine sentiment intent/     # Integration of sentiment and intent analysis
├── data_utils/                   # Data preprocessing and utility functions
├── discourse/                    # Discourse analysis and temporal modeling
├── intent_detection/            # Intent classification models and training
├── intent_utils/                # Intent preprocessing utilities  
├── relatedness/                 # Content similarity and relatedness analysis
├── sentiment_detection/         # Sentiment analysis models and evaluation
└── requirements.txt             # Python dependencies
```

### Key Components

#### 🎯 Intent Detection
- **Fine-tuning Scripts**: `fine_tune_bertje.py`, `fine_tune_deberta.py`, `fine_tune_robert.py`
- **Models**: BERT (Dutch), DeBERTa, RoBERTa fine-tuned for intent classification
- **Synthetic Data**: Generated training data for improved model performance
- **Prediction & Visualization**: `intent_pred_and_plot.ipynb`

#### 😊 Sentiment Detection  
- **Model Evaluation**: `models_evaluation_sentiment.ipynb`
- **Prediction & Plotting**: `sentiment_prediction_and_plot.ipynb`

#### 💬 Discourse Analysis
- **Temporal Analysis**: `discourse_over_time.ipynb` - tracks discourse patterns across time periods (Q1 2018 - Q1 2023)
- **Feature Analysis**: Multiple notebooks analyzing post features and discourse relationships
- **Clustering**: K-means clustering analysis of comment probabilities
- **Multinomial Logistic Modeling**: Advanced statistical modeling of discourse types

#### 📈 Data Processing
- **Author Extraction**: `extract_author_name.ipynb`
- **Data Cleaning**: `remove_names.ipynb` 
- **Preprocessing**: Text normalization, HTML cleaning, URL/mention standardization

#### 🔗 Relatedness Analysis
- **Similarity Checks**: `Sim_checks_other_posts.ipynb` for cross-post similarity analysis

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- CUDA-capable GPU (recommended for model training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd EngD_WP1_Analysis_Social_Media
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (if not automatically downloaded)
   - The scripts will automatically download pre-trained models from Hugging Face
   - Dutch BERT: `GRoNLP/bert-base-dutch-cased`
   - DeBERTa and RoBERTa models as specified in fine-tuning scripts

### Quick Start

1. **Data Preparation**
   - Ensure your data files are in the expected locations
   - Update file paths in the notebooks/scripts to match your data location
   - Run data preprocessing utilities in `data_utils/`

2. **Model Training**
   ```bash
   # Train intent detection models
   python intent_detection/fine_tune_bertje.py
   python intent_detection/fine_tune_deberta.py  
   python intent_detection/fine_tune_robert.py
   ```

3. **Analysis Notebooks**
   - Open Jupyter notebooks in respective directories
   - Run cells sequentially for analysis
   - Modify data paths as needed for your environment

## 📊 Models and Performance

### Intent Detection Models
- **BERT (Dutch)**: Fine-tuned `GRoNLP/bert-base-dutch-cased`
- **DeBERTa**: Advanced transformer for sequence classification
- **RoBERTa**: Robustly optimized BERT pre-training approach

All models use stratified k-fold cross-validation for robust evaluation.

### Text Preprocessing
The project includes sophisticated text preprocessing with:
- HTML entity unescaping and tag removal
- Unicode normalization (NFKC)
- URL, email, mention, and number standardization
- Punctuation normalization while preserving semantic markers
- Case and diacritic preservation for Dutch language processing

## 📈 Analysis Features

### Temporal Discourse Analysis
- Quarterly aggregation of discourse patterns (Q1 2018 - Q1 2023)
- Moving average smoothing for trend identification
- Cluster-based discourse type classification

### Statistical Modeling
- Multinomial logistic regression for discourse type prediction
- K-means clustering for content similarity
- Feature extraction from post characteristics

### Visualization
- Time series plots of discourse evolution
- Confusion matrices for model evaluation
- Cluster visualization and analysis

## 🔧 Configuration

### Model Configuration
- **Max sequence length**: 384 tokens
- **Random seed**: 42 (for reproducibility)
- **Training approach**: Stratified k-fold cross-validation

### Data Requirements
- Social media posts with timestamps
- Labeled data for supervised learning
- Synthetic data generation for data augmentation

## 📁 Data Structure Expected

```
Data files should follow these patterns:
- Comments with timestamps: `Comments_time` column
- Intent labels: `Intent` column  
- Synthetic data: `Synthetic Data` column
- Text content: Preprocessed text ready for model input
```

## 🤝 Contributing

This is a research project. For questions or collaboration:
1. Review the methodology in the notebooks
2. Check model implementations in the Python scripts
3. Ensure reproducibility by following the preprocessing steps

## 📝 Citation

If you use this work in your research, please cite the relevant EngD work package documentation and methodology.

## ⚠️ Important Notes

- **Data Privacy**: Ensure compliance with data protection regulations
- **Model Paths**: Update absolute file paths to match your environment
- **Dependencies**: Some notebooks may require additional packages not listed in requirements.txt
- **GPU Requirements**: Model training is optimized for GPU execution
- **Language**: Models are primarily optimized for Dutch language content

## 🔄 Workflow Overview

1. **Data Collection & Preprocessing** → `data_utils/`
2. **Model Training** → `intent_detection/`, `sentiment_detection/`  
3. **Analysis & Evaluation** → Individual analysis notebooks
4. **Discourse Analysis** → `discourse/` temporal and statistical analysis
5. **Integration** → `Combine sentiment intent/` for multi-modal analysis
6. **Similarity Assessment** → `relatedness/` for content relationships

## 📊 Expected Outputs

- Trained classification models for intent and sentiment
- Temporal analysis of discourse patterns
- Statistical models of post features and discourse types
- Visualization of trends and model performance
- Clustering results and similarity metrics

---

**Research Context**: This work is part of an Engineering Doctorate program investigating social media analysis techniques for understanding public discourse and communication patterns.