# Quick Start Guide

This guide helps you get up and running with the Social Media Analysis project quickly.

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
git clone <your-repository-url>
cd EngD_WP1_Analysis_Social_Media
python setup.py  # This will install everything automatically
```

### 2. Configure Your Data Paths
```python
# Edit config.py
from config import update_data_path

# Update with your data file locations
update_data_path("synthetic_data", "path/to/your/synthetic_data.xlsx")
update_data_path("clustered_comments", "path/to/your/comments.csv")
```

### 3. Test Your Setup
```bash
python config.py  # Should show environment info
pytest tests/    # Run tests (optional)
```

## 🎯 Quick Analysis Examples

### Intent Classification
```python
# Train a Dutch BERT model for intent detection
python intent_detection/fine_tune_bertje.py
```

### Sentiment Analysis
```jupyter
# Open and run the sentiment analysis notebook
jupyter notebook sentiment_detection/sentiment_prediction_and_plot.ipynb
```

### Discourse Analysis Over Time
```jupyter
# Analyze temporal patterns in discourse
jupyter notebook discourse/discourse_over_time.ipynb
```

## 📊 Expected Workflow

1. **Data Preparation**
   - Place your data files in appropriate locations
   - Update paths in `config.py`
   - Run data preprocessing notebooks in `data_utils/`

2. **Model Training**
   - Choose your model: BERT (Dutch), DeBERTa, or RoBERTa
   - Run training scripts in `intent_detection/`
   - Monitor training progress and validation metrics

3. **Analysis**
   - Run sentiment analysis notebooks
   - Perform temporal discourse analysis 
   - Analyze post relatedness and clustering
   - Combine results using notebooks in `Combine sentiment intent/`

4. **Results**
   - Models saved to `models/` directory
   - Plots and visualizations in `plots/`
   - Analysis results in `results/`

## 🔧 Common Issues

### Data Path Errors
```python
# Update your data paths in config.py
DEFAULT_DATA_PATHS["synthetic_data"] = Path("your/actual/data/path.xlsx")
```

### Model Download Issues
```bash
# For Dutch language processing
python -m spacy download nl_core_news_sm
```

### GPU/CUDA Issues
```python
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Key Files to Know

| File | Purpose |
|------|---------|
| `config.py` | Central configuration - **UPDATE YOUR PATHS HERE** |
| `setup.py` | Automated setup and validation |
| `requirements.txt` | All Python dependencies |
| `README.md` | Comprehensive documentation |
| `tests/` | Test files for validation |

## 🎓 Learning Path

1. **Start Here**: Read the main README.md
2. **Setup**: Run `python setup.py`  
3. **Explore**: Open notebooks in Jupyter
4. **Experiment**: Try different models and parameters
5. **Analyze**: Review results and visualizations
6. **Contribute**: See CONTRIBUTING.md for guidelines

## ⚡ Pro Tips

- Always run `python config.py` to check your environment
- Use `pytest` to validate your setup
- Check the CHANGELOG.md for recent updates
- Start with smaller datasets for faster experimentation
- Use GPU training for better performance with large models
- Save intermediate results to avoid re-computation

## 📞 Getting Help

1. Check this guide and README.md first
2. Run the test suite: `pytest tests/`
3. Review error messages carefully - they often contain solutions
4. Check your data file paths and formats
5. Open an issue if you find bugs or need features

---

**Ready to start?** Run `python setup.py` and follow the prompts! 🎉