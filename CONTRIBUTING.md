# Contributing to Social Media Analysis Project

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing to the EngD Work Package 1 Social Media Analysis repository.

## 🤝 Types of Contributions

We welcome several types of contributions:

### 🐛 Bug Reports
- Report issues with model training or notebook execution
- Document data preprocessing problems
- Identify visualization or plotting errors

### 💡 Feature Requests
- Suggest new analysis methods or models
- Propose additional visualization techniques
- Recommend improvements to preprocessing pipelines

### 📚 Documentation
- Improve existing documentation
- Add code comments and docstrings
- Create tutorial notebooks or examples

### 🔧 Code Contributions
- Fix bugs in existing code
- Add new analysis methods
- Improve model performance
- Enhance data preprocessing utilities

## 📋 Before You Start

1. **Understand the Research Context**
   - This is an academic research project for Engineering Doctorate studies
   - Focus on social media analysis: sentiment, intent, discourse, and relatedness
   - Methods should be reproducible and well-documented

2. **Check Existing Work**
   - Review open issues before starting new work
   - Check if similar analysis already exists in the codebase
   - Look through existing notebooks for related implementations

## 🚀 Getting Started

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git clone <your-fork-url>
   cd EngD_WP1_Analysis_Social_Media
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # For development dependencies
   pip install pytest black flake8
   ```

3. **Set Up Configuration**
   ```python
   # Update config.py with your data paths
   from config import update_data_path
   update_data_path("synthetic_data", "your/data/path.xlsx")
   ```

4. **Verify Setup**
   ```python
   python config.py  # Run environment check
   ```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate

### Notebook Guidelines
- Clear cell structure with markdown explanations
- Include data source information and requirements
- Add visualizations with proper titles and labels
- Document any manual steps or data preprocessing

### Model Development
- Use stratified cross-validation for evaluation
- Set random seeds for reproducibility
- Document hyperparameter choices
- Include baseline comparisons where relevant

### Data Handling
- Never commit sensitive or private data
- Use relative paths in config.py
- Document data format requirements
- Include data validation steps

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_preprocessing.py
pytest tests/test_models.py
```

### Testing Guidelines
- Write tests for new functions and classes
- Include edge cases and error conditions
- Test with sample data (not full datasets)
- Verify reproducibility of results

## 📊 Research Standards

### Reproducibility
- Set random seeds (`SEED = 42` in config)
- Document software versions and dependencies
- Include complete preprocessing steps
- Save model configurations and hyperparameters

### Documentation
- Explain methodological choices
- Reference relevant literature
- Document limitations and assumptions
- Include performance metrics and evaluation details

### Ethical Considerations
- Respect data privacy and consent
- Follow institutional ethics guidelines
- Document data sources and collection methods
- Consider bias and fairness in model development

## 🔄 Submission Process

### Pull Request Guidelines

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make Changes**
   - Implement your changes with clear, focused commits
   - Add or update tests as needed
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run tests
   pytest

   # Check code style
   black . --check
   flake8 .
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   ```

5. **Submit Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots for visualizations
   - Explain impact on existing functionality

### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Changes Made
- List of specific changes
- New features or bug fixes
- Updated documentation

## Testing
- How the changes were tested
- Any new test cases added
- Verification of reproducibility

## Impact
- Effect on existing functionality  
- Performance implications
- Breaking changes (if any)

## Related Issues
Fixes #issue_number
```

## 🏷️ Issue Guidelines

### Bug Reports
```markdown
**Description**: Clear description of the bug
**Steps to Reproduce**: Minimal steps to reproduce the issue
**Expected Behavior**: What should happen
**Actual Behavior**: What actually happens
**Environment**: Python version, OS, relevant package versions
**Data**: Information about data being processed (if relevant)
```

### Feature Requests
```markdown
**Feature Description**: What you'd like to see added
**Use Case**: Why this feature would be valuable
**Proposed Implementation**: Ideas for how it could be implemented
**Alternatives**: Other approaches considered
```

## 🎯 Research Areas

Priority areas for contributions:

### Model Improvements
- Fine-tuning hyperparameters
- Exploring new transformer architectures
- Ensemble methods for better performance
- Multi-task learning approaches

### Analysis Methods
- Advanced temporal analysis techniques
- Novel clustering approaches
- Improved similarity metrics
- Statistical modeling enhancements

### Data Processing
- Better text preprocessing methods
- Handling multilingual content
- Noise reduction techniques
- Data augmentation strategies

### Visualization
- Interactive plotting capabilities
- Advanced time series visualizations
- Model interpretation plots
- Clustering visualization improvements

## ⚠️ Important Notes

- **Data Privacy**: Never commit actual social media data
- **Academic Integrity**: Maintain research ethics standards
- **Reproducibility**: Always test that others can run your code
- **Documentation**: Academic research requires thorough documentation

## 📞 Getting Help

- Open an issue for questions about the codebase
- Review existing documentation and notebooks
- Check the README for setup instructions
- Contact project maintainers for research-specific questions

## 📜 License and Citation

By contributing, you agree that your contributions will be used according to the project's academic research purposes. If this work is used in publications, appropriate attribution will be given to all contributors.

---

Thank you for contributing to advancing social media analysis research! 🚀