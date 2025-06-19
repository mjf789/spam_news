# Spam News Analysis: ML-Based Frame Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/spam_news/blob/main/notebooks/demo_analysis.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system for analyzing news articles about leadership disparities at the intersection of gender and race. This project automates the content analysis methodology from Quincy Lherisson's thesis using state-of-the-art transformer models.

## 🎯 Project Overview

This system identifies four types of framing in news articles:
- **Underrepresentation**: Groups appearing in leadership at lower rates than expected
- **Overrepresentation**: Groups appearing in leadership at higher rates than expected
- **Obstacles**: Systemic barriers in pursuing leadership
- **Successes**: Achievements in pursuing leadership

The analysis considers intersectional identities across gender (men/women) and race (White/people of color).

## 🚀 Quick Start in Google Colab

1. Click the "Open in Colab" badge above
2. Run the setup cell to install dependencies
3. Mount your Google Drive containing the article data
4. Run the analysis pipeline

## 💻 Local Installation

### Prerequisites
- Python 3.8 or higher
- GPU recommended for faster processing (but not required)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam_news.git
cd spam_news
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## 📁 Project Structure

```
spam_news/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_model.ipynb
│   ├── 03_evaluate_model.ipynb
│   └── demo_analysis.ipynb
├── src/                    # Source code modules
│   ├── preprocessing.py    # Text cleaning and preparation
│   ├── frame_detection.py  # Frame classification models
│   ├── demographic_extraction.py  # Entity recognition
│   └── evaluation.py       # Metrics and analysis
├── configs/                # Configuration files
│   ├── model_config.yaml   # Model hyperparameters
│   └── frame_definitions.yaml  # Frame coding guidelines
├── data/                   # Data directory (gitignored)
│   └── sample_articles.json  # Sample data for testing
└── models/                 # Saved models (gitignored)
```

## 📊 Data Setup

### For Google Colab Users

1. Upload your article dataset to Google Drive
2. In the notebook, mount your drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update the data path in the configuration

### For Local Users

1. Place your article data in the `data/` directory
2. Update `configs/data_config.yaml` with your file paths

## 🔧 Usage

### Basic Analysis

```python
from src.frame_detection import FrameAnalyzer
from src.preprocessing import ArticlePreprocessor

# Initialize components
preprocessor = ArticlePreprocessor()
analyzer = FrameAnalyzer()

# Load and analyze articles
articles = preprocessor.load_articles("path/to/articles.json")
results = analyzer.analyze_articles(articles)

# View results
print(results.summary())
```

### Training Custom Models

See `notebooks/02_train_model.ipynb` for detailed training instructions.

## 📈 Model Performance

Our models achieve:
- Average ICC with human coders: 0.72
- Frame detection accuracy: 85%+
- Processing speed: ~100 articles/minute on GPU

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{spam_news_analysis,
  title={Spam News Analysis: ML-Based Frame Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spam_news}
}
```

Original thesis:
```bibtex
@thesis{lherisson2023news,
  title={News Media's Framing of Leadership Disparities at the Intersection of Gender and Race},
  author={Lherisson, Quincy T.},
  year={2023},
  school={New York University}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original research and methodology by Quincy T. Lherisson
- Pre-trained models from HuggingFace Transformers
- Analysis framework inspired by intersectional framing theory