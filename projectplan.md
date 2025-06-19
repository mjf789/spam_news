# Project Plan: Deep Learning Analysis of Thesis Articles

## Project Overview
Build a deep learning system to analyze the same 462 news articles from Quincy Lherisson's thesis, automatically detecting and quantifying the four framing patterns (underrepresentation, overrepresentation, obstacles, successes) that were originally coded by hand. This project is designed to work seamlessly with Git version control and Google Colab.

## Phase 1: Repository Setup & Data Access

### Checkpoint 1.1: Git-Friendly Project Structure ✅
- [x] Create .gitignore for data files, models, and credentials
- [x] Set up project structure:
  ```
  spam_news/
  ├── notebooks/          # Colab notebooks
  ├── src/               # Python modules
  ├── configs/           # Configuration files
  ├── data/              # Local data (gitignored)
  ├── models/            # Saved models (gitignored)
  ├── results/           # Analysis results
  └── requirements.txt   # Dependencies
  ```
- [x] Create setup.py for package installation
- [x] Create README.md with setup instructions
- [x] Add sample data file for testing (5-10 articles)

### Checkpoint 1.2: Google Colab Integration ✅
- [x] Create main analysis notebook for Colab
- [x] Add Google Drive mounting code snippet
- [x] Create data loading utilities that work with Drive
- [x] Add GPU runtime detection and setup
- [x] Create environment setup cell with pip installs
- [x] Add progress saving to prevent loss on disconnection

### Checkpoint 1.3: Data Pipeline Setup ✅
- [x] Build data loader class for Google Drive articles
- [x] Create article preprocessing pipeline
- [x] Implement caching system for processed data
- [x] Add human coding data parser
- [x] Create train/val/test split function (60/20/20)
- [x] Build data validation checks

## Phase 2: Modular Code Development

### Checkpoint 2.1: Core Processing Modules
- [ ] Create `src/preprocessing.py` with text cleaning functions
- [ ] Build `src/segmentation.py` for article chunking
- [ ] Implement `src/feature_extraction.py` for keyword detection
- [ ] Create `src/frame_detection.py` base classes
- [ ] Add `src/utils.py` for common utilities
- [ ] Write unit tests for each module

### Checkpoint 2.2: Configuration Management
- [ ] Create `configs/model_config.yaml` for hyperparameters
- [ ] Add `configs/data_config.yaml` for data paths
- [ ] Build `configs/frame_definitions.yaml` from coding guidelines
- [ ] Create config loader with validation
- [ ] Add Colab-specific path overrides

## Phase 3: Pre-trained Model Strategy

### Checkpoint 3.1: Base Model Selection
- [ ] Use HuggingFace transformers library for pre-trained models
- [ ] Primary candidate: `roberta-base` (125M params, excellent for classification)
- [ ] Backup options: 
  - `distilbert-base-uncased` (66M params, faster/lighter)
  - `deberta-v3-base` (184M params, state-of-the-art)
- [ ] Test models on sample articles for memory/speed
- [ ] Implement model caching to avoid re-downloading

### Checkpoint 3.2: Multi-task Classification Approach
- [ ] Use ONE model with multi-label classification head
- [ ] 4 output labels: underrep, overrep, obstacles, successes
- [ ] Fine-tune pre-trained model on our labeled data
- [ ] Alternative: Use existing models from HuggingFace:
  - [ ] Test `cardiffnlp/twitter-roberta-base-sentiment` for tone
  - [ ] Try `dslim/bert-base-NER` for demographic detection
  - [ ] Explore `zero-shot-classification` models

### Checkpoint 3.3: Demographic Detection Strategy
- [ ] Use existing NER models as base:
  - [ ] `dslim/bert-base-NER` for person detection
  - [ ] `Davlan/bert-base-multilingual-cased-ner-hrl` 
- [ ] Add custom rules for gender/race terms from thesis
- [ ] Create demographic-frame association logic
- [ ] No need to train from scratch - just adapt existing models

## Phase 4: Colab-Ready Training Pipeline

### Checkpoint 4.1: Training Infrastructure
- [ ] Create `train_model.ipynb` notebook
- [ ] Build training data generator
- [ ] Implement batch processing for memory efficiency
- [ ] Add wandb/tensorboard integration
- [ ] Create model saving to Google Drive
- [ ] Add training resumption capability

### Checkpoint 4.2: Efficient Training Loop
- [ ] Implement gradient accumulation for small batches
- [ ] Add mixed precision training
- [ ] Create learning rate scheduling
- [ ] Build early stopping mechanism
- [ ] Add periodic evaluation during training
- [ ] Implement model ensemble training

### Checkpoint 4.3: Counting System Implementation
- [ ] Create `ExemplarCounter` class
- [ ] Build confidence-based thresholding
- [ ] Implement sliding window counting
- [ ] Add demographic-frame pairing
- [ ] Create results aggregation pipeline

## Phase 5: Evaluation & Results

### Checkpoint 5.1: Evaluation Notebook
- [ ] Create `evaluate_model.ipynb`
- [ ] Implement ICC calculation functions
- [ ] Build confusion matrix visualizations
- [ ] Add per-frame performance metrics
- [ ] Create demographic accuracy analysis
- [ ] Generate comparison plots

### Checkpoint 5.2: Results Analysis
- [ ] Build results processing pipeline
- [ ] Create statistical analysis functions
- [ ] Generate thesis-compatible tables
- [ ] Implement results export to CSV/JSON
- [ ] Add visualization dashboard
- [ ] Create error analysis reports

### Checkpoint 5.3: Reproducibility Package
- [ ] Create `reproduce_results.ipynb`
- [ ] Add seed setting for reproducibility
- [ ] Document all hyperparameters
- [ ] Create results comparison tools
- [ ] Build automated testing suite
- [ ] Generate final report template

## Phase 6: Deployment & Sharing

### Checkpoint 6.1: GitHub Repository Setup
- [x] Create comprehensive README.md
- [x] Add LICENSE file
- [ ] Create CONTRIBUTING.md guidelines
- [ ] Add example notebooks with outputs
- [ ] Create GitHub Actions for testing
- [ ] Add badges for build status

### Checkpoint 6.2: Colab Demo Notebook
- [x] Create `demo_analysis.ipynb` for new users
- [x] Add "Open in Colab" buttons to README
- [ ] Include sample results visualization
- [ ] Create step-by-step tutorial
- [ ] Add troubleshooting section
- [ ] Include performance benchmarks

### Checkpoint 6.3: Package Distribution
- [ ] Create pip-installable package structure
- [ ] Add `__init__.py` files to modules
- [ ] Create command-line interface
- [ ] Build Docker container (optional)
- [ ] Add continuous integration
- [ ] Create release versioning system

## Key Technical Decisions for Git/Colab

1. **Data Management**: Store articles in Google Drive, only sample data in Git
2. **Model Storage**: Use Google Drive for model checkpoints, Git LFS for small models
3. **Dependencies**: Keep requirements minimal for Colab compatibility
4. **Notebooks**: Create self-contained notebooks that install dependencies
5. **Configuration**: Use YAML files for easy modification in Colab

## Model Strategy Explained

### Why These Models Are Good:

1. **RoBERTa-base** (Recommended)
   - Pre-trained on 160GB of text
   - Excellent at understanding context and nuance
   - Already knows English language patterns
   - Just needs fine-tuning for our specific frames

2. **Existing HuggingFace Models We Can Use:**
   - `zero-shot-classification` - Can classify without training!
   - `sentiment-analysis` - Detects positive/negative tone (useful for obstacles/successes)
   - `named-entity-recognition` - Identifies people, organizations
   - Many are already fine-tuned for similar tasks

3. **Multi-label Approach:**
   - ONE model can detect ALL 4 frames simultaneously
   - More efficient than 4 separate models
   - Captures relationships between frames

### Training Strategy:
- We're NOT training from scratch (would need millions of examples)
- We're fine-tuning: teaching existing models our specific frames
- With 462 articles, fine-tuning should work well
- Can also use zero-shot models that need NO training

## Project Structure for Git

```
spam_news/
├── README.md                    # Project overview & setup
├── requirements.txt             # Core dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Ignore data/models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_model.ipynb
│   ├── 03_evaluate_model.ipynb
│   └── demo_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── frame_detection.py
│   ├── demographic_extraction.py
│   └── evaluation.py
├── configs/
│   ├── model_config.yaml
│   └── frame_definitions.yaml
├── tests/
│   └── test_frame_detection.py
└── data/
    └── sample_articles.json     # 5-10 example articles
```

## Colab Setup Instructions

1. Mount Google Drive containing full article dataset
2. Clone GitHub repository
3. Install requirements
4. Load pre-trained models or train new ones
5. Run analysis pipeline
6. Save results back to Drive

## Success Criteria

- One-click setup in Google Colab
- Clear documentation for reproducibility
- Modular code for easy extension
- Achieve ICC ≥ 0.70 with human coding
- Process all 462 articles efficiently in Colab

## Review Section

[To be completed after implementation]