"""
Common utilities for the spam news analysis project
"""

import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_results(results: Dict[str, Any], output_path: Union[str, Path], 
                 format: str = 'json'):
    """
    Save analysis results to file
    
    Args:
        results: Results dictionary
        output_path: Path to save results
        format: Output format (json, csv, pickle)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    elif format == 'pickle':
        pd.to_pickle(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def calculate_icc(ratings1: np.ndarray, ratings2: np.ndarray) -> float:
    """
    Calculate Intraclass Correlation Coefficient (ICC)
    
    Args:
        ratings1: First set of ratings
        ratings2: Second set of ratings
        
    Returns:
        ICC value
    """
    n = len(ratings1)
    if n != len(ratings2):
        raise ValueError("Rating arrays must have same length")
    
    # Create data frame for ICC calculation
    data = pd.DataFrame({
        'target': list(range(n)) * 2,
        'rater': [0] * n + [1] * n,
        'rating': list(ratings1) + list(ratings2)
    })
    
    # Calculate mean squares
    grand_mean = data['rating'].mean()
    
    # Between-target variance
    target_means = data.groupby('target')['rating'].mean()
    ms_between = n * 2 * ((target_means - grand_mean) ** 2).mean()
    
    # Within-target variance
    ms_within = 0
    for target in range(n):
        target_ratings = data[data['target'] == target]['rating']
        ms_within += ((target_ratings - target_ratings.mean()) ** 2).sum()
    ms_within = ms_within / (n * (2 - 1))
    
    # Calculate ICC(2,1)
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    
    return icc


def calculate_agreement_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """
    Calculate agreement metrics between predictions and ground truth
    
    Args:
        predictions: List of prediction dicts with 'frames' key
        ground_truth: List of ground truth dicts with 'frames' key
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
    
    # Frame types
    frame_types = ['underrepresentation', 'overrepresentation', 'obstacles', 'successes']
    
    # Convert to binary arrays
    y_true = []
    y_pred = []
    
    for pred, truth in zip(predictions, ground_truth):
        # Binary encoding for each frame
        true_binary = [1 if frame in truth.get('frames', []) else 0 for frame in frame_types]
        pred_binary = [1 if frame in pred.get('frames', []) else 0 for frame in frame_types]
        
        y_true.append(true_binary)
        y_pred.append(pred_binary)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics for each frame
    metrics = {}
    
    for i, frame in enumerate(frame_types):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        
        metrics[f'{frame}_precision'] = precision
        metrics[f'{frame}_recall'] = recall
        metrics[f'{frame}_f1'] = f1
        
        # Cohen's kappa
        metrics[f'{frame}_kappa'] = cohen_kappa_score(y_true[:, i], y_pred[:, i])
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average='binary', zero_division=0
    )
    
    metrics['overall_precision'] = precision
    metrics['overall_recall'] = recall
    metrics['overall_f1'] = f1
    metrics['overall_accuracy'] = (y_true == y_pred).mean()
    
    return metrics


def merge_coding_with_predictions(articles_df: pd.DataFrame, 
                                 predictions: List[Dict]) -> pd.DataFrame:
    """
    Merge human coding with model predictions for comparison
    
    Args:
        articles_df: DataFrame with articles and human coding
        predictions: List of prediction dicts
        
    Returns:
        Merged DataFrame
    """
    # Add predictions to dataframe
    pred_df = pd.DataFrame(predictions)
    
    # Merge on article_id
    merged = articles_df.merge(
        pred_df,
        on='article_id',
        suffixes=('_human', '_predicted')
    )
    
    return merged


def create_frame_summary(coding_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for frame coding
    
    Args:
        coding_data: DataFrame with frame coding data
        
    Returns:
        Summary DataFrame
    """
    summary = []
    
    frame_types = ['underrepresentation', 'overrepresentation', 'obstacles', 'successes']
    demographic_groups = ['women', 'men', 'white_women', 'white_men', 
                         'women_of_color', 'men_of_color']
    
    for frame in frame_types:
        for demo in demographic_groups:
            # Count occurrences
            mask = (coding_data['frame_type'] == frame) & \
                   (coding_data['demographic_group'] == demo)
            count = coding_data[mask]['count'].sum()
            
            if count > 0:
                summary.append({
                    'frame': frame,
                    'demographic': demo,
                    'total_count': count,
                    'num_articles': coding_data[mask]['article_id'].nunique()
                })
    
    return pd.DataFrame(summary)


def format_results_for_thesis(results: Dict[str, Any]) -> str:
    """
    Format analysis results in thesis-compatible format
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Formatted string
    """
    output = []
    output.append("Frame Analysis Results")
    output.append("=" * 50)
    
    # Summary statistics
    if 'summary' in results:
        output.append("\nSummary Statistics:")
        for key, value in results['summary'].items():
            output.append(f"  {key}: {value}")
    
    # Frame frequencies
    if 'frame_frequencies' in results:
        output.append("\nFrame Frequencies:")
        for frame, freq in results['frame_frequencies'].items():
            output.append(f"  {frame}: {freq}")
    
    # Demographic breakdowns
    if 'demographic_analysis' in results:
        output.append("\nDemographic Analysis:")
        for demo, data in results['demographic_analysis'].items():
            output.append(f"\n  {demo}:")
            for key, value in data.items():
                output.append(f"    {key}: {value}")
    
    return "\n".join(output)


def get_device():
    """Get the appropriate device for PyTorch"""
    import torch
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger = logging.getLogger(__name__)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger = logging.getLogger(__name__)
        logger.info("Using CPU")
    
    return device


def hash_text(text: str) -> str:
    """Generate hash for text (useful for caching)"""
    return hashlib.md5(text.encode()).hexdigest()


def batch_iterator(items: List[Any], batch_size: int):
    """
    Yield batches of items
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        duration = (datetime.now() - self.start).total_seconds()
        self.logger.info(f"{self.name} took {duration:.2f} seconds")