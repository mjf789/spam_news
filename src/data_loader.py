"""
Data loading utilities for Google Drive and local files
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ArticleDataLoader:
    """Loads news articles from various sources including Google Drive"""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data file or directory
        """
        self.data_path = Path(data_path) if data_path else None
        self._articles = None
        self._df = None
        
    def load_from_json(self, filepath: Union[str, Path]) -> List[Dict]:
        """Load articles from JSON file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
        logger.info(f"Loaded {len(articles)} articles from {filepath}")
        return articles
    
    def load_from_drive(self, drive_path: str) -> List[Dict]:
        """
        Load articles from Google Drive
        
        Args:
            drive_path: Path within Google Drive (e.g., 'MyDrive/spam_news_data/articles.json')
        """
        # Check if running in Colab
        try:
            from google.colab import drive
            drive_mount_point = '/content/drive'
            
            # Check if drive is mounted
            if not Path(drive_mount_point).exists():
                logger.info("Mounting Google Drive...")
                drive.mount(drive_mount_point)
                
            full_path = Path(drive_mount_point) / drive_path
            return self.load_from_json(full_path)
            
        except ImportError:
            raise RuntimeError("Google Colab not detected. Use load_from_json for local files.")
    
    def load_articles(self, source: Optional[str] = None) -> pd.DataFrame:
        """
        Load articles and return as DataFrame
        
        Args:
            source: Optional source to filter by
        """
        if self._articles is None:
            if self.data_path:
                self._articles = self.load_from_json(self.data_path)
            else:
                # Try to find data
                possible_paths = [
                    Path('data/articles.json'),
                    Path('data/sample_articles.json'),
                    Path('/content/drive/MyDrive/spam_news_data/articles.json')
                ]
                
                for path in possible_paths:
                    if path.exists():
                        self._articles = self.load_from_json(path)
                        break
                else:
                    raise FileNotFoundError("No data file found in common locations")
        
        # Convert to DataFrame
        df = pd.DataFrame(self._articles)
        
        # Filter by source if specified
        if source:
            df = df[df['source'] == source]
            
        # Add useful columns
        df['content_length'] = df['content'].str.len()
        df['word_count'] = df['content'].str.split().str.len()
        
        self._df = df
        return df
    
    def get_coding_data(self) -> pd.DataFrame:
        """Extract human coding data into a normalized format"""
        if self._df is None:
            self.load_articles()
            
        records = []
        for idx, row in self._df.iterrows():
            coding = row['human_coding']
            for frame_type, demographics in coding.items():
                for demo_group, count in demographics.items():
                    records.append({
                        'article_id': row['article_id'],
                        'source': row['source'],
                        'frame_type': frame_type,
                        'demographic_group': demo_group,
                        'count': count
                    })
        
        return pd.DataFrame(records)
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split articles into train and test sets
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        if self._df is None:
            self.load_articles()
            
        # Shuffle and split
        df_shuffled = self._df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * (1 - test_size))
        
        train_df = df_shuffled[:split_idx]
        test_df = df_shuffled[split_idx:]
        
        logger.info(f"Train set: {len(train_df)} articles, Test set: {len(test_df)} articles")
        
        return train_df, test_df
    
    def save_preprocessed(self, save_path: Union[str, Path], format: str = 'pickle'):
        """Save preprocessed data for faster loading"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._df is None:
            self.load_articles()
            
        if format == 'pickle':
            self._df.to_pickle(save_path)
        elif format == 'csv':
            self._df.to_csv(save_path, index=False)
        elif format == 'json':
            self._df.to_json(save_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        logger.info(f"Saved preprocessed data to {save_path}")


def setup_colab_paths():
    """Setup paths for Google Colab environment"""
    import sys
    
    if 'google.colab' in sys.modules:
        # Add project root to path
        project_root = Path('/content/spam_news')
        if project_root.exists():
            sys.path.insert(0, str(project_root))
            
        # Return common paths
        return {
            'project_root': project_root,
            'data_path': Path('/content/drive/MyDrive/spam_news_data'),
            'model_path': Path('/content/drive/MyDrive/spam_news_models'),
            'results_path': Path('/content/drive/MyDrive/spam_news_results')
        }
    else:
        # Local paths
        return {
            'project_root': Path.cwd(),
            'data_path': Path('data'),
            'model_path': Path('models'),
            'results_path': Path('results')
        }