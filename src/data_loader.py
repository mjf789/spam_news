"""
Data loading utilities for Google Drive and local files
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pickle
import hashlib
from datetime import datetime

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
    
    def get_train_val_test_split(self, val_size: float = 0.2, test_size: float = 0.2, 
                                   random_state: int = 42):
        """
        Split articles into train, validation, and test sets (60/20/20 by default)
        
        Args:
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        if self._df is None:
            self.load_articles()
            
        # Shuffle
        df_shuffled = self._df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Calculate split indices
        test_idx = int(len(df_shuffled) * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split
        train_df = df_shuffled[:val_idx]
        val_df = df_shuffled[val_idx:test_idx]
        test_df = df_shuffled[test_idx:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} articles")
        
        return train_df, val_df, test_df
    
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


    def validate_data(self) -> Dict[str, any]:
        """Validate loaded data and return validation report"""
        if self._df is None:
            self.load_articles()
            
        report = {
            'total_articles': len(self._df),
            'sources': self._df['source'].value_counts().to_dict(),
            'date_range': (self._df['date'].min(), self._df['date'].max()),
            'missing_content': self._df['content'].isna().sum(),
            'missing_coding': 0,
            'coding_stats': {}
        }
        
        # Check human coding
        if 'human_coding' in self._df.columns:
            coding_df = self.get_coding_data()
            if len(coding_df) > 0:
                report['coding_stats'] = {
                    'total_annotations': len(coding_df),
                    'frames': coding_df['frame_type'].value_counts().to_dict(),
                    'demographics': coding_df['demographic_group'].value_counts().to_dict()
                }
            else:
                report['missing_coding'] = len(self._df)
        
        # Validate each article
        issues = []
        for idx, row in self._df.iterrows():
            if pd.isna(row['content']) or len(row['content']) < 50:
                issues.append(f"Article {row['article_id']}: Content too short or missing")
            if 'human_coding' not in row or not row['human_coding']:
                issues.append(f"Article {row['article_id']}: No human coding")
        
        report['issues'] = issues
        report['is_valid'] = len(issues) == 0
        
        return report


class DataCache:
    """Caching system for preprocessed data"""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, data_path: str, preprocessing_version: str = "v1") -> str:
        """Generate cache key based on file path and version"""
        key_string = f"{data_path}_{preprocessing_version}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cache file"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def load_cached(self, data_path: str, preprocessing_version: str = "v1") -> Optional[any]:
        """Load data from cache if available"""
        cache_key = self._get_cache_key(data_path, preprocessing_version)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached data from {cache_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def save_cache(self, data: any, data_path: str, preprocessing_version: str = "v1"):
        """Save data to cache"""
        cache_key = self._get_cache_key(data_path, preprocessing_version)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear_cache(self):
        """Clear all cached files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")


def parse_human_coding(coding_data: Dict) -> Dict[str, Dict[str, int]]:
    """Parse human coding data into standardized format"""
    parsed = {
        'underrepresentation': {},
        'overrepresentation': {},
        'obstacles': {},
        'successes': {}
    }
    
    for frame_type, demographics in coding_data.items():
        if frame_type in parsed and isinstance(demographics, dict):
            parsed[frame_type] = demographics
    
    return parsed


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