"""
Configuration management system
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import sys

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize config manager
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._environment = self._detect_environment()
        
    def _detect_environment(self) -> str:
        """Detect if running in Colab or locally"""
        if 'google.colab' in sys.modules:
            return 'colab'
        else:
            return 'local'
    
    def load_config(self, config_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration file
        
        Args:
            config_name: Name of config file (without extension)
            validate: Whether to validate the config
            
        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]
        
        # Try to find config file
        config_path = None
        for ext in ['.yaml', '.yml', '.json']:
            candidate = self.config_dir / f"{config_name}{ext}"
            if candidate.exists():
                config_path = candidate
                break
        
        if not config_path:
            raise FileNotFoundError(f"Config file not found: {config_name}")
        
        # Load config
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Apply environment-specific overrides
        config = self._apply_environment_overrides(config)
        
        # Validate if requested
        if validate:
            self._validate_config(config_name, config)
        
        self._configs[config_name] = config
        logger.info(f"Loaded config: {config_name} for environment: {self._environment}")
        
        return config
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides"""
        # For data config, use environment-specific paths
        if 'local' in config and 'colab' in config:
            env_config = config.get(self._environment, {})
            
            # Replace top-level paths with environment-specific ones
            for key, value in env_config.items():
                config[key] = value
        
        return config
    
    def _validate_config(self, config_name: str, config: Dict[str, Any]):
        """Validate configuration based on config type"""
        if config_name == 'data_config':
            self._validate_data_config(config)
        elif config_name == 'model_config':
            self._validate_model_config(config)
        elif config_name == 'frame_definitions':
            self._validate_frame_definitions(config)
    
    def _validate_data_config(self, config: Dict[str, Any]):
        """Validate data configuration"""
        # Check required sections
        required_sections = ['loading', 'splitting', 'human_coding', 'metadata']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in data_config: {section}")
        
        # Validate splitting ratios
        splitting = config['splitting']
        total_ratio = splitting['train_ratio'] + splitting['val_ratio'] + splitting['test_ratio']
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Check frame types
        frame_types = config['human_coding']['frame_types']
        expected_frames = ['underrepresentation', 'overrepresentation', 'obstacles', 'successes']
        if set(frame_types) != set(expected_frames):
            logger.warning(f"Frame types don't match expected: {expected_frames}")
    
    def _validate_model_config(self, config: Dict[str, Any]):
        """Validate model configuration"""
        # Check required sections
        required_sections = ['zero_shot', 'fine_tuning', 'preprocessing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in model_config: {section}")
        
        # Validate learning rate
        lr = config['fine_tuning']['learning_rate']
        if not (1e-6 <= lr <= 1e-2):
            logger.warning(f"Learning rate {lr} seems unusual")
        
        # Validate batch sizes
        batch_size = config['fine_tuning']['batch_size']
        if batch_size <= 0 or batch_size > 128:
            logger.warning(f"Batch size {batch_size} seems unusual")
    
    def _validate_frame_definitions(self, config: Dict[str, Any]):
        """Validate frame definitions"""
        # Check that all expected frames are defined
        expected_frames = ['underrepresentation', 'overrepresentation', 'obstacles', 'successes']
        frames = config.get('frames', {})
        
        for frame in expected_frames:
            if frame not in frames:
                raise ValueError(f"Missing frame definition: {frame}")
            
            frame_def = frames[frame]
            if 'definition' not in frame_def:
                raise ValueError(f"Missing definition for frame: {frame}")
            if 'keywords' not in frame_def:
                raise ValueError(f"Missing keywords for frame: {frame}")
    
    def get_paths(self) -> Dict[str, Path]:
        """Get environment-appropriate paths"""
        data_config = self.load_config('data_config')
        
        paths = {}
        path_keys = ['data_dir', 'articles_file', 'sample_articles', 'processed_data',
                    'cache_dir', 'models_dir', 'results_dir', 'logs_dir']
        
        for key in path_keys:
            if key in data_config:
                paths[key] = Path(data_config[key])
        
        return paths
    
    def get_frame_definitions(self) -> Dict[str, Any]:
        """Get frame definitions"""
        return self.load_config('frame_definitions')
    
    def get_model_params(self, model_type: str = 'zero_shot') -> Dict[str, Any]:
        """Get model parameters for specific model type"""
        model_config = self.load_config('model_config')
        
        if model_type not in model_config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_config[model_type]
    
    def create_directories(self):
        """Create necessary directories"""
        paths = self.get_paths()
        
        # Create directories that should exist
        for key, path in paths.items():
            if key.endswith('_dir'):
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """Update configuration in memory"""
        if config_name not in self._configs:
            self.load_config(config_name)
        
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        recursive_update(self._configs[config_name], updates)
        logger.info(f"Updated config: {config_name}")
    
    def save_config(self, config_name: str, output_path: Optional[Path] = None):
        """Save configuration to file"""
        if config_name not in self._configs:
            raise ValueError(f"Config not loaded: {config_name}")
        
        if output_path is None:
            output_path = self.config_dir / f"{config_name}.yaml"
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(self._configs[config_name], f, indent=2)
        
        logger.info(f"Saved config: {config_name} to {output_path}")


# Global config manager instance
_config_manager = None

def get_config_manager(config_dir: Union[str, Path] = "configs") -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


def load_config(config_name: str) -> Dict[str, Any]:
    """Convenience function to load configuration"""
    return get_config_manager().load_config(config_name)


def get_paths() -> Dict[str, Path]:
    """Convenience function to get paths"""
    return get_config_manager().get_paths()


def setup_environment():
    """Setup environment based on configuration"""
    config_manager = get_config_manager()
    
    # Create necessary directories
    config_manager.create_directories()
    
    # Setup logging
    data_config = config_manager.load_config('data_config')
    log_level = data_config.get('logging', {}).get('level', 'INFO')
    
    from .utils import setup_logging
    setup_logging(level=log_level)
    
    logger.info(f"Environment setup complete for: {config_manager._environment}")