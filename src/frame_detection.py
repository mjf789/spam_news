"""
Frame detection models and base classes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)


class BaseFrameDetector(ABC):
    """Abstract base class for frame detection models"""
    
    @abstractmethod
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Predict frames in text"""
        pass
    
    @abstractmethod
    def predict_proba(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Predict frame probabilities"""
        pass


class ZeroShotFrameDetector(BaseFrameDetector):
    """Zero-shot frame detection using pre-trained models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: Optional[int] = None):
        """
        Initialize zero-shot classifier
        
        Args:
            model_name: HuggingFace model for zero-shot classification
            device: Device to run on (0 for GPU, -1 for CPU, None for auto)
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
            
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        
        # Define frame labels with descriptions
        self.frame_labels = {
            'underrepresentation': 'underrepresentation in leadership positions',
            'overrepresentation': 'overrepresentation in leadership positions',
            'obstacles': 'barriers and obstacles to leadership',
            'successes': 'achievements and successes in leadership'
        }
        
        self.threshold = 0.5  # Confidence threshold
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Predict frames using zero-shot classification"""
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        results = []
        for t in texts:
            # Run classification
            output = self.classifier(
                t,
                candidate_labels=list(self.frame_labels.values()),
                multi_label=True
            )
            
            # Extract predictions above threshold
            predictions = {
                'frames': [],
                'scores': {}
            }
            
            for label, score in zip(output['labels'], output['scores']):
                # Map back to frame name
                frame_name = [k for k, v in self.frame_labels.items() if v == label][0]
                predictions['scores'][frame_name] = score
                
                if score >= self.threshold:
                    predictions['frames'].append(frame_name)
            
            results.append(predictions)
        
        return results[0] if single_input else results
    
    def predict_proba(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Get probability scores for each frame"""
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        results = []
        for t in texts:
            output = self.classifier(
                t,
                candidate_labels=list(self.frame_labels.values()),
                multi_label=True
            )
            
            # Create probability dict
            proba = {}
            for label, score in zip(output['labels'], output['scores']):
                frame_name = [k for k, v in self.frame_labels.items() if v == label][0]
                proba[frame_name] = score
            
            results.append(proba)
        
        return results[0] if single_input else results
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold for predictions"""
        self.threshold = threshold


class TransformerFrameDetector(BaseFrameDetector):
    """Fine-tuned transformer model for frame detection"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize with fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model or HuggingFace model ID
            device: Device to run on
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        
        # Frame labels (order matters for model output)
        self.frame_labels = ['underrepresentation', 'overrepresentation', 
                           'obstacles', 'successes']
        self.threshold = 0.5
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Predict frames using fine-tuned model"""
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        results = []
        
        with torch.no_grad():
            for t in texts:
                # Tokenize
                inputs = self.tokenizer(
                    t,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                
                # Extract predictions
                predictions = {
                    'frames': [],
                    'scores': {}
                }
                
                for i, frame in enumerate(self.frame_labels):
                    score = float(probs[i])
                    predictions['scores'][frame] = score
                    
                    if score >= self.threshold:
                        predictions['frames'].append(frame)
                
                results.append(predictions)
        
        return results[0] if single_input else results
    
    def predict_proba(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Get probability scores for each frame"""
        results = self.predict(text)
        if isinstance(results, dict):
            return results['scores']
        else:
            return [r['scores'] for r in results]
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold for predictions"""
        self.threshold = threshold


class EnsembleFrameDetector(BaseFrameDetector):
    """Ensemble of multiple frame detection models"""
    
    def __init__(self, detectors: List[BaseFrameDetector], weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            detectors: List of frame detector instances
            weights: Optional weights for each detector (must sum to 1)
        """
        self.detectors = detectors
        
        if weights is None:
            weights = [1.0 / len(detectors)] * len(detectors)
        else:
            assert len(weights) == len(detectors)
            assert abs(sum(weights) - 1.0) < 1e-6
            
        self.weights = weights
        self.threshold = 0.5
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Ensemble prediction"""
        # Get probabilities from all detectors
        all_probs = self.predict_proba(text)
        
        # Convert to predictions based on threshold
        single_input = isinstance(text, str)
        probs_list = [all_probs] if single_input else all_probs
        
        results = []
        for probs in probs_list:
            predictions = {
                'frames': [],
                'scores': probs
            }
            
            for frame, score in probs.items():
                if score >= self.threshold:
                    predictions['frames'].append(frame)
            
            results.append(predictions)
        
        return results[0] if single_input else results
    
    def predict_proba(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Weighted average of probabilities from all detectors"""
        single_input = isinstance(text, str)
        
        # Get predictions from all detectors
        all_predictions = []
        for detector in self.detectors:
            preds = detector.predict_proba(text)
            all_predictions.append(preds)
        
        # Aggregate predictions
        if single_input:
            # Single text input
            aggregated = {}
            frames = all_predictions[0].keys()
            
            for frame in frames:
                weighted_score = sum(
                    pred.get(frame, 0) * weight 
                    for pred, weight in zip(all_predictions, self.weights)
                )
                aggregated[frame] = weighted_score
            
            return aggregated
        else:
            # Multiple texts
            results = []
            num_texts = len(all_predictions[0])
            
            for i in range(num_texts):
                aggregated = {}
                frames = all_predictions[0][i].keys()
                
                for frame in frames:
                    weighted_score = sum(
                        pred[i].get(frame, 0) * weight 
                        for pred, weight in zip(all_predictions, self.weights)
                    )
                    aggregated[frame] = weighted_score
                
                results.append(aggregated)
            
            return results
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold for predictions"""
        self.threshold = threshold


class FrameAnalyzer:
    """High-level interface for frame analysis"""
    
    def __init__(self, detector: BaseFrameDetector, preprocessor=None, segmenter=None):
        """
        Initialize analyzer
        
        Args:
            detector: Frame detection model
            preprocessor: Optional preprocessor instance
            segmenter: Optional segmenter instance
        """
        self.detector = detector
        self.preprocessor = preprocessor
        self.segmenter = segmenter
    
    def analyze_article(self, article: Dict) -> Dict:
        """
        Comprehensive frame analysis of an article
        
        Args:
            article: Article dict with content
            
        Returns:
            Analysis results with frames, demographics, and counts
        """
        # Preprocess if available
        if self.preprocessor:
            processed = self.preprocessor.preprocess_article(article)
            text = processed['cleaned_content']
            demographics = processed['demographics_found']
        else:
            text = article['content']
            demographics = {}
        
        # Segment if available
        if self.segmenter:
            segments = self.segmenter.segment_for_analysis(article)
            analysis_units = segments['windows']
        else:
            analysis_units = [{'text': text}]
        
        # Detect frames in each segment
        frame_results = []
        for unit in analysis_units:
            unit_text = unit['text']
            predictions = self.detector.predict(unit_text)
            
            frame_results.append({
                'text': unit_text,
                'frames': predictions['frames'],
                'scores': predictions['scores'],
                'demographics': self._extract_demographics(unit_text) if not demographics else demographics
            })
        
        # Aggregate results
        aggregated = self._aggregate_results(frame_results)
        
        return {
            'article_id': article.get('article_id'),
            'frames_detected': aggregated['frames'],
            'frame_scores': aggregated['scores'],
            'demographics': aggregated['demographics'],
            'frame_counts': aggregated['counts'],
            'segments_analyzed': len(analysis_units),
            'detailed_results': frame_results if len(frame_results) > 1 else None
        }
    
    def _extract_demographics(self, text: str) -> List[str]:
        """Simple demographic extraction (placeholder)"""
        # This would use the demographic detection logic
        demographics = []
        text_lower = text.lower()
        
        if 'women' in text_lower or 'woman' in text_lower:
            demographics.append('women')
        if 'men' in text_lower or 'man' in text_lower:
            demographics.append('men')
        # Add more demographic detection...
        
        return demographics
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate frame results across segments"""
        all_frames = []
        all_demographics = []
        frame_scores = {'underrepresentation': [], 'overrepresentation': [], 
                       'obstacles': [], 'successes': []}
        
        for result in results:
            all_frames.extend(result['frames'])
            all_demographics.extend(result.get('demographics', []))
            
            for frame, score in result['scores'].items():
                frame_scores[frame].append(score)
        
        # Count frames
        from collections import Counter
        frame_counts = Counter(all_frames)
        demo_counts = Counter(all_demographics)
        
        # Average scores
        avg_scores = {}
        for frame, scores in frame_scores.items():
            avg_scores[frame] = np.mean(scores) if scores else 0.0
        
        return {
            'frames': list(set(all_frames)),
            'scores': avg_scores,
            'demographics': list(set(all_demographics)),
            'counts': {
                'frames': dict(frame_counts),
                'demographics': dict(demo_counts)
            }
        }