"""
Feature extraction for frame detection
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class FrameFeatureExtractor:
    """Extracts features relevant to frame detection"""
    
    def __init__(self):
        # Frame-specific vocabulary
        self.frame_lexicons = {
            'underrepresentation': {
                'strong': ['underrepresented', 'lacking', 'absence', 'scarcity', 'dearth'],
                'moderate': ['few', 'only', 'just', 'merely', 'small number'],
                'comparative': ['less than', 'fewer than', 'below', 'under'],
                'statistical': ['percent', 'percentage', 'minority', 'fraction']
            },
            'overrepresentation': {
                'strong': ['overrepresented', 'dominate', 'monopolize', 'control'],
                'moderate': ['majority', 'most', 'predominant', 'prevailing'],
                'comparative': ['more than', 'exceed', 'surpass', 'above'],
                'statistical': ['percent', 'percentage', 'lion\'s share']
            },
            'obstacles': {
                'structural': ['barrier', 'ceiling', 'wall', 'block', 'impediment'],
                'discrimination': ['bias', 'discrimination', 'prejudice', 'stereotypes'],
                'difficulty': ['struggle', 'challenge', 'difficulty', 'hardship'],
                'systemic': ['systemic', 'institutional', 'structural', 'entrenched']
            },
            'successes': {
                'achievement': ['achievement', 'accomplishment', 'success', 'triumph'],
                'milestone': ['first', 'breakthrough', 'milestone', 'landmark'],
                'advancement': ['promoted', 'appointed', 'elevated', 'advanced'],
                'recognition': ['award', 'honor', 'recognition', 'celebrated']
            }
        }
        
        # Demographic indicators
        self.demographic_terms = {
            'gender': {
                'women': ['women', 'woman', 'female', 'females', 'she', 'her'],
                'men': ['men', 'man', 'male', 'males', 'he', 'his']
            },
            'race': {
                'white': ['white', 'caucasian'],
                'black': ['black', 'african american', 'african-american'],
                'hispanic': ['hispanic', 'latino', 'latina', 'latinx'],
                'asian': ['asian', 'asian american', 'asian-american'],
                'indigenous': ['indigenous', 'native american', 'native'],
                'poc': ['people of color', 'minority', 'minorities', 'diverse']
            },
            'intersectional': {
                'women_of_color': ['women of color', 'black women', 'latina women', 'asian women'],
                'white_women': ['white women', 'caucasian women'],
                'white_men': ['white men', 'caucasian men'],
                'men_of_color': ['men of color', 'black men', 'latino men', 'asian men']
            }
        }
        
        # Leadership context terms
        self.leadership_terms = {
            'positions': ['ceo', 'executive', 'director', 'manager', 'president', 
                         'vice president', 'vp', 'chief', 'head', 'leader'],
            'organizations': ['company', 'corporation', 'firm', 'organization', 
                            'board', 'committee', 'department', 'division'],
            'sectors': ['business', 'corporate', 'government', 'academic', 
                       'education', 'nonprofit', 'public sector', 'private sector']
        }
        
        # Linguistic patterns
        self.linguistic_patterns = {
            'passive_voice': r'\b(was|were|been|being|is|are|am)\s+\w+ed\b',
            'active_voice': r'\b(holds?|leads?|manages?|directs?|heads?)\b',
            'comparison': r'\b(more|less|fewer|greater|higher|lower)\s+than\b',
            'statistics': r'\b\d+\.?\d*\s*(?:percent|%|percentage)\b',
            'absolute': r'\b(all|every|none|never|always|only)\b'
        }
    
    def extract_lexical_features(self, text: str) -> Dict[str, float]:
        """Extract frame-specific lexical features"""
        text_lower = text.lower()
        features = {}
        
        # Count frame indicators
        for frame, categories in self.frame_lexicons.items():
            frame_score = 0
            for category, terms in categories.items():
                category_score = 0
                for term in terms:
                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                    category_score += count
                    
                features[f'{frame}_{category}_count'] = category_score
                # Weight stronger indicators more heavily
                weight = 2.0 if category == 'strong' else 1.0
                frame_score += category_score * weight
                
            features[f'{frame}_total_score'] = frame_score
        
        # Normalize by text length
        text_length = len(text.split())
        for key in list(features.keys()):
            features[f'{key}_normalized'] = features[key] / text_length if text_length > 0 else 0
        
        return features
    
    def extract_demographic_features(self, text: str) -> Dict[str, int]:
        """Extract demographic mention features"""
        text_lower = text.lower()
        features = {}
        
        for category, groups in self.demographic_terms.items():
            for group_name, terms in groups.items():
                count = 0
                for term in terms:
                    # Use word boundaries for accurate matching
                    pattern = r'\b' + re.escape(term) + r'\b'
                    count += len(re.findall(pattern, text_lower))
                
                features[f'demo_{category}_{group_name}'] = count
        
        # Add co-occurrence features
        if features.get('demo_gender_women', 0) > 0 and features.get('demo_race_black', 0) > 0:
            features['demo_intersect_black_women'] = 1
        if features.get('demo_gender_men', 0) > 0 and features.get('demo_race_white', 0) > 0:
            features['demo_intersect_white_men'] = 1
            
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic pattern features"""
        features = {}
        
        for pattern_name, pattern in self.linguistic_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'ling_{pattern_name}_count'] = len(matches)
        
        # Sentence-level features
        sentences = text.split('.')
        features['ling_avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()])
        features['ling_num_sentences'] = len(sentences)
        
        # Question marks and exclamations (emotional language)
        features['ling_questions'] = text.count('?')
        features['ling_exclamations'] = text.count('!')
        
        return features
    
    def extract_statistical_features(self, text: str) -> Dict[str, any]:
        """Extract features related to statistics and numbers"""
        features = {
            'stats_has_percentages': 0,
            'stats_has_numbers': 0,
            'stats_has_comparisons': 0,
            'stats_percentage_values': []
        }
        
        # Find percentages
        percent_pattern = r'(\d+(?:\.\d+)?)\s*(?:percent|%)'
        percentages = re.findall(percent_pattern, text, re.IGNORECASE)
        if percentages:
            features['stats_has_percentages'] = 1
            features['stats_percentage_values'] = [float(p) for p in percentages]
            features['stats_min_percentage'] = min(features['stats_percentage_values'])
            features['stats_max_percentage'] = max(features['stats_percentage_values'])
        
        # Find raw numbers
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        features['stats_has_numbers'] = 1 if numbers else 0
        features['stats_count_numbers'] = len(numbers)
        
        # Find comparisons
        comparison_pattern = r'(\d+(?:\.\d+)?)\s*times\s*(more|less|higher|lower)'
        comparisons = re.findall(comparison_pattern, text, re.IGNORECASE)
        features['stats_has_comparisons'] = 1 if comparisons else 0
        
        return features
    
    def extract_context_features(self, text: str, window_size: int = 50) -> Dict[str, any]:
        """Extract features based on context windows around key terms"""
        features = {}
        text_lower = text.lower()
        
        # Find leadership mentions and their contexts
        leadership_contexts = []
        for term in self.leadership_terms['positions']:
            for match in re.finditer(r'\b' + term + r'\b', text_lower):
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                context = text[start:end]
                leadership_contexts.append(context)
        
        features['context_leadership_mentions'] = len(leadership_contexts)
        
        # Analyze contexts for frame indicators
        if leadership_contexts:
            context_text = ' '.join(leadership_contexts)
            # Check for frame indicators in leadership contexts
            for frame in self.frame_lexicons:
                frame_count = 0
                for category_terms in self.frame_lexicons[frame].values():
                    for term in category_terms:
                        if term in context_text.lower():
                            frame_count += 1
                features[f'context_{frame}_near_leadership'] = frame_count
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, any]:
        """Extract all features for frame detection"""
        features = {}
        
        # Combine all feature types
        features.update(self.extract_lexical_features(text))
        features.update(self.extract_demographic_features(text))
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_statistical_features(text))
        features.update(self.extract_context_features(text))
        
        # Add text length as a feature
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        return features
    
    def extract_features_for_training(self, segments: List[Dict[str, str]], 
                                    labels: Optional[List[List[str]]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract features from text segments for model training
        
        Args:
            segments: List of text segments with metadata
            labels: Optional list of frame labels for each segment
            
        Returns:
            Feature matrix and optional label matrix
        """
        feature_dicts = []
        
        for segment in segments:
            text = segment.get('text', segment.get('content', ''))
            features = self.extract_all_features(text)
            feature_dicts.append(features)
        
        # Convert to matrix
        from sklearn.feature_extraction import DictVectorizer
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(feature_dicts)
        
        # Process labels if provided
        y = None
        if labels:
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer(classes=['underrepresentation', 'overrepresentation', 
                                              'obstacles', 'successes'])
            y = mlb.fit_transform(labels)
        
        return X, y