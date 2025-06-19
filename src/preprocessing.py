"""
Text preprocessing utilities for article analysis
"""

import re
import string
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class ArticlePreprocessor:
    """Preprocesses news articles for frame detection"""
    
    def __init__(self):
        self.demographic_patterns = {
            'women': r'\b(women?|females?|girls?)\b',
            'men': r'\b(men|males?|boys?)\b',
            'white': r'\b(white|caucasian)\b',
            'black': r'\b(black|african[- ]american)\b',
            'hispanic': r'\b(hispanic|latinx?|latina|latino)\b',
            'asian': r'\b(asian|asian[- ]american)\b',
            'poc': r'\b(people of color|minority|minorities)\b'
        }
        
        self.leadership_terms = [
            'ceo', 'executive', 'director', 'manager', 'leader',
            'president', 'vp', 'vice president', 'board', 'c-suite',
            'leadership', 'management', 'supervisor', 'chief',
            'head', 'chair', 'partner', 'principal'
        ]
        
        self.frame_indicators = {
            'underrepresentation': [
                'underrepresented', 'lower rates', 'less than',
                'only', 'just', 'few', 'lacking', 'scarce',
                'minority', 'small percentage', 'rarely'
            ],
            'overrepresentation': [
                'overrepresented', 'dominate', 'majority',
                'most', 'predominantly', 'disproportionately',
                'overwhelmingly', 'comprise', 'hold most'
            ],
            'obstacles': [
                'barrier', 'ceiling', 'discrimination', 'harder',
                'challenges', 'difficulty', 'struggle', 'bias',
                'stereotypes', 'prejudice', 'hurdles', 'impediments'
            ],
            'successes': [
                'first', 'breakthrough', 'achievement', 'milestone',
                'appointed', 'promoted', 'advanced', 'succeeded',
                'accomplished', 'pioneering', 'historic', 'landmark'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or common paragraph patterns
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 20]
    
    def find_leadership_context(self, text: str) -> List[Tuple[int, int, str]]:
        """Find mentions of leadership positions"""
        text_lower = text.lower()
        contexts = []
        
        for term in self.leadership_terms:
            pattern = r'\b' + term + r'\b'
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                contexts.append((start, end, context))
        
        return contexts
    
    def detect_demographics(self, text: str) -> Dict[str, List[str]]:
        """Detect demographic mentions in text"""
        text_lower = text.lower()
        found = {}
        
        for demo, pattern in self.demographic_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                found[demo] = matches
        
        # Detect intersectional identities
        if 'women' in found and ('black' in found or 'hispanic' in found or 
                                 'asian' in found or 'poc' in found):
            found['women_of_color'] = ['women of color']
            
        if 'men' in found and 'white' in found:
            found['white_men'] = ['white men']
            
        if 'women' in found and 'white' in found:
            found['white_women'] = ['white women']
            
        if 'men' in found and ('black' in found or 'hispanic' in found or 
                               'asian' in found or 'poc' in found):
            found['men_of_color'] = ['men of color']
        
        return found
    
    def extract_statistics(self, text: str) -> List[Dict[str, str]]:
        """Extract statistical mentions (percentages, numbers)"""
        stats = []
        
        # Find percentages
        percent_pattern = r'(\d+(?:\.\d+)?)\s*(?:percent|%)'
        for match in re.finditer(percent_pattern, text, re.IGNORECASE):
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            stats.append({
                'type': 'percentage',
                'value': match.group(1),
                'context': text[context_start:context_end]
            })
        
        # Find comparisons (X times more/less)
        comparison_pattern = r'(\d+(?:\.\d+)?)\s*times\s*(more|less|higher|lower)'
        for match in re.finditer(comparison_pattern, text, re.IGNORECASE):
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            stats.append({
                'type': 'comparison',
                'value': match.group(1),
                'direction': match.group(2),
                'context': text[context_start:context_end]
            })
        
        return stats
    
    def identify_frame_candidates(self, text: str) -> Dict[str, List[str]]:
        """Find potential frame indicators in text"""
        text_lower = text.lower()
        candidates = {}
        
        for frame, indicators in self.frame_indicators.items():
            found = []
            for indicator in indicators:
                if indicator in text_lower:
                    # Find context around indicator
                    pattern = r'.{0,50}\b' + re.escape(indicator) + r'\b.{0,50}'
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    found.extend(matches)
            
            if found:
                candidates[frame] = found
        
        return candidates
    
    def preprocess_article(self, article: Dict) -> Dict:
        """Full preprocessing pipeline for an article"""
        text = article['content']
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Extract structure
        sentences = self.extract_sentences(cleaned)
        paragraphs = self.extract_paragraphs(cleaned)
        
        # Find relevant contexts
        leadership_contexts = self.find_leadership_context(cleaned)
        demographics = self.detect_demographics(cleaned)
        statistics = self.extract_statistics(cleaned)
        frame_candidates = self.identify_frame_candidates(cleaned)
        
        # Create preprocessed article
        preprocessed = {
            'article_id': article['article_id'],
            'source': article['source'],
            'date': article['date'],
            'title': article['title'],
            'original_content': article['content'],
            'cleaned_content': cleaned,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'leadership_contexts': leadership_contexts,
            'demographics_found': demographics,
            'statistics': statistics,
            'frame_candidates': frame_candidates,
            'word_count': len(cleaned.split()),
            'sentence_count': len(sentences)
        }
        
        # Include human coding if available
        if 'human_coding' in article:
            preprocessed['human_coding'] = article['human_coding']
        
        return preprocessed
    
    def create_training_examples(self, article: Dict) -> List[Dict]:
        """Create training examples from preprocessed article"""
        examples = []
        
        # For each paragraph that contains leadership terms
        for para in article['paragraphs']:
            para_lower = para.lower()
            
            # Check if paragraph is relevant (contains leadership terms)
            if any(term in para_lower for term in self.leadership_terms):
                # Detect demographics in this paragraph
                demos = self.detect_demographics(para)
                
                # Create example
                example = {
                    'text': para,
                    'article_id': article['article_id'],
                    'demographics': list(demos.keys()) if demos else [],
                }
                
                # Add labels from human coding if available
                if 'human_coding' in article:
                    labels = []
                    for frame_type, demo_counts in article['human_coding'].items():
                        if demo_counts:  # If this frame was coded
                            labels.append(frame_type)
                    example['labels'] = labels
                
                examples.append(example)
        
        return examples