"""
Text segmentation utilities for article analysis
"""

import re
from typing import List, Tuple, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)


class ArticleSegmenter:
    """Segments articles into analyzable chunks"""
    
    def __init__(self, window_size: int = 3, overlap: int = 1):
        """
        Initialize segmenter
        
        Args:
            window_size: Number of sentences per window
            overlap: Number of overlapping sentences between windows
        """
        self.window_size = window_size
        self.overlap = overlap
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def segment_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = sent_tokenize(text)
        # Filter out very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def segment_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Multiple newline patterns
        paragraph_patterns = [
            r'\n\s*\n',  # Double newline
            r'\r\n\s*\r\n',  # Windows style
            r'\n{2,}',  # Multiple newlines
        ]
        
        # Try each pattern
        paragraphs = [text]
        for pattern in paragraph_patterns:
            new_paragraphs = []
            for para in paragraphs:
                splits = re.split(pattern, para)
                new_paragraphs.extend(splits)
            paragraphs = new_paragraphs
        
        # Clean and filter
        cleaned = []
        for p in paragraphs:
            p = p.strip()
            if len(p) > 20:  # Minimum paragraph length
                cleaned.append(p)
        
        return cleaned
    
    def create_sliding_windows(self, sentences: List[str]) -> List[Dict[str, any]]:
        """
        Create overlapping windows of sentences
        
        Returns:
            List of windows with metadata
        """
        if len(sentences) <= self.window_size:
            return [{
                'text': ' '.join(sentences),
                'sentences': sentences,
                'start_idx': 0,
                'end_idx': len(sentences) - 1,
                'window_id': 0
            }]
        
        windows = []
        step = self.window_size - self.overlap
        
        for i in range(0, len(sentences) - self.window_size + 1, step):
            window_sentences = sentences[i:i + self.window_size]
            windows.append({
                'text': ' '.join(window_sentences),
                'sentences': window_sentences,
                'start_idx': i,
                'end_idx': i + self.window_size - 1,
                'window_id': len(windows)
            })
        
        # Add final window if needed
        if windows and windows[-1]['end_idx'] < len(sentences) - 1:
            remaining = sentences[windows[-1]['end_idx'] + 1 - self.overlap:]
            if remaining:
                windows.append({
                    'text': ' '.join(remaining),
                    'sentences': remaining,
                    'start_idx': windows[-1]['end_idx'] + 1 - self.overlap,
                    'end_idx': len(sentences) - 1,
                    'window_id': len(windows)
                })
        
        return windows
    
    def segment_around_keywords(self, text: str, keywords: List[str], 
                                context_size: int = 100) -> List[Dict[str, any]]:
        """
        Extract text segments around specific keywords
        
        Args:
            text: Full text to segment
            keywords: Keywords to find context around
            context_size: Characters before/after keyword
        
        Returns:
            List of context segments with metadata
        """
        segments = []
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Find all occurrences
            for match in re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                start = max(0, match.start() - context_size)
                end = min(len(text), match.end() + context_size)
                
                # Extend to sentence boundaries
                # Find previous sentence end
                prev_period = text.rfind('.', start, match.start())
                if prev_period != -1:
                    start = prev_period + 1
                
                # Find next sentence end
                next_period = text.find('.', match.end(), end)
                if next_period != -1:
                    end = next_period + 1
                
                segment = text[start:end].strip()
                
                segments.append({
                    'text': segment,
                    'keyword': keyword,
                    'keyword_position': match.start() - start,
                    'original_position': (start, end),
                    'segment_id': len(segments)
                })
        
        return segments
    
    def extract_quotes(self, text: str) -> List[Dict[str, str]]:
        """Extract quoted text from articles"""
        quotes = []
        
        # Pattern for quotes with attribution
        quote_patterns = [
            # "Quote," said Person
            r'"([^"]+)"\s*,?\s*(?:said|says|according to)\s+([^,.]+)',
            # Person said, "Quote"
            r'([^,]+?)\s+(?:said|says|stated)\s*,?\s*"([^"]+)"',
            # Simple quotes
            r'"([^"]{20,})"'  # At least 20 chars to avoid short phrases
        ]
        
        for pattern in quote_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if len(match.groups()) == 2:
                    # Quote with attribution
                    if 'said' in pattern:
                        quote_text = match.group(1)
                        attribution = match.group(2)
                    else:
                        attribution = match.group(1)
                        quote_text = match.group(2)
                else:
                    # Just the quote
                    quote_text = match.group(1)
                    attribution = None
                
                quotes.append({
                    'text': quote_text.strip(),
                    'attribution': attribution.strip() if attribution else None,
                    'full_match': match.group(0)
                })
        
        return quotes
    
    def segment_for_analysis(self, article: Dict) -> Dict[str, List[Dict]]:
        """
        Comprehensive segmentation for frame analysis
        
        Args:
            article: Article dict with 'content' field
            
        Returns:
            Dict with different segmentation types
        """
        text = article.get('cleaned_content', article['content'])
        
        # Get sentences and paragraphs
        sentences = self.segment_by_sentences(text)
        paragraphs = self.segment_by_paragraphs(text)
        
        # Create sliding windows
        windows = self.create_sliding_windows(sentences)
        
        # Extract quotes
        quotes = self.extract_quotes(text)
        
        # Segment around leadership terms
        leadership_keywords = [
            'CEO', 'executive', 'director', 'president',
            'leadership', 'board', 'management'
        ]
        leadership_segments = self.segment_around_keywords(
            text, leadership_keywords, context_size=150
        )
        
        return {
            'sentences': sentences,
            'paragraphs': paragraphs,
            'windows': windows,
            'quotes': quotes,
            'leadership_segments': leadership_segments,
            'metadata': {
                'total_sentences': len(sentences),
                'total_paragraphs': len(paragraphs),
                'total_windows': len(windows),
                'total_quotes': len(quotes),
                'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            }
        }