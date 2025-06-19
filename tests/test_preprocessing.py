"""
Tests for preprocessing module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import ArticlePreprocessor


class TestArticlePreprocessor:
    """Test cases for ArticlePreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return ArticlePreprocessor()
    
    @pytest.fixture
    def sample_article(self):
        """Sample article for testing"""
        return {
            'article_id': 'test_001',
            'source': 'Test News',
            'date': '2021-01-01',
            'title': 'Women in Leadership',
            'content': 'Despite making up nearly half of the workforce, women hold only 21% of C-suite positions. This underrepresentation highlights barriers that women face.',
            'human_coding': {
                'underrepresentation': {'women': 1},
                'obstacles': {'women': 1}
            }
        }
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning"""
        # Test URL removal
        text = "Check out https://example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "https://example.com" not in cleaned
        
        # Test whitespace normalization
        text = "This   has    extra     spaces"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "This has extra spaces"
        
        # Test quote normalization
        text = "She said "hello" to the 'world'"
        cleaned = preprocessor.clean_text(text)
        assert '"' in cleaned
        assert "'" in cleaned
    
    def test_extract_sentences(self, preprocessor):
        """Test sentence extraction"""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = preprocessor.extract_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
    
    def test_detect_demographics(self, preprocessor):
        """Test demographic detection"""
        # Test basic demographics
        text = "Women and men work together"
        demographics = preprocessor.detect_demographics(text)
        assert 'women' in demographics
        assert 'men' in demographics
        
        # Test intersectional identities
        text = "Black women face unique challenges compared to white men"
        demographics = preprocessor.detect_demographics(text)
        assert 'women_of_color' in demographics
        assert 'white_men' in demographics
        
        # Test race detection
        text = "Hispanic and Asian leaders are increasing"
        demographics = preprocessor.detect_demographics(text)
        assert 'hispanic' in demographics
        assert 'asian' in demographics
    
    def test_extract_statistics(self, preprocessor):
        """Test statistical extraction"""
        text = "Women hold only 21% of leadership positions, which is 3 times less than men."
        stats = preprocessor.extract_statistics(text)
        
        # Check percentage extraction
        percentages = [s for s in stats if s['type'] == 'percentage']
        assert len(percentages) > 0
        assert any(s['value'] == '21' for s in percentages)
        
        # Check comparison extraction
        comparisons = [s for s in stats if s['type'] == 'comparison']
        assert len(comparisons) > 0
        assert any(s['value'] == '3' for s in comparisons)
    
    def test_identify_frame_candidates(self, preprocessor):
        """Test frame candidate identification"""
        text = "Women are underrepresented in leadership, facing barriers to advancement, but some have achieved breakthrough successes."
        
        candidates = preprocessor.identify_frame_candidates(text)
        
        assert 'underrepresentation' in candidates
        assert 'obstacles' in candidates
        assert 'successes' in candidates
        
        # Check that contexts are extracted
        assert any('underrepresented' in c for c in candidates['underrepresentation'])
        assert any('barriers' in c for c in candidates['obstacles'])
        assert any('breakthrough' in c for c in candidates['successes'])
    
    def test_preprocess_article(self, preprocessor, sample_article):
        """Test full preprocessing pipeline"""
        processed = preprocessor.preprocess_article(sample_article)
        
        # Check all expected fields
        assert processed['article_id'] == sample_article['article_id']
        assert 'cleaned_content' in processed
        assert 'sentences' in processed
        assert 'demographics_found' in processed
        assert 'statistics' in processed
        assert 'frame_candidates' in processed
        
        # Check that demographics were found
        assert 'women' in processed['demographics_found']
        
        # Check that frame candidates were identified
        assert 'underrepresentation' in processed['frame_candidates']
        assert 'obstacles' in processed['frame_candidates']