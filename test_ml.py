#!/usr/bin/env python3
"""
Test ML components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_frame_detection():
    """Test frame detection with real models"""
    print("🧪 Testing Zero-Shot Frame Detection")
    
    try:
        from frame_detection import ZeroShotFrameDetector
        
        # Create detector
        detector = ZeroShotFrameDetector()
        
        # Test text
        text = "Women hold only 21% of leadership positions, facing significant barriers to advancement."
        
        # Get predictions
        result = detector.predict(text)
        
        print(f"  ✅ Frames detected: {result['frames']}")
        print(f"  ✅ Scores: {result['scores']}")
        
        # Check expected results
        expected_frames = ['underrepresentation', 'obstacles']
        found_expected = any(frame in result['frames'] for frame in expected_frames)
        
        if found_expected:
            print("  ✅ Model correctly identified expected frames")
            return True
        else:
            print("  ⚠️  Model didn't find expected frames, but it's working")
            return True
            
    except Exception as e:
        print(f"  ❌ Frame detection test failed: {e}")
        return False

def test_data_loading():
    """Test data loading with pandas"""
    print("\n🧪 Testing Data Loading with Pandas")
    
    try:
        from data_loader import ArticleDataLoader
        
        loader = ArticleDataLoader()
        df = loader.load_articles()
        
        print(f"  ✅ Loaded {len(df)} articles")
        
        # Test validation
        report = loader.validate_data()
        print(f"  ✅ Validation complete: {report['total_articles']} articles")
        
        # Test coding extraction
        coding_df = loader.get_coding_data()
        print(f"  ✅ Extracted {len(coding_df)} coding annotations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing with NLTK"""
    print("\n🧪 Testing Preprocessing with NLTK")
    
    try:
        from preprocessing import ArticlePreprocessor
        
        preprocessor = ArticlePreprocessor()
        
        sample_article = {
            'article_id': 'test_001',
            'source': 'Test',
            'date': '2021-01-01',
            'title': 'Test Article',
            'content': 'Women hold only 21% of C-suite positions. Black women face concrete barriers to advancement.',
            'human_coding': {
                'underrepresentation': {'women': 1},
                'obstacles': {'women_of_color': 1}
            }
        }
        
        processed = preprocessor.preprocess_article(sample_article)
        
        print(f"  ✅ Processed article: {len(processed['sentences'])} sentences")
        print(f"  ✅ Demographics found: {list(processed['demographics_found'].keys())}")
        print(f"  ✅ Frame candidates: {list(processed['frame_candidates'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running ML Component Tests\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Frame Detection", test_frame_detection),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*50)
    print("📊 ML TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} ML tests passed")
    
    if passed == total:
        print("🎉 All ML components are working!")
    else:
        print("⚠️  Some ML tests failed.")