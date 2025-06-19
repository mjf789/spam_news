#!/usr/bin/env python3
"""
Integration test script to verify all phases work correctly
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_phase_1_data_pipeline():
    """Test Phase 1: Data Pipeline Setup"""
    print("🧪 Testing Phase 1: Data Pipeline Setup")
    
    try:
        # Test data loading
        from data_loader import ArticleDataLoader
        loader = ArticleDataLoader()
        
        # Load sample articles
        df = loader.load_articles()
        print(f"  ✅ Loaded {len(df)} sample articles")
        
        # Test train/val/test split
        train_df, val_df, test_df = loader.get_train_val_test_split()
        print(f"  ✅ Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        # Test validation
        report = loader.validate_data()
        print(f"  ✅ Data validation: {report['total_articles']} articles, valid: {report['is_valid']}")
        
        # Test human coding extraction
        coding_df = loader.get_coding_data()
        print(f"  ✅ Extracted {len(coding_df)} coding annotations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 1 failed: {e}")
        traceback.print_exc()
        return False


def test_phase_2_modular_code():
    """Test Phase 2: Modular Code Development"""
    print("\n🧪 Testing Phase 2: Modular Code Development")
    
    try:
        # Test configuration system
        from config import ConfigManager
        config_manager = ConfigManager()
        
        model_config = config_manager.load_config('model_config')
        data_config = config_manager.load_config('data_config')
        frame_defs = config_manager.load_config('frame_definitions')
        
        print(f"  ✅ Loaded configurations: model, data, frame definitions")
        
        # Test preprocessing
        from preprocessing import ArticlePreprocessor
        preprocessor = ArticlePreprocessor()
        
        sample_text = "Women hold only 21% of C-suite positions, facing significant barriers to advancement."
        demographics = preprocessor.detect_demographics(sample_text)
        stats = preprocessor.extract_statistics(sample_text)
        frames = preprocessor.identify_frame_candidates(sample_text)
        
        print(f"  ✅ Preprocessing: found {len(demographics)} demographics, {len(stats)} stats, {len(frames)} frame candidates")
        
        # Test segmentation
        from segmentation import ArticleSegmenter
        segmenter = ArticleSegmenter()
        
        sentences = segmenter.segment_by_sentences(sample_text)
        windows = segmenter.create_sliding_windows(sentences)
        
        print(f"  ✅ Segmentation: {len(sentences)} sentences, {len(windows)} windows")
        
        # Test feature extraction
        from feature_extraction import FrameFeatureExtractor
        feature_extractor = FrameFeatureExtractor()
        
        features = feature_extractor.extract_all_features(sample_text)
        print(f"  ✅ Feature extraction: {len(features)} features extracted")
        
        # Test utilities
        from utils import setup_logging, calculate_icc
        import numpy as np
        
        # Test ICC calculation
        ratings1 = np.array([1, 0, 1, 1, 0])
        ratings2 = np.array([1, 0, 1, 0, 0])
        icc = calculate_icc(ratings1, ratings2)
        print(f"  ✅ Utils: ICC calculation = {icc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 2 failed: {e}")
        traceback.print_exc()
        return False


def test_frame_detection_basic():
    """Test basic frame detection without heavy models"""
    print("\n🧪 Testing Frame Detection (Basic)")
    
    try:
        # Test frame detection classes (without loading heavy models)
        from frame_detection import FrameAnalyzer
        
        # Create a mock detector for testing
        class MockDetector:
            def predict(self, text):
                # Simple keyword-based mock
                frames = []
                scores = {}
                
                if 'only' in text.lower() or 'just' in text.lower():
                    frames.append('underrepresentation')
                    scores['underrepresentation'] = 0.8
                else:
                    scores['underrepresentation'] = 0.2
                    
                if 'barrier' in text.lower() or 'challenge' in text.lower():
                    frames.append('obstacles')
                    scores['obstacles'] = 0.9
                else:
                    scores['obstacles'] = 0.1
                
                scores['overrepresentation'] = 0.1
                scores['successes'] = 0.1
                
                return {'frames': frames, 'scores': scores}
        
        # Test analyzer
        analyzer = FrameAnalyzer(MockDetector())
        
        sample_article = {
            'article_id': 'test_001',
            'content': 'Women hold only 21% of leadership positions, facing significant barriers to advancement.'
        }
        
        results = analyzer.analyze_article(sample_article)
        print(f"  ✅ Frame analysis: detected {len(results['frames_detected'])} frames")
        print(f"      Frames: {results['frames_detected']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Frame detection test failed: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline integration"""
    print("\n🧪 Testing Full Pipeline Integration")
    
    try:
        # Load configuration
        from config import get_config_manager
        config_manager = get_config_manager()
        
        # Load sample data
        from data_loader import ArticleDataLoader
        loader = ArticleDataLoader()
        df = loader.load_articles()
        
        # Process first article
        article = df.iloc[0].to_dict()
        
        # Preprocess
        from preprocessing import ArticlePreprocessor
        preprocessor = ArticlePreprocessor()
        processed = preprocessor.preprocess_article(article)
        
        # Segment
        from segmentation import ArticleSegmenter
        segmenter = ArticleSegmenter()
        segments = segmenter.segment_for_analysis(processed)
        
        # Extract features
        from feature_extraction import FrameFeatureExtractor
        feature_extractor = FrameFeatureExtractor()
        features = feature_extractor.extract_all_features(processed['cleaned_content'])
        
        print(f"  ✅ Full pipeline completed for article: {article['article_id']}")
        print(f"      Processed content length: {len(processed['cleaned_content'])} chars")
        print(f"      Segments: {segments['metadata']['total_sentences']} sentences")
        print(f"      Features: {len(features)} extracted")
        print(f"      Demographics found: {list(processed['demographics_found'].keys())}")
        print(f"      Frame candidates: {list(processed['frame_candidates'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Full pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_environment_detection():
    """Test environment detection and path setup"""
    print("\n🧪 Testing Environment Detection")
    
    try:
        from config import get_config_manager
        config_manager = get_config_manager()
        
        print(f"  ✅ Environment detected: {config_manager._environment}")
        
        # Test path configuration
        paths = config_manager.get_paths()
        print(f"  ✅ Configured paths:")
        for key, path in paths.items():
            if key.endswith('_dir'):
                exists = path.exists()
                print(f"      {key}: {path} {'(exists)' if exists else '(will be created)'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Integration Tests for Spam News Analysis Project\n")
    
    tests = [
        ("Environment Detection", test_environment_detection),
        ("Phase 1: Data Pipeline", test_phase_1_data_pipeline),
        ("Phase 2: Modular Code", test_phase_2_modular_code),
        ("Frame Detection Basic", test_frame_detection_basic),
        ("Full Pipeline Integration", test_full_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for the next phase.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    # Change to project directory
    import os
    os.chdir(Path(__file__).parent)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)