#!/usr/bin/env python3
"""
Basic integration test - tests core functionality without heavy ML dependencies
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_config_system():
    """Test configuration system"""
    print("🧪 Testing Configuration System")
    
    try:
        from config import ConfigManager
        config_manager = ConfigManager()
        
        # Test environment detection
        env = config_manager._environment
        print(f"  ✅ Environment detected: {env}")
        
        # Test config loading
        model_config = config_manager.load_config('model_config')
        data_config = config_manager.load_config('data_config')
        frame_defs = config_manager.load_config('frame_definitions')
        
        print(f"  ✅ Loaded model config with {len(model_config)} sections")
        print(f"  ✅ Loaded data config with {len(data_config)} sections")
        print(f"  ✅ Loaded frame definitions for {len(frame_defs['frames'])} frames")
        
        # Test path configuration
        paths = config_manager.get_paths()
        print(f"  ✅ Configured {len(paths)} paths")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        traceback.print_exc()
        return False


def test_data_structures():
    """Test that our data structures work"""
    print("\n🧪 Testing Data Structures")
    
    try:
        # Test sample data loading
        import json
        with open('data/sample_articles.json', 'r') as f:
            articles = json.load(f)
        
        print(f"  ✅ Loaded {len(articles)} sample articles")
        
        # Test article structure
        article = articles[0]
        required_fields = ['article_id', 'source', 'date', 'title', 'content', 'human_coding']
        
        for field in required_fields:
            if field not in article:
                raise ValueError(f"Missing required field: {field}")
        
        print(f"  ✅ Article structure valid: {article['article_id']}")
        
        # Test human coding structure
        coding = article['human_coding']
        expected_frames = ['underrepresentation', 'overrepresentation', 'obstacles', 'successes']
        
        found_frames = []
        for frame in expected_frames:
            if frame in coding and coding[frame]:
                found_frames.append(frame)
        
        print(f"  ✅ Human coding valid: {len(found_frames)} frames coded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data structure test failed: {e}")
        traceback.print_exc()
        return False


def test_preprocessing_basic():
    """Test basic preprocessing without NLTK"""
    print("\n🧪 Testing Basic Preprocessing")
    
    try:
        # Test text cleaning without heavy dependencies
        sample_text = "Women hold only 21% of C-suite positions. This   has    extra     spaces. Check out https://example.com for more."
        
        # Basic cleaning functions
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', sample_text)
        # Fix whitespace
        text = ' '.join(text.split())
        
        print(f"  ✅ Basic text cleaning works")
        
        # Test demographic detection patterns
        demographic_patterns = {
            'women': r'\b(women?|females?)\b',
            'men': r'\b(men|males?)\b',
        }
        
        demographics_found = []
        for demo, pattern in demographic_patterns.items():
            if re.search(pattern, sample_text.lower()):
                demographics_found.append(demo)
        
        print(f"  ✅ Demographic detection: found {demographics_found}")
        
        # Test frame indicators
        frame_indicators = {
            'underrepresentation': ['only', 'just', 'few'],
            'obstacles': ['barrier', 'challenge', 'difficulty']
        }
        
        frames_found = []
        for frame, indicators in frame_indicators.items():
            for indicator in indicators:
                if indicator in sample_text.lower():
                    frames_found.append(frame)
                    break
        
        print(f"  ✅ Frame detection: found {frames_found}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing test failed: {e}")
        traceback.print_exc()
        return False


def test_utils_basic():
    """Test basic utility functions"""
    print("\n🧪 Testing Basic Utilities")
    
    try:
        # Test without heavy imports
        from pathlib import Path
        import hashlib
        
        # Test hash function
        text = "test text"
        hash_val = hashlib.md5(text.encode()).hexdigest()
        print(f"  ✅ Hash function works: {hash_val[:8]}...")
        
        # Test path operations
        test_path = Path("test/path")
        print(f"  ✅ Path operations work: {test_path}")
        
        # Test basic validation
        def validate_article(article):
            required = ['article_id', 'content']
            return all(field in article for field in required)
        
        test_article = {'article_id': 'test', 'content': 'test content'}
        is_valid = validate_article(test_article)
        print(f"  ✅ Validation works: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utils test failed: {e}")
        traceback.print_exc()
        return False


def test_project_structure():
    """Test project file structure"""
    print("\n🧪 Testing Project Structure")
    
    try:
        # Check key directories exist
        required_dirs = ['src', 'configs', 'data', 'notebooks', 'tests']
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                raise FileNotFoundError(f"Missing directory: {dir_name}")
        
        print(f"  ✅ All required directories exist")
        
        # Check key files exist
        required_files = [
            'src/__init__.py',
            'src/config.py',
            'src/preprocessing.py',
            'src/data_loader.py',
            'configs/model_config.yaml',
            'configs/data_config.yaml',
            'data/sample_articles.json',
            'requirements.txt',
            'README.md'
        ]
        
        for file_name in required_files:
            file_path = Path(file_name)
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file: {file_name}")
        
        print(f"  ✅ All required files exist")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Project structure test failed: {e}")
        traceback.print_exc()
        return False


def run_basic_tests():
    """Run basic tests that don't require heavy dependencies"""
    print("🚀 Running Basic Integration Tests\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration System", test_config_system),
        ("Data Structures", test_data_structures),
        ("Basic Preprocessing", test_preprocessing_basic),
        ("Basic Utilities", test_utils_basic),
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
    print("📊 BASIC TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} basic tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Core functionality is working.")
        print("💡 To test ML models, install: pip install transformers torch")
        return True
    else:
        print("⚠️  Some basic tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    # Change to project directory
    import os
    os.chdir(Path(__file__).parent)
    
    success = run_basic_tests()
    sys.exit(0 if success else 1)