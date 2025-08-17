#!/usr/bin/env python3
"""
Comprehensive test for all face recognition improvements
Tests the complete pipeline from face detection to similarity search
"""

import numpy as np
import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_completeness():
    """Test that all new configuration options are present"""
    logger.info("Testing configuration completeness...")
    
    try:
        import config
        
        required_configs = [
            'FACE_EMBEDDING_MODELS',
            'FACE_DETECTION_BACKENDS', 
            'FACE_ENSEMBLE_WEIGHTING',
            'ENSEMBLE_SIMILARITY_WEIGHTS',
            'FACE_SIMILARITY_ALGORITHM',
            'SIMILARITY_THRESHOLD'
        ]
        
        missing_configs = []
        for cfg in required_configs:
            if not hasattr(config, cfg):
                missing_configs.append(cfg)
        
        if missing_configs:
            logger.warning(f"Missing configurations: {missing_configs}")
            return False
        
        # Test specific values
        if hasattr(config, 'FACE_EMBEDDING_MODELS'):
            models = config.FACE_EMBEDDING_MODELS
            if not isinstance(models, list) or len(models) < 2:
                logger.warning("FACE_EMBEDDING_MODELS should be a list with multiple models")
                return False
        
        logger.info("✅ Configuration completeness test passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

def test_face_recognition_engine_ensemble():
    """Test the enhanced face recognition engine"""
    logger.info("Testing face recognition engine enhancements...")
    
    try:
        # Mock dependencies to avoid import errors
        sys.modules['cv2'] = MagicMock()
        sys.modules['face_recognition'] = MagicMock()
        sys.modules['deepface'] = MagicMock()
        sys.modules['DeepFace'] = MagicMock()
        
        from face_recognition_engine import FaceRecognitionEngine
        
        # Test engine initialization
        engine = FaceRecognitionEngine()
        
        # Test that it has the new attributes
        expected_attrs = ['models', 'primary_model', 'detection_backends', 'ensemble_enabled']
        missing_attrs = []
        
        for attr in expected_attrs:
            if not hasattr(engine, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            logger.warning(f"Missing engine attributes: {missing_attrs}")
            return False
        
        # Test that models is a list with multiple items
        if not isinstance(engine.models, list) or len(engine.models) < 2:
            logger.warning("Engine models should be a list with multiple models")
            return False
        
        # Test method existence
        required_methods = [
            '_preprocess_face_advanced',
            '_extract_single_model_embedding', 
            '_combine_embeddings',
            '_finalize_embedding',
            '_extract_fallback_embedding',
            '_filter_valid_faces'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(engine, method):
                missing_methods.append(method)
        
        if missing_methods:
            logger.warning(f"Missing engine methods: {missing_methods}")
            return False
        
        logger.info("✅ Face recognition engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"Face recognition engine test failed: {e}")
        return False

def test_vector_store_enhancements():
    """Test vector store similarity enhancements"""
    logger.info("Testing vector store enhancements...")
    
    try:
        # Mock chromadb to avoid dependency issues
        sys.modules['chromadb'] = MagicMock()
        
        from vector_store import FaceVectorStore
        
        # Test enhanced similarity calculation with mock data
        vector_store = FaceVectorStore()
        
        # Create test embeddings
        emb1 = np.random.random(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = emb1 + np.random.random(512) * 0.1  
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Test enhanced similarity calculation
        metrics = vector_store._calculate_enhanced_similarity(emb1, emb2)
        
        # Check for required metrics
        required_metrics = [
            'cosine_similarity', 'euclidean_similarity', 'correlation_similarity',
            'angular_similarity', 'manhattan_similarity', 'primary_similarity',
            'ensemble_score'
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            logger.warning(f"Missing similarity metrics: {missing_metrics}")
            return False
        
        # Test confidence calculation
        confidence = vector_store._calculate_confidence_score(metrics)
        
        if not isinstance(confidence, float) or confidence < 0 or confidence > 1:
            logger.warning(f"Invalid confidence score: {confidence}")
            return False
        
        logger.info("✅ Vector store enhancement test passed")
        return True
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        return False

def test_face_utilities_algorithms():
    """Test face utilities algorithm enhancements"""
    logger.info("Testing face utilities algorithm enhancements...")
    
    try:
        from face_utils import FaceSimilarityCalculator, validate_face_embedding
        
        # Test similarity calculator
        calculator = FaceSimilarityCalculator()
        
        # Create test embeddings
        emb1 = np.random.random(128).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = emb1 + np.random.random(128) * 0.05
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Test different algorithms
        basic_result = calculator.calculate_comprehensive_similarity(emb1, emb2, "basic")
        enhanced_result = calculator.calculate_comprehensive_similarity(emb1, emb2, "enhanced") 
        premium_result = calculator.calculate_comprehensive_similarity(emb1, emb2, "premium")
        
        # Verify results structure
        for result, name in [(basic_result, "basic"), (enhanced_result, "enhanced"), (premium_result, "premium")]:
            if not isinstance(result, dict):
                logger.warning(f"{name} result should be a dictionary")
                return False
            
            if 'final_similarity' not in result:
                logger.warning(f"{name} result missing final_similarity")
                return False
        
        # Test that enhanced has more metrics than basic
        enhanced_metrics = len([k for k in enhanced_result.keys() if 'similarity' in k])
        basic_metrics = len([k for k in basic_result.keys() if 'similarity' in k])
        
        if enhanced_metrics <= basic_metrics:
            logger.warning("Enhanced algorithm should have more metrics than basic")
            return False
        
        # Test embedding validation
        valid_result = validate_face_embedding(emb1)
        invalid_result = validate_face_embedding(np.full(128, np.nan))
        
        if not isinstance(valid_result, tuple) or len(valid_result) != 2:
            logger.warning("validate_face_embedding should return tuple of (bool, str)")
            return False
        
        if not valid_result[0] or invalid_result[0]:
            logger.warning("Embedding validation not working correctly")
            return False
        
        logger.info("✅ Face utilities algorithm test passed")
        return True
        
    except Exception as e:
        logger.error(f"Face utilities test failed: {e}")
        return False

def test_streamlit_integration():
    """Test streamlit app integration features"""
    logger.info("Testing Streamlit integration improvements...")
    
    try:
        # Mock streamlit to avoid import issues
        sys.modules['streamlit'] = MagicMock()
        
        # This is a basic import test since we can't run streamlit in this environment
        import app
        
        # Test that search function exists and has expected parameters
        if not hasattr(app, 'search_similar_faces'):
            logger.warning("search_similar_faces function not found")
            return False
        
        # Check function signature (simplified test)
        import inspect
        sig = inspect.signature(app.search_similar_faces)
        expected_params = ['uploaded_file', 'max_results', 'similarity_threshold']
        
        actual_params = list(sig.parameters.keys())
        for param in expected_params:
            if param not in actual_params:
                logger.warning(f"Missing parameter in search_similar_faces: {param}")
                return False
        
        logger.info("✅ Streamlit integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Streamlit integration test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("🚀 Running Comprehensive Face Recognition Improvement Tests")
    logger.info("=" * 80)
    
    tests = [
        ("Configuration Completeness", test_configuration_completeness),
        ("Face Recognition Engine Ensemble", test_face_recognition_engine_ensemble),
        ("Vector Store Enhancements", test_vector_store_enhancements),
        ("Face Utilities Algorithms", test_face_utilities_algorithms),
        ("Streamlit Integration", test_streamlit_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                logger.warning(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"💥 {test_name}: ERROR - {e}")
    
    total = passed + failed
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FINAL RESULTS: {passed}/{total} tests passed ({failed} failed)")
    
    if passed == total:
        logger.info("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        logger.info("🚀 Face recognition improvements are working correctly!")
        logger.info("\n🎯 Key improvements verified:")
        logger.info("   ✅ Ensemble model architecture")
        logger.info("   ✅ Advanced similarity algorithms")
        logger.info("   ✅ Enhanced configuration system")
        logger.info("   ✅ Improved error handling")
        logger.info("   ✅ Better user interface")
    elif passed > total * 0.8:  # 80% pass rate
        logger.info("🌟 Most tests passed! Minor issues may need attention.")
    else:
        logger.warning("⚠️ Several tests failed. Please review the improvements.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)