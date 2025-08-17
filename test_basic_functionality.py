#!/usr/bin/env python3
"""
Basic functionality test for improved face recognition system
This test can run without external dependencies to verify core improvements
"""

import numpy as np
import sys
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_similarity_algorithms():
    """Test the improved similarity calculation algorithms"""
    try:
        from face_utils import FaceSimilarityCalculator
        
        calculator = FaceSimilarityCalculator()
        
        # Create test embeddings
        identical_emb = np.random.random(512).astype(np.float32)
        identical_emb = identical_emb / np.linalg.norm(identical_emb)  # Normalize
        
        similar_emb = identical_emb + np.random.random(512) * 0.05
        similar_emb = similar_emb / np.linalg.norm(similar_emb)
        
        different_emb = np.random.random(512).astype(np.float32)
        different_emb = different_emb / np.linalg.norm(different_emb)
        
        # Test different algorithms
        basic_result = calculator.calculate_comprehensive_similarity(
            identical_emb, similar_emb, "basic"
        )
        
        enhanced_result = calculator.calculate_comprehensive_similarity(
            identical_emb, similar_emb, "enhanced"
        )
        
        premium_result = calculator.calculate_comprehensive_similarity(
            identical_emb, similar_emb, "premium"
        )
        
        logger.info("Similarity Algorithm Tests:")
        logger.info(f"Basic similarity: {basic_result.get('final_similarity', 0.0):.4f}")
        logger.info(f"Enhanced similarity: {enhanced_result.get('final_similarity', 0.0):.4f}")
        logger.info(f"Premium similarity: {premium_result.get('final_similarity', 0.0):.4f}")
        
        # Test that enhanced is more sophisticated than basic
        enhanced_metrics_count = len([k for k in enhanced_result.keys() if 'similarity' in k])
        basic_metrics_count = len([k for k in basic_result.keys() if 'similarity' in k])
        
        if enhanced_metrics_count > basic_metrics_count:
            logger.info("✅ Enhanced algorithm uses more metrics than basic")
            return True
        else:
            logger.warning("❌ Enhanced algorithm not properly implemented")
            return False
            
    except Exception as e:
        logger.error(f"❌ Similarity algorithm test failed: {e}")
        return False

def test_embedding_optimization():
    """Test the embedding optimization functionality"""
    try:
        from face_utils import FaceEmbeddingOptimizer
        
        optimizer = FaceEmbeddingOptimizer()
        
        # Create test embedding with some issues
        test_embedding = np.random.random(512).astype(np.float32)
        test_embedding[100] = np.nan  # Add NaN value
        test_embedding[200] = np.inf  # Add inf value
        test_embedding *= 10  # Make it unnormalized
        
        # Optimize embedding
        optimized = optimizer.optimize_embedding(test_embedding)
        
        # Check that optimization worked
        has_nan = np.any(np.isnan(optimized))
        has_inf = np.any(np.isinf(optimized))
        is_normalized = abs(np.linalg.norm(optimized) - 1.0) < 0.01  # Should be unit vector
        
        logger.info("Embedding Optimization Tests:")
        logger.info(f"Removed NaN values: {not has_nan}")
        logger.info(f"Removed Inf values: {not has_inf}")
        logger.info(f"Properly normalized: {is_normalized}")
        
        if not has_nan and not has_inf and is_normalized:
            logger.info("✅ Embedding optimization working correctly")
            return True
        else:
            logger.warning("❌ Embedding optimization has issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding optimization test failed: {e}")
        return False

def test_config_enhancements():
    """Test that configuration enhancements are loaded"""
    try:
        import config
        
        # Check for new configuration options
        has_ensemble_models = hasattr(config, 'FACE_EMBEDDING_MODELS')
        has_detection_backends = hasattr(config, 'FACE_DETECTION_BACKENDS')
        has_ensemble_weighting = hasattr(config, 'FACE_ENSEMBLE_WEIGHTING')
        has_similarity_weights = hasattr(config, 'ENSEMBLE_SIMILARITY_WEIGHTS')
        
        logger.info("Configuration Enhancement Tests:")
        logger.info(f"Has ensemble models config: {has_ensemble_models}")
        logger.info(f"Has detection backends config: {has_detection_backends}")
        logger.info(f"Has ensemble weighting config: {has_ensemble_weighting}")
        logger.info(f"Has similarity weights config: {has_similarity_weights}")
        
        if has_ensemble_models and has_detection_backends:
            logger.info("✅ Configuration properly enhanced")
            return True
        else:
            logger.warning("❌ Configuration enhancements missing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_vector_store_enhancements():
    """Test vector store similarity enhancements"""
    try:
        from vector_store import FaceVectorStore
        
        # Create a test instance (won't actually connect to DB)
        vector_store = FaceVectorStore()
        
        # Test enhanced similarity calculation
        emb1 = np.random.random(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = emb1 + np.random.random(512) * 0.1
        emb2 = emb2 / np.linalg.norm(emb2)
        
        metrics = vector_store._calculate_enhanced_similarity(emb1, emb2)
        
        # Check that new metrics are present
        expected_metrics = [
            'cosine_similarity', 'euclidean_similarity', 'correlation_similarity',
            'angular_similarity', 'manhattan_similarity', 'primary_similarity',
            'ensemble_score'
        ]
        
        has_all_metrics = all(metric in metrics for metric in expected_metrics)
        
        logger.info("Vector Store Enhancement Tests:")
        logger.info(f"Has all enhanced metrics: {has_all_metrics}")
        logger.info(f"Primary similarity: {metrics.get('primary_similarity', 0.0):.4f}")
        logger.info(f"Ensemble score: {metrics.get('ensemble_score', 0.0):.4f}")
        
        if has_all_metrics and metrics.get('primary_similarity', 0.0) > 0:
            logger.info("✅ Vector store enhancements working")
            return True
        else:
            logger.warning("❌ Vector store enhancements have issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector store enhancement test failed: {e}")
        return False

def run_all_tests():
    """Run all basic functionality tests"""
    logger.info("🚀 Running Basic Functionality Tests for Face Recognition Improvements")
    logger.info("=" * 70)
    
    tests = [
        ("Similarity Algorithms", test_similarity_algorithms),
        ("Embedding Optimization", test_embedding_optimization), 
        ("Configuration Enhancements", test_config_enhancements),
        ("Vector Store Enhancements", test_vector_store_enhancements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic functionality tests passed!")
        logger.info("The face recognition improvements are working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
        logger.info("Some improvements may need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)