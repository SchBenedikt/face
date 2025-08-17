"""
Test script to validate face recognition improvements
"""

import numpy as np
from face_recognition_engine import FaceRecognitionEngine
from vector_store import FaceVectorStore
from face_utils import FaceSimilarityCalculator, validate_face_embedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_quality():
    """Test embedding extraction and quality validation"""
    logger.info("Testing embedding quality improvements...")
    
    engine = FaceRecognitionEngine()
    
    # Create test embeddings
    test_embedding_1 = np.random.random(512).astype(np.float32)
    test_embedding_2 = test_embedding_1 + np.random.random(512) * 0.1  # Similar embedding
    test_embedding_3 = np.random.random(512).astype(np.float32)  # Different embedding
    
    # Test normalization
    normalized_1 = engine._advanced_normalize_embedding(test_embedding_1)
    normalized_2 = engine._advanced_normalize_embedding(test_embedding_2)
    normalized_3 = engine._advanced_normalize_embedding(test_embedding_3)
    
    logger.info(f"Original embedding 1 norm: {np.linalg.norm(test_embedding_1):.4f}")
    logger.info(f"Normalized embedding 1 norm: {np.linalg.norm(normalized_1):.4f}")
    
    # Test similarity calculation
    similarity_12, metrics_12 = engine.compare_faces(normalized_1, normalized_2)
    similarity_13, metrics_13 = engine.compare_faces(normalized_1, normalized_3)
    
    logger.info(f"Similarity between similar embeddings: {similarity_12:.4f}")
    logger.info(f"Similarity between different embeddings: {similarity_13:.4f}")
    logger.info(f"Similar embeddings metrics: {metrics_12}")
    
    return similarity_12 > similarity_13

def test_advanced_similarity():
    """Test advanced similarity calculation"""
    logger.info("Testing advanced similarity calculation...")
    
    calculator = FaceSimilarityCalculator()
    
    # Create test embeddings
    identical_emb = np.random.random(512).astype(np.float32)
    similar_emb = identical_emb + np.random.random(512) * 0.05
    different_emb = np.random.random(512).astype(np.float32)
    
    # Test different algorithms
    basic_identical = calculator.calculate_comprehensive_similarity(
        identical_emb, identical_emb, "basic"
    )
    enhanced_identical = calculator.calculate_comprehensive_similarity(
        identical_emb, identical_emb, "enhanced"
    )
    
    basic_similar = calculator.calculate_comprehensive_similarity(
        identical_emb, similar_emb, "basic"
    )
    enhanced_similar = calculator.calculate_comprehensive_similarity(
        identical_emb, similar_emb, "enhanced"
    )
    
    basic_different = calculator.calculate_comprehensive_similarity(
        identical_emb, different_emb, "basic"
    )
    
    logger.info(f"Identical embeddings - Basic: {basic_identical['final_similarity']:.4f}")
    logger.info(f"Identical embeddings - Enhanced: {enhanced_identical['final_similarity']:.4f}")
    logger.info(f"Similar embeddings - Basic: {basic_similar['final_similarity']:.4f}")
    logger.info(f"Similar embeddings - Enhanced: {enhanced_similar['final_similarity']:.4f}")
    logger.info(f"Different embeddings - Basic: {basic_different['final_similarity']:.4f}")
    
    return (enhanced_identical['final_similarity'] > enhanced_similar['final_similarity'] and
            enhanced_similar['final_similarity'] > basic_different['final_similarity'])

def test_embedding_validation():
    """Test embedding validation"""
    logger.info("Testing embedding validation...")
    
    # Valid embedding
    valid_emb = np.random.random(512).astype(np.float32)
    is_valid, reason = validate_face_embedding(valid_emb)
    logger.info(f"Valid embedding test: {is_valid}, reason: {reason}")
    
    # Invalid embeddings
    nan_emb = np.full(512, np.nan)
    is_valid_nan, reason_nan = validate_face_embedding(nan_emb)
    logger.info(f"NaN embedding test: {is_valid_nan}, reason: {reason_nan}")
    
    zero_emb = np.zeros(512)
    is_valid_zero, reason_zero = validate_face_embedding(zero_emb)
    logger.info(f"Zero embedding test: {is_valid_zero}, reason: {reason_zero}")
    
    return is_valid and not is_valid_nan and not is_valid_zero

def test_vector_store_improvements():
    """Test vector store similarity improvements"""
    logger.info("Testing vector store improvements...")
    
    try:
        vector_store = FaceVectorStore()
        
        # Create test embeddings
        query_embedding = np.random.random(512).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Test the enhanced similarity calculation method
        stored_embedding = query_embedding + np.random.random(512) * 0.1
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
        
        similarity_metrics = vector_store._calculate_enhanced_similarity(query_embedding, stored_embedding)
        
        logger.info(f"Enhanced similarity metrics: {similarity_metrics}")
        
        confidence = vector_store._calculate_confidence_score(similarity_metrics)
        logger.info(f"Confidence score: {confidence:.4f}")
        
        return similarity_metrics['primary_similarity'] > 0.0
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        return False

def run_all_tests():
    """Run all improvement tests"""
    logger.info("🚀 Running Face Recognition Improvement Tests...")
    
    tests = [
        ("Embedding Quality", test_embedding_quality),
        ("Advanced Similarity", test_advanced_similarity),
        ("Embedding Validation", test_embedding_validation),
        ("Vector Store Improvements", test_vector_store_improvements)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Testing {test_name} ---")
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.warning(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    logger.info(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All face recognition improvements working correctly!")
        return True
    else:
        logger.warning("⚠️ Some tests failed - check the logs above")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)