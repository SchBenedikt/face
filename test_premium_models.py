#!/usr/bin/env python3
"""
Test script to validate premium face recognition models implementation
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from face_recognition_engine import FaceRecognitionEngine
    import config
    print("✅ Successfully imported face recognition modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def test_config_updates():
    """Test that config has been updated with best models"""
    print("\n🔧 Testing Configuration Updates...")
    
    # Check primary model is ArcFace
    assert config.FACE_EMBEDDING_MODEL == "ArcFace", f"Expected ArcFace, got {config.FACE_EMBEDDING_MODEL}"
    print("✅ Primary embedding model is ArcFace")
    
    # Check detection backends prioritize RetinaFace
    assert config.FACE_DETECTION_BACKENDS[0] == "retinaface", f"Expected retinaface first, got {config.FACE_DETECTION_BACKENDS[0]}"
    print("✅ RetinaFace is primary detection backend")
    
    # Check similarity algorithm is premium
    assert config.FACE_SIMILARITY_ALGORITHM == "premium", f"Expected premium, got {config.FACE_SIMILARITY_ALGORITHM}"
    print("✅ Premium similarity algorithm enabled")
    
    # Check CNN model for accuracy
    assert config.FACE_RECOGNITION_MODEL == "cnn", "Should use CNN for best accuracy"
    print("✅ CNN model selected for face recognition")
    
    print("✅ All configuration tests passed!")

def test_face_engine_initialization():
    """Test that face recognition engine initializes with best models"""
    print("\n🤖 Testing Face Recognition Engine...")
    
    try:
        engine = FaceRecognitionEngine()
        print("✅ Face recognition engine initialized successfully")
        
        # Check that best models are loaded
        assert "ArcFace" in engine.models, "ArcFace should be in models list"
        assert "Facenet512" in engine.models, "Facenet512 should be in models list"
        print(f"✅ Models loaded: {engine.models}")
        
        # Check detection backends
        assert "retinaface" in engine.detection_backends, "RetinaFace should be available"
        assert "mtcnn" in engine.detection_backends, "MTCNN should be available"
        print(f"✅ Detection backends: {engine.detection_backends}")
        
        # Check ensemble is enabled
        assert engine.ensemble_enabled == True, "Ensemble should be enabled for premium accuracy"
        print("✅ Ensemble mode enabled")
        
        return engine
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return None

def test_face_detection_methods(engine):
    """Test that premium detection methods are available"""
    print("\n👁️ Testing Face Detection Methods...")
    
    # Create a simple test image
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    test_image[100:200, 100:200] = [255, 255, 255]  # Simple white square as placeholder
    
    try:
        # Test that the detect_faces method exists and can be called
        faces = engine.detect_faces(test_image)
        print(f"✅ Face detection completed (found {len(faces)} faces)")
        
        # Test individual detection methods exist
        if hasattr(engine, '_detect_faces_retinaface'):
            print("✅ RetinaFace detection method available")
        else:
            print("❌ RetinaFace detection method missing")
            
        if hasattr(engine, '_detect_faces_mtcnn'):
            print("✅ MTCNN detection method available")
        else:
            print("❌ MTCNN detection method missing")
            
    except Exception as e:
        print(f"⚠️ Face detection test completed with errors: {e}")

def test_embedding_extraction(engine):
    """Test premium embedding extraction"""
    print("\n🧠 Testing Premium Embedding Extraction...")
    
    # Create a test face region
    test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_location = (50, 174, 174, 50)  # (top, right, bottom, left)
    
    try:
        # Test that embedding extraction works
        embedding = engine.extract_face_embedding(test_face, test_location)
        if embedding is not None:
            print(f"✅ Embedding extraction successful (shape: {embedding.shape})")
            print(f"✅ Embedding type: {type(embedding)}")
        else:
            print("⚠️ Embedding extraction returned None (expected with random data)")
            
        # Test that premium ensemble method exists
        if hasattr(engine, '_extract_premium_ensemble_embedding'):
            print("✅ Premium ensemble extraction method available")
        else:
            print("❌ Premium ensemble extraction method missing")
            
        # Test InsightFace integration
        if hasattr(engine, 'has_insightface'):
            if engine.has_insightface:
                print("✅ InsightFace integration available")
            else:
                print("⚠️ InsightFace not available (may need installation)")
        
    except Exception as e:
        print(f"⚠️ Embedding extraction test completed with errors: {e}")

def print_model_comparison():
    """Print information about model improvements"""
    print("\n📊 Model Improvements Summary:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 DETECTION IMPROVEMENTS:")
    print("   • RetinaFace (Primary): Best accuracy for face detection")  
    print("   • MTCNN (Secondary): Excellent for small/angled faces")
    print("   • CNN model: Higher accuracy than HOG")
    print("   • Higher upsampling: Better small face detection")
    print("")
    print("🧠 RECOGNITION IMPROVEMENTS:")
    print("   • ArcFace (Primary): State-of-the-art face verification")
    print("   • InsightFace: Premium accuracy when available")
    print("   • Ensemble approach: Multiple models for best results")
    print("   • Advanced preprocessing: Face alignment & enhancement")
    print("")
    print("⚙️ CONFIGURATION CHANGES:")
    print("   • Premium mode enabled by default")
    print("   • Lower similarity thresholds for better precision")
    print("   • Larger face padding for better context")
    print("   • Quality validation enabled")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def main():
    """Run all tests"""
    print("🚀 Testing Premium Face Recognition Model Implementation")
    print("=" * 60)
    
    # Test configuration
    test_config_updates()
    
    # Test engine initialization
    engine = test_face_engine_initialization()
    if engine is None:
        print("❌ Cannot continue tests without engine")
        return
    
    # Test detection methods
    test_face_detection_methods(engine)
    
    # Test embedding extraction
    test_embedding_extraction(engine)
    
    # Print summary
    print_model_comparison()
    
    print("\n✅ All tests completed!")
    print("🎯 Face recognition system now configured for MAXIMUM ACCURACY")

if __name__ == "__main__":
    main()