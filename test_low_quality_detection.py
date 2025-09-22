#!/usr/bin/env python3
"""
Test script to validate improvements in low-quality face detection
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from face_recognition_engine import FaceRecognitionEngine
from utils import load_and_preprocess_image, get_image_files
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LowQualityDetectionTester:
    """
    Test class to validate improvements in detecting faces in low-quality images
    """
    
    def __init__(self):
        self.engine = FaceRecognitionEngine()
        self.results = {}
    
    def simulate_low_quality_conditions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create various low-quality versions of an image for testing
        
        Args:
            image: Original high-quality image
            
        Returns:
            Dictionary of degraded images with condition names
        """
        degraded_images = {}
        
        try:
            # 1. Blur simulation (camera shake, motion blur)
            blur_kernel = cv2.getGaussianKernel(9, 2.0)
            blurred = cv2.sepFilter2D(image, -1, blur_kernel, blur_kernel)
            degraded_images['blurred'] = blurred
            
            # 2. Low resolution simulation
            height, width = image.shape[:2]
            small = cv2.resize(image, (width//3, height//3))
            low_res = cv2.resize(small, (width, height))
            degraded_images['low_resolution'] = low_res
            
            # 3. Noise simulation
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
            degraded_images['noisy'] = noisy
            
            # 4. Low contrast/brightness
            dark = cv2.convertScaleAbs(image, alpha=0.5, beta=-30)
            degraded_images['dark'] = dark
            
            # 5. Overexposed/bright
            bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
            degraded_images['overexposed'] = bright
            
            # 6. JPEG compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
            _, encimg = cv2.imencode('.jpg', image, encode_param)
            compressed = cv2.imdecode(encimg, 1)
            if compressed is not None:
                degraded_images['compressed'] = compressed
            
            # 7. Combined degradation (worst case)
            combined = cv2.GaussianBlur(dark, (5, 5), 1.5)
            combined = cv2.add(combined, np.random.normal(0, 15, combined.shape).astype(np.uint8))
            degraded_images['combined_worst'] = combined
            
            logger.info(f"Created {len(degraded_images)} degraded versions of the image")
            return degraded_images
            
        except Exception as e:
            logger.error(f"Error creating degraded images: {e}")
            return {}
    
    def test_detection_improvements(self, image_path: Path) -> Dict[str, Any]:
        """
        Test face detection on original and degraded versions of an image
        
        Args:
            image_path: Path to test image
            
        Returns:
            Test results dictionary
        """
        try:
            logger.info(f"Testing detection improvements on: {image_path.name}")
            
            # Load original image
            original_image = load_and_preprocess_image(image_path)
            if original_image is None:
                return {"error": f"Failed to load image: {image_path}"}
            
            results = {
                "image_path": str(image_path),
                "original_faces": 0,
                "degraded_results": {},
                "improvement_summary": {}
            }
            
            # Test on original image
            logger.info("Testing on original image...")
            original_faces = self.engine.detect_faces(original_image, enhanced_detection=True)
            results["original_faces"] = len(original_faces)
            logger.info(f"Original image: {len(original_faces)} faces detected")
            
            # Test on degraded versions
            degraded_images = self.simulate_low_quality_conditions(original_image)
            
            for condition, degraded_image in degraded_images.items():
                logger.info(f"Testing on {condition} image...")
                
                try:
                    # Test with old method (single backend, no enhancement)
                    old_faces = self.engine.detect_faces(degraded_image, enhanced_detection=False)
                    
                    # Test with new enhanced method
                    new_faces = self.engine.detect_faces(degraded_image, enhanced_detection=True)
                    
                    condition_results = {
                        "old_method_faces": len(old_faces),
                        "new_method_faces": len(new_faces),
                        "improvement": len(new_faces) - len(old_faces),
                        "improvement_percent": ((len(new_faces) - len(old_faces)) / max(1, len(old_faces))) * 100
                    }
                    
                    results["degraded_results"][condition] = condition_results
                    
                    logger.info(f"{condition}: Old={len(old_faces)}, New={len(new_faces)}, "
                              f"Improvement={condition_results['improvement']} (+{condition_results['improvement_percent']:.1f}%)")
                    
                except Exception as condition_error:
                    logger.error(f"Error testing {condition}: {condition_error}")
                    results["degraded_results"][condition] = {"error": str(condition_error)}
            
            # Calculate overall improvement summary
            total_old = sum(r.get("old_method_faces", 0) for r in results["degraded_results"].values() if "error" not in r)
            total_new = sum(r.get("new_method_faces", 0) for r in results["degraded_results"].values() if "error" not in r)
            
            results["improvement_summary"] = {
                "total_faces_old_method": total_old,
                "total_faces_new_method": total_new,
                "total_improvement": total_new - total_old,
                "average_improvement_percent": ((total_new - total_old) / max(1, total_old)) * 100 if total_old > 0 else 0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing image {image_path}: {e}")
            return {"error": str(e), "image_path": str(image_path)}
    
    def test_embedding_quality(self, image_path: Path) -> Dict[str, Any]:
        """
        Test embedding quality improvements on low-quality images
        
        Args:
            image_path: Path to test image
            
        Returns:
            Embedding quality test results
        """
        try:
            logger.info(f"Testing embedding quality on: {image_path.name}")
            
            original_image = load_and_preprocess_image(image_path)
            if original_image is None:
                return {"error": f"Failed to load image: {image_path}"}
            
            results = {
                "image_path": str(image_path),
                "embedding_results": {}
            }
            
            # Detect faces first
            faces = self.engine.detect_faces(original_image, enhanced_detection=True)
            if not faces:
                return {"error": "No faces detected in image"}
            
            # Test first face
            face_location = faces[0]
            
            # Create degraded versions
            degraded_images = self.simulate_low_quality_conditions(original_image)
            
            for condition, degraded_image in degraded_images.items():
                try:
                    # Test embedding extraction with old method
                    old_embedding = self.engine.extract_face_embedding(degraded_image, face_location, high_quality=False)
                    
                    # Test embedding extraction with new enhanced method
                    new_embedding = self.engine.extract_face_embedding(degraded_image, face_location, high_quality=True)
                    
                    condition_results = {
                        "old_method_success": old_embedding is not None,
                        "new_method_success": new_embedding is not None,
                        "old_embedding_quality": self.engine._validate_embedding_quality(old_embedding) if old_embedding is not None else False,
                        "new_embedding_quality": self.engine._validate_embedding_quality(new_embedding) if new_embedding is not None else False
                    }
                    
                    results["embedding_results"][condition] = condition_results
                    
                    logger.info(f"{condition}: Old success={condition_results['old_method_success']}, "
                              f"New success={condition_results['new_method_success']}")
                    
                except Exception as condition_error:
                    logger.error(f"Error testing embedding for {condition}: {condition_error}")
                    results["embedding_results"][condition] = {"error": str(condition_error)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing embedding quality for {image_path}: {e}")
            return {"error": str(e), "image_path": str(image_path)}
    
    def run_comprehensive_test(self, test_images_dir: Path = None) -> Dict[str, Any]:
        """
        Run comprehensive test on available images
        
        Args:
            test_images_dir: Directory containing test images (optional)
            
        Returns:
            Comprehensive test results
        """
        try:
            if test_images_dir is None:
                test_images_dir = config.IMAGES_DIR
            
            logger.info(f"Running comprehensive test on images in: {test_images_dir}")
            
            # Find test images
            image_files = get_image_files(test_images_dir)
            if not image_files:
                logger.warning(f"No images found in {test_images_dir}")
                return {"error": "No test images found"}
            
            # Limit to first 5 images for testing
            test_images = image_files[:5]
            logger.info(f"Testing on {len(test_images)} images")
            
            comprehensive_results = {
                "test_summary": {
                    "total_images_tested": len(test_images),
                    "successful_tests": 0,
                    "failed_tests": 0
                },
                "detection_results": [],
                "embedding_results": []
            }
            
            for image_path in test_images:
                try:
                    logger.info(f"Processing image: {image_path.name}")
                    
                    # Test detection improvements
                    detection_results = self.test_detection_improvements(image_path)
                    comprehensive_results["detection_results"].append(detection_results)
                    
                    # Test embedding quality improvements
                    embedding_results = self.test_embedding_quality(image_path)
                    comprehensive_results["embedding_results"].append(embedding_results)
                    
                    if "error" not in detection_results and "error" not in embedding_results:
                        comprehensive_results["test_summary"]["successful_tests"] += 1
                    else:
                        comprehensive_results["test_summary"]["failed_tests"] += 1
                        
                except Exception as image_error:
                    logger.error(f"Error processing {image_path}: {image_error}")
                    comprehensive_results["test_summary"]["failed_tests"] += 1
            
            # Calculate overall statistics
            all_detection_results = [r for r in comprehensive_results["detection_results"] if "error" not in r]
            
            if all_detection_results:
                total_improvement = sum(r["improvement_summary"]["total_improvement"] 
                                      for r in all_detection_results)
                avg_improvement_percent = np.mean([r["improvement_summary"]["average_improvement_percent"] 
                                                 for r in all_detection_results])
                
                comprehensive_results["overall_statistics"] = {
                    "total_additional_faces_detected": total_improvement,
                    "average_improvement_percentage": avg_improvement_percent
                }
            
            logger.info("Comprehensive test completed")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive test: {e}")
            return {"error": str(e)}
    
    def print_test_summary(self, results: Dict[str, Any]):
        """
        Print a human-readable summary of test results
        """
        print("\n" + "="*60)
        print("FACE DETECTION IMPROVEMENTS - TEST RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"Test failed: {results['error']}")
            return
        
        if "overall_statistics" in results:
            stats = results["overall_statistics"]
            print(f"Total additional faces detected: {stats['total_additional_faces_detected']}")
            print(f"Average improvement: {stats['average_improvement_percentage']:.1f}%")
        
        print(f"\nTest Summary:")
        print(f"- Images tested: {results['test_summary']['total_images_tested']}")
        print(f"- Successful: {results['test_summary']['successful_tests']}")
        print(f"- Failed: {results['test_summary']['failed_tests']}")
        
        print("\nDetailed Results:")
        for i, detection_result in enumerate(results.get("detection_results", [])):
            if "error" in detection_result:
                continue
                
            print(f"\nImage {i+1}: {Path(detection_result['image_path']).name}")
            print(f"Original faces: {detection_result['original_faces']}")
            
            for condition, cond_result in detection_result["degraded_results"].items():
                if "error" in cond_result:
                    continue
                print(f"  {condition}: {cond_result['old_method_faces']} → {cond_result['new_method_faces']} "
                      f"(+{cond_result['improvement']}, {cond_result['improvement_percent']:+.1f}%)")

def main():
    """
    Main function to run the low-quality detection tests
    """
    print("Starting Low-Quality Face Detection Improvement Tests...")
    
    tester = LowQualityDetectionTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print results
    tester.print_test_summary(results)
    
    # Save results to file
    import json
    results_file = Path(__file__).parent / "test_results_low_quality_detection.json"
    try:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()