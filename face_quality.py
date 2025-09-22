"""
Face Quality Assessment for Enhanced Recognition
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceQualityAssessor:
    """
    Assess face quality for better recognition results
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.50,
            'poor': 0.30
        }
    
    def assess_face_quality(self, face_image: np.ndarray, face_location: Tuple[int, int, int, int] = None) -> Dict[str, float]:
        """
        Comprehensive face quality assessment for uploaded images
        
        Args:
            face_image: Face image or full image
            face_location: Optional face location (top, right, bottom, left)
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # Extract face region if location provided
            if face_location is not None:
                top, right, bottom, left = face_location
                face_crop = face_image[top:bottom, left:right]
            else:
                face_crop = face_image
            
            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                return self._get_poor_quality_metrics()
            
            metrics = {}
            
            # 1. Sharpness Assessment (Laplacian variance)
            metrics['sharpness'] = self._assess_sharpness(face_crop)
            
            # 2. Brightness and Contrast Assessment
            brightness_contrast = self._assess_brightness_contrast(face_crop)
            metrics.update(brightness_contrast)
            
            # 3. Size Assessment
            metrics['size_quality'] = self._assess_size_quality(face_crop)
            
            # 4. Noise Assessment
            metrics['noise_level'] = self._assess_noise_level(face_crop)
            
            # 5. Color Balance Assessment
            metrics['color_balance'] = self._assess_color_balance(face_crop)
            
            # 6. Overall Quality Score
            metrics['overall_quality'] = self._calculate_overall_quality(metrics)
            
            # 7. Quality Category
            metrics['quality_category'] = self._get_quality_category(metrics['overall_quality'])
            
            # 8. Recommendations
            metrics['recommendations'] = self._generate_recommendations(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return self._get_poor_quality_metrics()
    
    def _assess_sharpness(self, face_image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance"""
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (empirically determined thresholds)
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            return float(sharpness_score)
            
        except Exception as e:
            logger.debug(f"Sharpness assessment failed: {e}")
            return 0.0
    
    def _assess_brightness_contrast(self, face_image: np.ndarray) -> Dict[str, float]:
        """Assess brightness and contrast levels"""
        try:
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Brightness assessment (mean intensity)
            mean_brightness = np.mean(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.6 for faces
            brightness_quality = 1.0 - abs(mean_brightness - 0.5) * 2.0
            brightness_quality = max(0.0, brightness_quality)
            
            # Contrast assessment (standard deviation)
            std_contrast = np.std(gray) / 128.0  # Normalize to 0-2 range
            contrast_quality = min(1.0, std_contrast)
            
            # Dynamic range assessment
            min_val, max_val = np.min(gray), np.max(gray)
            dynamic_range = (max_val - min_val) / 255.0
            
            return {
                'brightness_quality': float(brightness_quality),
                'contrast_quality': float(contrast_quality),
                'dynamic_range': float(dynamic_range)
            }
            
        except Exception as e:
            logger.debug(f"Brightness/contrast assessment failed: {e}")
            return {'brightness_quality': 0.0, 'contrast_quality': 0.0, 'dynamic_range': 0.0}
    
    def _assess_size_quality(self, face_image: np.ndarray) -> float:
        """Assess if face size is adequate for recognition"""
        try:
            height, width = face_image.shape[:2]
            face_area = height * width
            
            # Quality based on face area (empirically determined)
            if face_area >= 10000:  # 100x100 or larger
                size_quality = 1.0
            elif face_area >= 6400:  # 80x80
                size_quality = 0.9
            elif face_area >= 3600:  # 60x60
                size_quality = 0.7
            elif face_area >= 1600:  # 40x40
                size_quality = 0.5
            elif face_area >= 900:   # 30x30
                size_quality = 0.3
            else:
                size_quality = 0.1
            
            return float(size_quality)
            
        except Exception as e:
            logger.debug(f"Size assessment failed: {e}")
            return 0.0
    
    def _assess_noise_level(self, face_image: np.ndarray) -> float:
        """Assess noise level (lower is better for recognition)"""
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image
            
            # Use median filtering to estimate noise
            median_filtered = cv2.medianBlur(gray, 5)
            noise = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
            
            # Normalize noise level (lower noise = higher quality)
            noise_quality = max(0.0, 1.0 - (noise / 50.0))
            
            return float(noise_quality)
            
        except Exception as e:
            logger.debug(f"Noise assessment failed: {e}")
            return 0.5
    
    def _assess_color_balance(self, face_image: np.ndarray) -> float:
        """Assess color balance quality"""
        try:
            if len(face_image.shape) != 3:
                return 0.5  # Neutral score for grayscale
            
            # Calculate channel means
            mean_r = np.mean(face_image[:, :, 0])
            mean_g = np.mean(face_image[:, :, 1])
            mean_b = np.mean(face_image[:, :, 2])
            
            # Good color balance has channels relatively close to each other
            channel_std = np.std([mean_r, mean_g, mean_b])
            
            # Normalize (lower std = better balance)
            balance_quality = max(0.0, 1.0 - (channel_std / 100.0))
            
            return float(balance_quality)
            
        except Exception as e:
            logger.debug(f"Color balance assessment failed: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        try:
            # Weighted combination of quality metrics
            weights = {
                'sharpness': 0.25,
                'brightness_quality': 0.15,
                'contrast_quality': 0.20,
                'size_quality': 0.20,
                'noise_level': 0.15,
                'color_balance': 0.05
            }
            
            overall_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics and not np.isnan(metrics[metric]):
                    overall_score += metrics[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_score = overall_score / total_weight
            
            return float(np.clip(overall_score, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Overall quality calculation failed: {e}")
            return 0.0
    
    def _get_quality_category(self, overall_quality: float) -> str:
        """Get quality category based on score"""
        if overall_quality >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif overall_quality >= self.quality_thresholds['good']:
            return 'good'
        elif overall_quality >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> list:
        """Generate improvement recommendations based on quality metrics"""
        recommendations = []
        
        try:
            if metrics.get('sharpness', 0) < 0.4:
                recommendations.append("📸 Verwenden Sie ein schärferes Bild mit besserer Fokussierung")
            
            if metrics.get('brightness_quality', 0) < 0.5:
                recommendations.append("💡 Verbessern Sie die Beleuchtung - das Gesicht ist zu dunkel oder zu hell")
            
            if metrics.get('contrast_quality', 0) < 0.4:
                recommendations.append("🎨 Erhöhen Sie den Kontrast für bessere Gesichtsdefinition")
            
            if metrics.get('size_quality', 0) < 0.5:
                recommendations.append("🔍 Verwenden Sie ein größeres Bild oder nähere Aufnahme des Gesichts")
            
            if metrics.get('noise_level', 0) < 0.6:
                recommendations.append("✨ Reduzieren Sie das Bildrauschen durch bessere Kameraeinstellungen")
            
            if metrics.get('color_balance', 0) < 0.4:
                recommendations.append("🌈 Verbessern Sie die Farbbalance für natürlichere Hauttöne")
            
            if not recommendations:
                recommendations.append("✅ Bildqualität ist gut für die Gesichtserkennung")
            
        except Exception as e:
            logger.debug(f"Recommendation generation failed: {e}")
            recommendations = ["⚠️ Qualitätsanalyse unvollständig"]
        
        return recommendations
    
    def _get_poor_quality_metrics(self) -> Dict[str, float]:
        """Return poor quality metrics for error cases"""
        return {
            'sharpness': 0.0,
            'brightness_quality': 0.0,
            'contrast_quality': 0.0,
            'size_quality': 0.0,
            'noise_level': 0.0,
            'color_balance': 0.0,
            'overall_quality': 0.0,
            'quality_category': 'poor',
            'recommendations': ['❌ Bild konnte nicht analysiert werden - versuchen Sie ein anderes Bild']
        }

# Global instance for easy access
face_quality_assessor = FaceQualityAssessor()