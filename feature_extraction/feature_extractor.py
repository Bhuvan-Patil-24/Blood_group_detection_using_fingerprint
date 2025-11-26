"""
Feature Extraction Module for Fingerprint Validation
Extracts HOG, LBP, statistical, and ridge orientation features
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from scipy.stats import entropy as scipy_entropy


class FingerprintFeatureExtractor:
    """
    Extract discriminative features from preprocessed fingerprint images
    for fingerprint vs non-fingerprint classification
    """
    
    def __init__(self, target_size=(128, 128)):
        """
        Initialize feature extractor
        
        Args:
            target_size (tuple): Expected input image dimensions (width, height)
        """
        self.target_size = target_size
        
        # HOG parameters optimized for fingerprints
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius  # 24 points
        
        # Ridge orientation histogram bins
        self.orientation_bins = 9
    
    def extract_features(self, img):
        """
        Extract complete feature vector from preprocessed fingerprint image
        
        Args:
            img (np.ndarray): Preprocessed image (can be grayscale or normalized float)
            
        Returns:
            np.ndarray: Flattened feature vector as float32
            
        Raises:
            ValueError: If image is invalid or feature extraction fails
        """
        try:
            # Step 1: Validate and prepare image
            gray = self._prepare_image(img)
            
            # Step 2: Extract HOG features
            hog_features = self._extract_hog_features(gray)
            
            # Step 3: Extract LBP features
            lbp_features = self._extract_lbp_features(gray)
            
            # Step 4: Extract statistical texture features
            stat_features = self._extract_statistical_features(gray)
            
            # Step 5: Extract ridge orientation features
            orientation_features = self._extract_ridge_orientation_features(gray)
            
            # Step 6: Combine all features into single vector
            feature_vector = np.concatenate([
                hog_features,
                lbp_features,
                stat_features,
                orientation_features
            ])
            
            # Step 7: Normalize feature vector for stable ML training
            normalized_features = self._normalize_features(feature_vector)
            
            return normalized_features.astype(np.float32)
            
        except Exception as e:
            raise ValueError(f"Feature extraction failed: {str(e)}")
    
    def _prepare_image(self, img):
        """
        Validate and convert image to uint8 grayscale format
        """
        if img is None:
            raise ValueError("Input image is None")
        
        # Convert to numpy array if needed
        img = np.array(img)
        
        # Handle normalized float images [0, 1]
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Convert to grayscale if color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Verify dimensions
        if gray.shape[:2] != self.target_size:
            gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        return gray
    
    def _extract_hog_features(self, gray):
        """
        Extract Histogram of Oriented Gradients (HOG) features
        Captures edge and gradient structure, useful for ridge patterns
        """
        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        return hog_features
    
    def _extract_lbp_features(self, gray):
        """
        Extract Local Binary Pattern (LBP) features using uniform patterns
        Captures local texture information
        """
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method='uniform'
        )
        
        # Compute histogram of uniform LBP patterns
        # For uniform LBP with P points, there are P + 2 bins
        n_bins = self.lbp_n_points + 2
        lbp_hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        
        return lbp_hist
    
    def _extract_statistical_features(self, gray):
        """
        Extract statistical texture features
        Captures overall intensity distribution characteristics
        """
        # Normalize to [0, 1] for consistent statistics
        normalized = gray.astype(np.float32) / 255.0
        
        # Mean intensity
        mean = np.mean(normalized)
        
        # Standard deviation (measure of contrast)
        std = np.std(normalized)
        
        # Entropy (measure of randomness/texture complexity)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        entropy = scipy_entropy(hist, base=2)
        
        # Skewness (measure of asymmetry)
        skewness = np.mean((normalized - mean) ** 3) / (std ** 3 + 1e-10)
        
        # Kurtosis (measure of tailedness)
        kurtosis = np.mean((normalized - mean) ** 4) / (std ** 4 + 1e-10)
        
        # Energy (measure of uniformity)
        energy = np.sum(hist ** 2)
        
        stat_features = np.array([mean, std, entropy, skewness, kurtosis, energy])
        
        return stat_features
    
    def _extract_ridge_orientation_features(self, gray):
        """
        Extract ridge orientation features using Sobel gradients
        Captures directional information specific to fingerprint ridges
        """
        # Compute gradients using Sobel operator
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and orientation
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx)
        
        # Convert orientation to degrees [0, 180)
        orientation_deg = (orientation * 180 / np.pi) % 180
        
        # Create weighted histogram using gradient magnitude as weights
        # This emphasizes strong ridge orientations
        hist_orient, _ = np.histogram(
            orientation_deg,
            bins=self.orientation_bins,
            range=(0, 180),
            weights=magnitude,
            density=True
        )
        
        # Additional orientation statistics
        # Compute orientation coherence (how aligned the ridges are)
        mean_orientation = np.arctan2(
            np.sum(magnitude * np.sin(2 * orientation)),
            np.sum(magnitude * np.cos(2 * orientation))
        ) / 2
        
        # Orientation strength (coherence measure)
        orientation_strength = np.sqrt(
            np.sum(magnitude * np.cos(2 * orientation)) ** 2 +
            np.sum(magnitude * np.sin(2 * orientation)) ** 2
        ) / (np.sum(magnitude) + 1e-10)
        
        # Mean gradient magnitude
        mean_magnitude = np.mean(magnitude)
        
        # Combine histogram with additional features
        orientation_features = np.concatenate([
            hist_orient,
            [orientation_strength, mean_magnitude]
        ])
        
        return orientation_features
    
    def _normalize_features(self, features):
        """
        Normalize feature vector using z-score normalization
        Ensures features have zero mean and unit variance
        """
        # Handle edge case of constant features
        feature_std = np.std(features)
        if feature_std < 1e-10:
            return features
        
        # Z-score normalization
        normalized = (features - np.mean(features)) / (feature_std + 1e-10)
        
        return normalized
    
    def get_feature_dimensions(self):
        """
        Calculate and return the total feature vector dimension
        Useful for ML model initialization
        """
        # Create dummy image to calculate dimensions
        dummy_img = np.zeros(self.target_size, dtype=np.uint8)
        dummy_features = self.extract_features(dummy_img)
        
        return len(dummy_features)