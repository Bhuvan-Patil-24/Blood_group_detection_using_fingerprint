import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self):
        self.target_size = (128, 128)
    
    def extract_features(self, img):
        """
        Extract features from preprocessed fingerprint image
        Returns: Feature vector
        """
        try:
            # Ensure image is in the correct format
            if img is None:
                raise ValueError("Input image is None")
            
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = img.astype(np.uint8)
            
            # Resize if necessary
            if gray.shape != self.target_size:
                gray = cv2.resize(gray, self.target_size)
            
            # 1. Extract LBP features
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # 2. Extract HOG features
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=False)
            
            # 3. Extract statistical features
            mean = np.mean(gray)
            std = np.std(gray)
            entropy = np.sum(-p * np.log2(p + 1e-10) for p in np.histogram(gray, bins=256)[0] / gray.size)
            
            # 4. Ridge orientation features
            # Compute gradients
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(gy, gx) * 180 / np.pi % 180
            magnitude = np.sqrt(gx**2 + gy**2)
            
            # Compute orientation histogram
            hist_orient = np.histogram(orientation, bins=9, weights=magnitude)[0]
            hist_orient = hist_orient / (hist_orient.sum() + 1e-10)  # Normalize
            
            # Combine all features
            features = np.concatenate([
                lbp.ravel() / (lbp.max() + 1e-10),  # Normalized LBP
                hog_features,                        # HOG features
                [mean / 255.0, std / 255.0, entropy],  # Statistical features
                hist_orient                         # Ridge orientation features
            ])
            
            # Ensure features are normalized
            features = (features - features.mean()) / (features.std() + 1e-10)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
