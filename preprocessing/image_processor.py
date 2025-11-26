"""
Fingerprint Image Preprocessor for Blood Group Detection
Optimized pipeline for fingerprint enhancement and normalization
"""

import cv2
import numpy as np
from pathlib import Path


class ImageProcessor:
    """
    Preprocessing pipeline for fingerprint images before ML model training/inference
    """
    
    def __init__(self, target_size=(128, 128)):
        """
        Initialize preprocessor with target image dimensions
        
        Args:
            target_size (tuple): Target dimensions (width, height) for output image
        """
        self.target_width, self.target_height = target_size
        
    def preprocess_fingerprint(self, image_path):
        """
        Main preprocessing pipeline for fingerprint images
        
        Args:
            image_path (str): Path to input fingerprint image
            
        Returns:
            np.ndarray: Preprocessed image as float32 array normalized to [0,1]
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Step 1: Validate and load image
            img = self._load_image(image_path)
            
            # Step 2: Resize with aspect ratio preservation and padding
            resized_img = self._resize_with_padding(img)
            
            # Step 3: Convert to grayscale
            gray_img = self._convert_to_grayscale(resized_img)
            
            # Step 4: Noise reduction using bilateral filtering
            denoised_img = self._reduce_noise(gray_img)
            
            # # Step 6: Ridge enhancement using Gabor filters
            # ridge_enhanced_img = self._enhance_ridges(denoised_img)
            
            # Step 7: Adaptive thresholding for better ridge clarity
            thresholded_img = self._apply_adaptive_threshold(denoised_img)
            
            # # Step 8: Morphological cleanup
            # cleaned_img = self._morphological_cleanup(thresholded_img)
            
            # Step 9: Normalize to [0, 1] range as float32
            normalized_img = self._normalize_image(thresholded_img)
            
            return normalized_img
            
        except Exception as e:
            raise ValueError(f"Preprocessing failed for {image_path}: {str(e)}")
    
    def _load_image(self, image_path):
        """Load and validate image file"""
        # Check if file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Failed to load image. Check if file is a valid image: {image_path}")
        
        return img
    
    def _resize_with_padding(self, img):
        """
        Resize image to target size while maintaining aspect ratio
        Pad with black background to fit exact dimensions
        """
        h, w = img.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = w / h
        target_aspect = self.target_width / self.target_height
        
        # Determine new dimensions maintaining aspect ratio
        if aspect_ratio > target_aspect:
            # Width is limiting factor
            new_width = self.target_width
            new_height = int(self.target_width / aspect_ratio)
        else:
            # Height is limiting factor
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        
        # Resize image
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create black canvas of target size
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets to center the image
        y_offset = (self.target_height - new_height) // 2
        x_offset = (self.target_width - new_width) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def _convert_to_grayscale(self, img):
        """Convert image to grayscale"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _reduce_noise(self, img):
        """
        Apply bilateral filtering for noise reduction
        Preserves edges while smoothing noise
        """
        # Bilateral filter: d=9, sigmaColor=75, sigmaSpace=75
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised
    
    def _enhance_contrast(self, img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Enhances local contrast without amplifying noise
        """
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        return enhanced
    
    def _enhance_ridges(self, img):
        """
        Apply Gabor filter for fingerprint ridge enhancement
        Enhances ridge patterns at multiple orientations
        """
        # Gabor filter parameters optimized for fingerprints
        ksize = 32  # Kernel size
        sigma = 4.5  # Standard deviation
        lambd = 9   # Wavelength
        gamma = 0.5  # Aspect ratio
        
        # Apply Gabor filters at multiple orientations and combine
        enhanced = np.zeros_like(img, dtype=np.float32)
        
        # Use 4 orientations: 0°, 45°, 90°, 135°
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            enhanced += filtered
        
        # Normalize and convert back to uint8
        enhanced = np.clip(enhanced / 4.0, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _apply_adaptive_threshold(self, img):
        """
        Apply adaptive thresholding to create binary image
        Improves ridge clarity by adjusting threshold locally
        """
        binary = cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,  # Size of pixel neighborhood
            C=3           # Constant subtracted from mean
        )
        return binary
    
    def _morphological_cleanup(self, img):
        """
        Apply morphological operations to clean up binary image
        Removes small artifacts and closes gaps
        """
        # Create elliptical structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply morphological opening (erosion followed by dilation)
        # Removes small noise
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply morphological closing (dilation followed by erosion)
        # Closes small gaps in ridges
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closed
    
    def _normalize_image(self, img):
        """
        Normalize image to [0, 1] range as float32
        Ensures consistent input scale for ML models
        """
        # Convert to float32 and normalize to [0, 1]
        normalized = img.astype(np.float32) / 255.0
        
        return normalized


# # Example usage
# if __name__ == "__main__":
#     # Initialize preprocessor
#     preprocessor = ImageProcessor(target_size=(128, 128))
    
#     # Process a single image
#     try:
#         processed_image = preprocessor.preprocess_fingerprint("sample_fingerprint.jpg")
#         print(f"Preprocessing successful!")
#         print(f"Output shape: {processed_image.shape}")
#         print(f"Output dtype: {processed_image.dtype}")
#         print(f"Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
#         # Optional: Save processed image for visualization
#         cv2.imwrite("processed_fingerprint.jpg", (processed_image * 255).astype(np.uint8))
        
#     except Exception as e:
#         print(f"Error: {e}")