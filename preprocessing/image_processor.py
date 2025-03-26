import cv2
import numpy as np
from skimage import exposure

class ImageProcessor:
    def __init__(self):
        self.img_height = 128
        self.img_width = 128
    
    def process_image(self, image_path):
        """
        Process fingerprint image for feature extraction
        Steps:
        1. Load and resize image
        2. Convert to grayscale
        3. Apply advanced fingerprint enhancement
        4. Return normalized image
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Resize image while maintaining aspect ratio
            aspect = img.shape[1] / img.shape[0]
            if aspect > 1:
                new_width = self.img_width
                new_height = int(self.img_width / aspect)
            else:
                new_height = self.img_height
                new_width = int(self.img_height * aspect)
            
            resized = cv2.resize(img, (new_width, new_height))
            
            # Create a black canvas of target size
            processed_img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            
            # Calculate padding
            y_offset = (self.img_height - new_height) // 2
            x_offset = (self.img_width - new_width) // 2
            
            # Place the resized image in the center
            processed_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            # Convert to grayscale
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            
            # Advanced fingerprint enhancement
            # 1. Noise reduction with bilateral filter (preserves edges better)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 2. Enhance contrast using CLAHE with optimal parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 3. Ridge enhancement using Gabor filter
            # Prepare the kernel size
            ksize = 31
            sigma = 4.5
            theta = np.pi/4
            lambd = 10
            gamma = 0.5
            
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            ridge_enhanced = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)
            
            # 4. Local adaptive thresholding
            binary = cv2.adaptiveThreshold(
                ridge_enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,  # Larger block size for better adaptation
                4    # Slightly increased C value
            )
            
            # 5. Morphological operations for cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to RGB
            processed_img = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
            
            # Normalize to float32 in range [0, 1]
            processed_img = processed_img.astype(np.float32)
            
            return processed_img
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise ValueError(f"Image processing failed: {str(e)}")
    
    def segment_fingerprint(self, img):
        """
        Segment fingerprint from background
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Otsu's thresholding
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find sure foreground area
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Convert to uint8
        sure_fg = sure_fg.astype(np.uint8)
        
        return sure_fg
