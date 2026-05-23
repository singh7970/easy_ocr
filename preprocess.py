import cv2
import numpy as np

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Minimal preprocessing for handwriting OCR
    Let EasyOCR handle most of the work - just enhance slightly
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Slight upscaling for better detail
    height, width = gray.shape
    if height < 1500:
        scale = 1500 / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Very light denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=5)
    
    # Slight contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # NO thresholding - keep grayscale for EasyOCR
    return enhanced
