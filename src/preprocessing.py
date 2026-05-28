import cv2
import numpy as np
import os

def ben_graham_filter(image_path, target_size=224, sigmaX=10):
    """
    Applies Ben Graham's local contrast enhancement optimization technique 
    to isolate clinical pathology features from digital fundus photography.
    """
    # 1. Load the raw image
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    # 2. Convert from BGR (OpenCV default) to RGB (TensorFlow/Keras default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Auto-crop black padding borders to isolate the eyeball sphere
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    
    # Check if image is completely dark to avoid zero-division errors
    if not np.any(mask):
        return cv2.resize(image, (target_size, target_size))
        
    # Crop bounding box around mask
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    image = image[y:y+h, x:x+w]
    
    # 4. Resize to target dimension specified by EfficientNetB0 standard input
    image = cv2.resize(image, (target_size, target_size))
    
    # 5. Apply the Ben Graham Gaussian Blur subtraction filter
    # Formula: Processed = Alpha * Input + Beta * Blurred + Gamma
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    # 6. Apply a smooth circular mask overlay to eliminate boundary noise
    height, width, _ = enhanced.shape
    mask_circle = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_circle, (int(width / 2), int(height / 2)), int(target_size / 2 * 0.9), 255, -1)
    
    final_image = cv2.bitwise_and(enhanced, enhanced, mask=mask_circle)
    
    return final_image