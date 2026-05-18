import cv2
import numpy as np
import os

def crop_image_from_gray(img, tol=7):
    """
    Crops out the unnecessary black borders around the circular eye retina image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): # Image is too dark, return original
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_and_preprocess_image(image_path, output_size=224):
    """
    Loads an eye fundus image, crops it, resizes it, and applies 
    Ben Graham's processing to enhance blood vessel visibility.
    """
    # 1. Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Crop black borders
    image = crop_image_from_gray(image)
    
    # 3. Resize to standard model input size
    image = cv2.resize(image, (output_size, output_size))
    
    # 4. Apply Ben Graham's processing (Gaussian Blur weight blending)
    # This enhances features like exudates and hemorrhages
    processed_image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), output_size / 10), -4, 128)
    
    return processed_image

if __name__ == "__main__":
    print("Preprocessing script successfully initialized!")
    # This block allows us to test the script later locally
