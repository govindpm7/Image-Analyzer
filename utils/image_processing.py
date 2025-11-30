"""
Image processing utilities using OpenCV
"""
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        pass
    
    def resize_image(self, image, max_size=1024):
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        
        return cv2.resize(image, (new_w, new_h))
    
    def check_lighting(self, image):
        """Check if image is too dark"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:,:,2])
        brightness_score = (avg_brightness / 255.0) * 10
        needs_enhancement = avg_brightness < 80
        return needs_enhancement, brightness_score
    
    def validate_image(self, image_path):
        """Validate image file"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Invalid image file"
            if img.size == 0:
                return False, "Empty image"
            return True, None
        except Exception as e:
            return False, str(e)


