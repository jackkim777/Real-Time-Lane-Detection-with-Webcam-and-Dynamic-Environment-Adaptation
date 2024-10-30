import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """
    def __init__(self):
        """ Init Thresholding."""
        pass

    def forward(self, img):
        """ Take an image and extract all relevant pixels.

        Parameters:
            img (np.array): Input image

        Returns:
            binary (np.array): A binary image representing all positions of relevant pixels.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        # Thresholds for yellow color in HSV space
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))

        # Apply thresholds for other channels (optional based on your requirement)
        s_binary = threshold_rel(s_channel, 0.7, 1.0)
        v_binary = threshold_rel(v_channel, 0.7, 1.0)

        # Combine masks to get final binary image
        combined_binary = np.zeros_like(h_channel)
        combined_binary[(yellow_mask > 0) | (s_binary > 0) | (v_binary > 0)] = 255
        
        return combined_binary
