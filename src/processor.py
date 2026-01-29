import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class PartExtractor:
    def __init__(self, checkpoint_path, model_type="vit_b", device="mps"):
        self.device = device
        # Load the SAM model weights
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(f"Image not found at {path}")
        return img

    def get_mask(self, image, point):
        # image should be BGR (OpenCV standard)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]), 
            multimask_output=True
        )
        return masks[np.argmax(scores)]

    def enhance_image(self, image_crop):
        # Convert to gray and apply CLAHE for high-contrast 2D lines
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Laplacian-style sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)