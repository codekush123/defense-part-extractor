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
        # 1. Convert BGR to RGB (Critical for SAM)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_image)
        
        # 2. Convert point to numpy array
        input_point = np.array([point])
        input_label = np.array([1]) # 1 means 'Foreground'

        # 3. Predict masks
        # We set multimask_output=True to get 3 options, then pick the best
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Pick the mask with the highest confidence score
        return masks[np.argmax(scores)]
    def enhance_image(self, image_crop):
        # Convert to gray and apply CLAHE for high-contrast 2D lines
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Laplacian-style sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)