# src/processor.py
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class PartExtractor:
    def __init__(self, checkpoint_path, model_type="vit_b", device="cpu"):
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def load_image(self, path):
        return cv2.imread(path)

    def get_mask(self, image, points):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_image)
        
        input_points = np.array(points)
        # We use '1' for all points to mark them as foreground
        input_labels = np.ones(len(input_points))

        # IMPORTANT: SAM can return 3 masks. 
        # Sometimes the 'best' mask (index 0) is empty if the point is tiny.
        # We will pick the mask with the most 'True' pixels.
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True, # Change to True to see all options
        )
        
        # Select the mask that actually contains data
        mask_areas = [np.sum(m) for m in masks]
        return masks[np.argmax(mask_areas)]

    def enhance_image(self, image):
        # Let's keep it simple: just a slight contrast boost
        # Over-processing often ruins technical drawings
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)