import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

class PartExtractor:
    def __init__(self, checkpoint_path, model_type="vit_b", device="cpu"):
        # Load the SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image at {path}")
        return img

    def get_mask(self, image, points):
        # 1. Prepare image for SAM
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_image)
        
        # 2. Convert 4 points into a Bounding Box [x1, y1, x2, y2]
        # This is more stable than points for technical drawings
        pts = np.array(points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        input_box = np.array([x_min, y_min, x_max, y_max])

        # 3. Run Predictor
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks[0]

    def enhance_image(self, image):
        # Basic normalization to ensure the part is visible
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)