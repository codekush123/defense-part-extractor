import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from src.processor import PartExtractor
app = FastAPI(title="Defense Part Extractor API")

# Configuration
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/sam_vit_b_01ec64.pth"
DEVICE = "cpu"

# Initialize processor
extractor = PartExtractor(checkpoint_path=MODEL_PATH, device=DEVICE)

# Inside src/api.py
@app.post("/extract_multi/")
async def extract_part_multi(filename: str, coords: str):
    # Parse points
    raw_coords = [int(i) for i in coords.split(",")]
    point_list = [raw_coords[i:i+2] for i in range(0, len(raw_coords), 2)]

    input_path = os.path.join(RAW_DIR, filename)
    img = extractor.load_image(input_path)
    
    # Get the AI Mask
    mask = extractor.get_mask(img, point_list)
    
    # Create a pure white background
    final_output = np.ones_like(img) * 255
    
    # ONLY copy the pixels the AI identified
    final_output[mask] = img[mask]
    
    # Crop tightly to the object
    y_idx, x_idx = np.where(mask)
    if len(y_idx) > 0:
        y1, y2 = np.min(y_idx), np.max(y_idx)
        x1, x2 = np.min(x_idx), np.max(x_idx)
        final_output = final_output[y1:y2, x1:x2]
    
    output_path = os.path.join(PROCESSED_DIR, f"extracted_{filename}")
    cv2.imwrite(output_path, final_output)
    
    return {"status": "success", "saved_at": output_path}