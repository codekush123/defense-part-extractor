import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from processor import PartExtractor
app = FastAPI(title="Defense Part Extractor API")

# Configuration
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/sam_vit_b_01ec64.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize processor
extractor = PartExtractor(checkpoint_path=MODEL_PATH, device=DEVICE)

@app.post("/extract/")
async def extract_part(filename: str, x: int, y: int):
    input_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Raw image not found.")

    img = extractor.load_image(input_path)
    mask = extractor.get_mask(img, [x, y])
    
    isolated = np.ones_like(img) * 255
    isolated[mask] = img[mask]
    
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0), coords.max(axis=0)
    crop = isolated[y0[0]:x0[0], y0[1]:x0[1]]
    
    final_img = extractor.enhance_image(crop)
    
    output_path = os.path.join(PROCESSED_DIR, f"extracted_{filename}")
    cv2.imwrite(output_path, final_img)
    
    return {"status": "success", "saved_at": output_path}