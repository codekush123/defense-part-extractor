import streamlit as st
import cv2
import numpy as np
from src.processor import PartExtractor

st.set_page_config(page_title="Defense Part Extractor", layout="wide")
st.title("🛡️ Defense Part Extractor UI")

@st.cache_resource
def get_extractor():
    return PartExtractor("models/sam_vit_b_01ec64.pth")

extractor = get_extractor()

uploaded_file = st.file_uploader("Upload Drawing", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.write("Clicking logic: For this demo, please use the API or enter coordinates below.")
    # Note: For full click-interaction, 'streamlit-opencv-canvas' is required as discussed.
    
    x = st.number_input("X Coordinate", value=0)
    y = st.number_input("Y Coordinate", value=0)

    if st.button("Extract Part"):
        with st.spinner("Processing..."):
            mask = extractor.get_mask(image, [x, y])
            isolated = np.ones_like(image) * 255
            isolated[mask] = image[mask]
            
            # Crop & Enhance
            coords = np.argwhere(mask)
            y0, x0 = coords.min(axis=0), coords.max(axis=0)
            final = extractor.enhance_image(isolated[y0[0]:x0[0], y0[1]:x0[1]])
            
            st.image(final, caption="Enhanced 2D Part")