# app.py

import streamlit as st
import requests
import cv2
import numpy as np

st.title("🛡️ Defense Part Extractor")

uploaded_file = st.file_uploader("Upload Drawing", type=['png', 'jpg'])

if uploaded_file:
    # Display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Drawing", use_container_width=True)

    # Multi-point input
    coords_input = st.text_input("Enter coordinates (x1,y1,x2,y2,x3,y3,x4,y4)", placeholder="e.g. 450,1200,480,1250,500,1190,460,1210")

    if st.button("Extract Part"):
        if coords_input:
            # Send to your API
            response = requests.post(
                f"http://127.0.0.1:8000/extract_multi/?filename={uploaded_file.name}&coords={coords_input}"
            )

            if response.status_code == 200:
                result = response.json()
                # Use .get() to avoid KeyError if the API returns something else
                saved_path = result.get('saved_at')

                if saved_path:
                    st.success(f"Success! Saved at: {saved_path}")
                    st.image(saved_path, caption="Extracted Part")
                else:
                    st.warning("Part extracted, but the API didn't provide a file path.")
                    st.write("Full API Response:", result)