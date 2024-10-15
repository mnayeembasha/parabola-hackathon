import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import io

# Path to your manually downloaded models
det_model_dir = './inference/det_r50_vd_db_infer'  # Detection model
rec_model_dir = './inference/rec_r34_vd_crnn_en_infer'  # Recognition model
cls_model_dir = './inference/cls_mv3_infer'  # Classification model

# Initialize PaddleOCR with all inferences
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_model_dir=det_model_dir,
    rec_model_dir=rec_model_dir,
    cls_model_dir=cls_model_dir
)

# Streamlit UI for image upload
st.title("Team-6 OCR App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to a NumPy array for PaddleOCR
    img_array = np.array(image)

    # Perform OCR using the specified models
    result = ocr.ocr(img_array, cls=True)

    # Prepare the JSON output
    output_json = {}
    line_number = 1
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]
            score = line[1][1]
            output_json[line_number] = {
                "data": text,
                "score": score
            }
            line_number += 1

    # Display the OCR results in JSON format
    st.subheader("OCR Results")
    st.json(output_json)
