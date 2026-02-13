from fastapi.responses import HTMLResponse
from fastapi import Request

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Livestock Disease Detection API")

# ---- Load Model Once ----
#model = tf.keras.models.load_model("mobilenetv2_multilabel_final_final.h5")
#model = tf.keras.models.load_model("mobilenetv2_multilabel_final_mild_noise_augmented_dataset_single_only.h5")
#model = tf.keras.models.load_model("mobilenetv2_multilabel_final_mild-noise-augmented-dataset-combined.h5")
model = tf.keras.models.load_model("mobilenetv2_multilabel_final_pair_augmented_dataset.h5")
#model = tf.keras.models.load_model("mobilenetv2_multilabel_final_single_augmented_dataset.h5")

# Class names (must match training order)
classes = ["FMD", "IBK", "Lumpy"]

# ---- Image Preprocessing ----
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---- Prediction Endpoint ----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image)[0]

    result = {}
    threshold = 0.5

    for i, disease in enumerate(classes):
        result[disease] = {
            "predicted": bool(predictions[i] > threshold),
            "confidence": float(predictions[i])
        }

    # Healthy logic
    if all(predictions < threshold):
        result["Healthy"] = True
    else:
        result["Healthy"] = False

    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Livestock Disease Detection</title>
        </head>
        <body>
            <h2>Upload Cattle Image</h2>
            <form action="/predict-web" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

import base64

@app.post("/predict-web", response_class=HTMLResponse)
async def predict_web(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # Convert to base64 for preview
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    image_src = f"data:image/jpeg;base64,{encoded_image}"

    processed_image = preprocess_image(image_bytes)
    predictions = model.predict(processed_image)[0]
    threshold = 0.5

    result_html = """
    <html>
    <head>
        <title>Prediction Result</title>
    </head>
    <body>
        <h2>Uploaded Image:</h2>
        <img src="{}" width="300"><br><br>
        <h3>Prediction Results:</h3>
        <ul>
    """.format(image_src)

    detected_diseases = []

    for i, disease in enumerate(classes):
        confidence = float(predictions[i])
        predicted = confidence > threshold

        if predicted:
            detected_diseases.append(disease)

        result_html += f"<li>{disease}: {'Detected' if predicted else 'Not Detected'} (Confidence: {confidence:.2f})</li>"

    result_html += "</ul>"

    # -------- STATUS LOGIC --------
    if len(detected_diseases) == 0:
        status_message = "No cattle disease is found"
    else:
        diseases_list = ", ".join(detected_diseases)
        status_message = f"Cattle disease found: {diseases_list}"

    result_html += f"""
        <h2 style="color:red; font-weight:bold;">
            Status: {status_message}
        </h2>

        <br>
        <a href="/">Upload Another Image</a>
    </body>
    </html>
    """

    return result_html






