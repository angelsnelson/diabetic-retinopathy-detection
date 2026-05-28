import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2

# Initialize the Web Server application
app = FastAPI(title="Diabetic Retinopathy Clinical Detection API")

# CRITICAL: Allow your frontend website (GitHub Pages) to securely read data from this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows connections from any website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store our loaded brain matrix
MODEL_PATH = "models/best_dr_model.h5"
model = None

# Stage Mapping Index Dict
SEVERITY_MAPPING = {
    0: "No Diabetic Retinopathy (Normal/Healthy)",
    1: "Mild Non-Proliferative Diabetic Retinopathy",
    2: "Moderate Non-Proliferative Diabetic Retinopathy",
    3: "Severe Non-Proliferative Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy (Advanced Stage)"
}

@app.on_event("startup")
def load_clinical_model():
    """Loads the heavy deep learning file into memory once when server fires up."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Weight matrix not found at {MODEL_PATH}. Run train.py first!")
    print(f" [API] Initializing and caching deep learning weights from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" [API] Weights locked and cached successfully into RAM!")

@app.get("/")
def home():
    return {"status": "online", "message": "Clinical AI Diagnostician backend engine running smoothly."}

@app.post("/predict")
async def predict_retinopathy(file: UploadFile = File(...)):
    """Receives an eye image file via web upload, processes it, and diagnoses it using our .h5 file."""
    if global_model := model:
        pass
    else:
         raise HTTPException(status_code=503, detail="Neural Network model not instantiated yet.")
         
    try:
        # 1. Read the bytes from the uploaded image payload safely
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid graphic format processed.")
            
        # 2. Pipeline Match: Convert to RGB and resize to match EfficientNet's 224x224 input grid
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # 3. Normalize values exactly like our data streams did (1./255)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 4. Expand dimensions from [224, 224, 3] to [1, 224, 224, 3] (TensorFlow expects batch format)
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # 5. Run Real Inference!
        predictions = model.predict(img_batch)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence_score = float(predictions[0][predicted_class_idx])
        
        # 6. Return JSON response to the browser website UI dashboard
        return {
            "success": True,
            "diagnosis_code": predicted_class_idx,
            "diagnosis_label": SEVERITY_MAPPING[predicted_class_idx],
            "confidence": f"{confidence_score * 100:.2f}%",
            "raw_probabilities": predictions[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Analysis Failure: {str(e)}")