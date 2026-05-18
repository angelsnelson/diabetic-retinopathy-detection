import numpy as np
import tensorflow as tf
from preprocessing import load_and_preprocess_image

# Map the numerical class outputs (0-4) back to human-readable medical diagnoses
DIAGNOSIS_MAP = {
    0: "No Diabetic Retinopathy (Healthy Retina)",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy (Advanced Stage)"
}

def predict_eye_disease(image_path, model_path='models/best_dr_model.h5'):
    """
    Takes a path to a raw eye image, preprocesses it using domain specific rules,
    runs it through the saved model weights, and outputs the diagnostic staging.
    """
    print(head := f"Analyzing image: {image_path}...")
    
    # 1. Preprocess the image using the Ben Graham pipeline we built earlier
    processed_img = load_and_preprocess_image(image_path, output_size=224)
    
    # 2. Prepare the image for neural network batch processing
    # Normalizes pixel values between 0 and 1, and changes shape from (224,224,3) to (1,224,224,3)
    input_tensor = np.expand_dims(processed_img, axis=0) / 255.0
    
    try:
        # 3. Load the trained model weights
        model = tf.keras.models.load_model(model_path)
        
        # 4. Generate prediction probabilities across all 5 classes
        predictions = model.predict(input_tensor)
        
        # 5. Extract the class index with the highest probability score
        predicted_class = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class] * 100
        
        # 6. Translate to readable results
        diagnosis = DIAGNOSIS_MAP[predicted_class]
        
        print("\n=== CLINICAL PREDICTION RESULTS ===")
        print(f"Diagnosis: {diagnosis}")
        print(f"Model Confidence: {confidence_score:.2f}%")
        print("===================================\n")
        
        return predicted_class, confidence_score
        
    except IOError:
        # Friendly fallback if running the script before training weights locally
        print(f"\n[Simulation Notice] Model weights file not found at '{model_path}'.")
        print("In production, this script loads saved weights to serve real-time predictions.")
        print(f"Pipeline flow validated successfully: Image preprocessed to shape {input_tensor.shape}.\n")
        return None, None

if __name__ == "__main__":
    # Example validation path to test the pipeline flow
    predict_eye_disease(image_path='data/sample_eye.png')
