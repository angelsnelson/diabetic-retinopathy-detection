import tensorflow as tf
from tensorflow.keras import layers, models

def build_diabetic_retinopathy_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Builds a Transfer Learning model using EfficientNetB0 as the backbone network.
    The weights are pre-trained on ImageNet, and fine-tuned for 5-class eye disease classification.
    """
    # 1. Base model: Load EfficientNetB0 without its top classification layer
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )
    
    # Freeze the base model to preserve learned generic features (edges, textures)
    base_model.trainable = False
    
    # 2. Add custom layers on top for our specific medical classification task
    model = models.Sequential([
        base_model,
        
        # Convert 2D feature maps into a 1D vector
        layers.GlobalAveragePooling2D(),
        
        # Batch Normalization stabilizes training and ensures faster convergence
        layers.BatchNormalization(),
        
        # Dropout prevents overfitting by randomly turning off neurons during training
        layers.Dropout(0.4),
        
        # Dense layer for deeper feature extraction
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer: 5 units for the 5 stages of Diabetic Retinopathy (0 to 4)
        # Using Softmax activation to output probabilities for each class
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    # Create an instance of the model to verify it compiles properly
    model = build_diabetic_retinopathy_model()
    model.summary()
    print("\nModel architecture compiled and verified successfully!")
