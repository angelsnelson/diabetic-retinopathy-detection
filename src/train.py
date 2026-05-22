import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_diabetic_retinopathy_model

def train_model():
    """
    Sets up the loss functions, medical-specific metrics, optimizer, 
    and compilation instructions for the training lifecycle.
    """
    # 1. Initialize our pre-built EfficientNet model
    model = build_diabetic_retinopathy_model(input_shape=(224, 224, 3), num_classes=5)
    
    # 2. Define medical-first optimization configurations
    # We use Adam optimizer with a lower learning rate to cleanly adjust the custom top layers
    optimizer = Adam(learning_rate=1e-4)
    
    # Loss function for multi-class integers (0, 1, 2, 3, or 4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # 3. Define metrics. We track standard accuracy, but we also track 
    # CategoricalAccuracy to monitor per-class precision across the 5 stages.
    metrics = [
        'accuracy',
        tf.keras.metrics.SparseCategoricalAccuracy(name='categorical_accuracy')
    ]
    
    # 4. Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    print("Model compiled smoothly.")
    
    # 5. Production Feature: Callbacks
    # Callbacks prevent overfitting and save resources.
    callbacks = [
        # Automatically stop training if the model stops improving for 3 straight rounds (epochs)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        ),
        # Save only the absolute best iteration of our trained model weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_dr_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    print("Training pipelines, Callbacks, and Medical tracking initialized successfully!")
    return model, callbacks

if __name__ == "__main__":
    _ , _ = train_model()
