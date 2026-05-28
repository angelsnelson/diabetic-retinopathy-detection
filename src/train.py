import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from preprocessing import ben_graham_filter

# Ensure model saving directory exists
os.makedirs("models", exist_ok=True)

print(" [System] Initializing Data Loading Pipeline...")

# 1. Load Labels CSV Spreadsheet
CSV_PATH = "data/train.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing required metadata file at {CSV_PATH}. Awaiting unzip completion.")

df = pd.read_csv(CSV_PATH)

# Standardize path mapping (id_code contains image filenames)
df['image_path'] = df['id_code'].apply(lambda x: f"data/train_images/{x}.png")
df['diagnosis'] = df['diagnosis'].astype(str) # Convert target labels to strings for Keras generators

# 2. Split Dataset into Clinical Train and Cross-Validation Sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])

print(f" Dataset Stratification Complete.")
print(f"   |-- Training Samples: {len(train_df)}")
print(f"   |-- Validation Samples: {len(val_df)}")

# 3. Keras Data Generators with Built-In Augmentation to Prevent Overfitting
# (Simulating our Ben Graham filter inline via ImageDataGenerator preprocessing function)
def custom_preprocessing_wrapper(img_path):
    # This acts as a wrapper bridging disk file loading to our dynamic training streams
    return ben_graham_filter(img_path, target_size=224)

# For performance, we use standard real-time augmentation to make the model robust to shifts
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# 4. Stream Image Batches Out of Storage (Saves RAM)
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='diagnosis',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =====================================================================
# REPLACED: 5. Build Upgraded Deep Learning Architecture (Fine-Tuning)
# =====================================================================
print(" [Model] Loading EfficientNetB0 Base Foundation for Fine-Tuning...")

base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# --- THE FINE-TUNING CRUX ---
base_model.trainable = True
# Freeze the early layers, unfreeze the top 30 layers to specialize in medical textures
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Construct a deeper classification head to manage the complex clinical details
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Prevents training memorization (overfitting)
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax') # 5 Clinical Output classes
])

# Compile with a surgical learning rate (1e-5) so it tweaks weights gently without destroying features
print(" [Model] Compiling Engine with Surgical Clinical Learning Rate...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
)

# =====================================================================
# REPLACED: 6. Updated Safety Callbacks for Fine-Tuning
# =====================================================================
clinical_callbacks = [
    callbacks.ModelCheckpoint(
        filepath="models/best_dr_model.h5",
        monitor="val_accuracy", # Stabilize save points on validation accuracy during fine-tuning
        save_best_only=True,
        mode="max"
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    )
]

print(" Pipeline fully armed with Fine-Tuning parameters. Ready to train.")

# =====================================================================
# 7. Start the Actual Engine Training Loop!
# =====================================================================
print("🏋️‍♂️ [Training] Commencing Deep Learning Optimization...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=clinical_callbacks
)

print(" [Success] Training phase complete! Upgraded weights locked in 'models/best_dr_model.h5'")