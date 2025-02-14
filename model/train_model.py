import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dataset Path
dataset_path = "dataset"

# Load CSV File
csv_path = os.path.join(dataset_path, "butterflies and moths.csv")
df = pd.read_csv(csv_path)

# Image Augmentation (Optimized)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

img_size = (224, 224)
batch_size = 16  # Reduced batch size

# Training Data Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df[df["data set"] == "train"],
    directory=dataset_path,
    x_col="filepaths",
    y_col="labels",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

# Validation Data Generator
val_generator = train_datagen.flow_from_dataframe(
    dataframe=df[df["data set"] == "valid"],
    directory=dataset_path,
    x_col="filepaths",
    y_col="labels",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Optimized Model: MobileNetV2 (Lightweight & Fast)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(train_generator.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model (Lower Learning Rate for Stability)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model (Optimized for Speed)
model.fit(train_generator, validation_data=val_generator, epochs=1)


# Save Model
model.save("model/butterfly_moth_classifier.h5")

print("Model training complete. Saved as butterfly_moth_classifier.h5")
