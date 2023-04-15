import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the input shape of the images
input_shape = (500, 500, 3)

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(4, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Set the data generator for training and validation data
data_dir = "data"
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2)

# Set the batch size and number of epochs
batch_size = 32
epochs = 2

# Generate the training and validation data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    subset="training")

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation")

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator)

# Save the model
model.save("quadrant_classifier.h5")
