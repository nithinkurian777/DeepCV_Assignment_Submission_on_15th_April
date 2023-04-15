import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model("quadrant_classifier.h5")

# Load the test image
img_path = "data/quadrant1/apple_2.jpg"
img = image.load_img(img_path, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0 # normalize the image

# Predict the quadrant of the test image
preds = model.predict(x)
quadrant = np.argmax(preds)

# Print the predicted quadrant
print("The image belongs to quadrant:", quadrant)
