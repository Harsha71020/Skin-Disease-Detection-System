
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os

# Load model
model = load_model('skin_disease_model.h5')

# Load image
img_path = sys.argv[1] if len(sys.argv) > 1 else 'sample.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
predicted_class = np.argmax(pred)
print(f"Predicted class index: {predicted_class}")
