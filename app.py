
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('skin_disease_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return np.argmax(prediction)

def upload_image():
    path = filedialog.askopenfilename()
    if path:
        result = predict_image(path)
        result_label.config(text=f"Predicted class index: {result}")

# GUI layout
root = tk.Tk()
root.title("Skin Disease Detector")
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)
result_label = tk.Label(root, text="Prediction will appear here")
result_label.pack(pady=10)
root.mainloop()
