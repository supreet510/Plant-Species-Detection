# predict_pipeline.py
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

class PlantPredictor:
    def __init__(self, model_path="plant_species_model.keras", class_file="class_names.txt"):
        self.model = keras.models.load_model(model_path)
        with open(class_file, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def preprocess_image(self, image: Image.Image):
        # Resize to match training image size (256x256)
        img = image.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # ❌ Do NOT divide by 255.0 (model trained without normalization)
        return img_array

    def predict(self, image: Image.Image):
        img_array = self.preprocess_image(image)
        preds = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100
        return predicted_class, confidence
