import json
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_PATH = "modelo/cnn_cifar10.keras"
CLASSES_PATH = "modelo/class_names.json"

def load_classifier():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def predict_image(image, model, class_names):
    image = image.convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), probabilities
