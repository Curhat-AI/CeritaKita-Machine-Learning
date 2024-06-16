import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Define and register the custom activation function
def swish_activation(x):
    return tf.nn.swish(x)

get_custom_objects().update({'swish_activation': swish_activation})

# Load the model
model = load_model(os.getenv("IMAGE_MODEL_PATH"))

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((48, 48))  # Adjust the size to match your model input
    img = img.convert('L')  # Convert image to grayscale if needed
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch axis
    img_array = np.expand_dims(img_array, axis=1)  # Add channel axis
    return img_array

def predict_image(contents: bytes) -> list:
    img = Image.open(io.BytesIO(contents))
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    
    # Get the top 2 predictions
    top_indices = prediction[0].argsort()[-2:][::-1]
    top_class_names = [class_names[i] for i in top_indices]
    
    return top_class_names
