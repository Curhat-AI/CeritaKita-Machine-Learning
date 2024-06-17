import os
import tensorflow as tf
from transformers import MobileBertTokenizer
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model
import numpy as np
from googletrans import Translator

from transformers import TFMobileBertModel
from tensorflow.keras.utils import custom_object_scope

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense1 = Dense(128, activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(0.5)
        
        self.dense2 = Dense(64, activation='relu')
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(0.5)
        
        super(CustomLayer, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.batch_norm2(x)
        return self.dropout2(x)

def load_model_from_h5(model_path):
    with custom_object_scope({'CustomLayer': CustomLayer, 'TFMobileBertModel': TFMobileBertModel}):
        return load_model(model_path)

model = load_model_from_h5(os.getenv("TEXT_MODEL_PATH"))  

def preprocess_input(text_input):
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    tokenized_text = tokenizer(
        text_input,
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='tf' 
    )
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    return input_ids, attention_mask

def translate_text(text_input, target_language='en'):
    translator = Translator()
    translation = translator.translate(text_input, dest=target_language)
    return translation.text

def predict_text(text_input, top_n=2):
    emotion_mapping = {
        0: 'Sedih',
        1: 'Senang',
        2: 'Cinta',
        3: 'Marah',
        4: 'Takut',
        5: 'Terkejut'
    }

    translated_text = translate_text(text_input)
    input_ids, attention_mask = preprocess_input(translated_text)
    predictions = model.predict([input_ids, attention_mask])
    
    top_n_indices = np.argsort(predictions, axis=1)[0, -top_n:][::-1] 
    top_n_emotions = [emotion_mapping[idx] for idx in top_n_indices]
    
    return top_n_emotions