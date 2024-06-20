import os
import tensorflow as tf
import numpy as np
from googletrans import Translator

from transformers import AutoTokenizer
from tensorflow.keras.models import load_model
import transformers


import re
import string

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def cleaning_text(tweet):
    stopwords_id = stopwords.words('indonesian')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_factory = StopWordRemoverFactory()
    more_stopword = ['dengan', 'ia','bahwa','oleh', 'dgn', 'yg']
    data = stop_factory.get_stop_words() + more_stopword + stopwords_id
    # stopword_indonesia = stop_factory.create_stop_word_remover()
  # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'(\[[reRE]+\s+[\w]+\])', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    #remove <string>
    tweet = re.sub("<.*?>+", "", tweet)
    #remove kata typo + campur angka (alay)
    tweet = re.sub("\w*\d\w*", "", tweet)
    #remove word start from number
    tweet = re.sub("\d+", "", tweet)
    tweet = re.sub("amp", " ", tweet)
    tweet = re.sub(r'(?:@\S*|#\S*|http(?=.*://)\S*)', '', tweet)
    #remove rp/RP (awalan uang)
    # tweet = re.sub(r'[rR][pP]', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    tweet = re.sub(r'[^a-zA-Z0-9 ,\.!?]', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in data and #stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return " ".join(tweets_clean)


def preprocess_input(text_input):
    tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
    tokenized_text = tokenizer(
        text_input,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True 
    )
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    return input_ids, attention_mask

def translate_text(text_input, target_language='en'):
    translator = Translator()
    translation = translator.translate(text_input, dest=target_language)
    return translation.text

def load_model_from_h5(model_path):
    return load_model(model_path, custom_objects={"TFBertModel": transformers.TFBertModel})

model = load_model_from_h5(os.getenv("MENTAL_MODEL_PATH"))  

def predict_mental_issue(text_input, top_n=2):
    mental_issue_mapping = {
        0: 'Depresi',
        1: 'Stress',
        2: 'Cemas',
    }
    cleaned_text = cleaning_text(text_input)
    translated_text = translate_text(cleaned_text)
    input_ids, attention_mask = preprocess_input(translated_text)
    prediction = model.predict([input_ids, attention_mask])
    

    # interpreter = tf.lite.Interpreter(model_path=os.getenv("MENTAL_MODEL_PATH"))
    # interpreter.allocate_tensors()
    # interpreter.get_input_details()

    # sample_input_index = np.expand_dims(input_ids[0], 0).astype(np.int32)
    # sample_masks_index = np.expand_dims(attention_mask[0], 0).astype(np.int32)

    # input_tensors = [sample_input_index, sample_masks_index]
    

    # bert_input_index = interpreter.get_input_details()[0]["index"]
    # bert_input_masks_index = interpreter.get_input_details()[1]["index"]
    # output_index = interpreter.get_output_details()[0]["index"]

    # interpreter.set_tensor(bert_input_index, sample_input_index)
    # interpreter.set_tensor(bert_input_masks_index, sample_masks_index)
    # interpreter.invoke()
    # prediction = interpreter.tensor(output_index)
    # # print(prediction()[0])

    # # input_index = interpreter.get_input_details()[0]["index"]
    # # output_index = interpreter.get_output_details()[0]["index"]

    # # interpreter.set_tensor(input_index, img)
    # # interpreter.invoke()
    # # prediction = []
    # prediction = (interpreter.get_tensor(output_index))

    # # predicted_label = np.argmax(prediction)
    # # class_names = ['rock', 'paper', 'scissors']

    # # return class_names[predicted_label]

    # # translated_text = translate_text(text_input)
    # # input_ids, attention_mask = preprocess_input(translated_text)
    # # predictions = model.predict([input_ids, attention_mask])

    top_n_indices = np.argsort(prediction, axis=1)[0, -top_n:][::-1] 
    top_n_mental_issue = [mental_issue_mapping[idx] for idx in top_n_indices]
    top_n_mental_issue
    
    return top_n_mental_issue