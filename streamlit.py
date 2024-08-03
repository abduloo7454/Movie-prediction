import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model
import pickle

with open('simple_rnn_imdb.pkl', 'rb') as pickle_in:
  model = pickle.load(pickle_in)

# Step 2: Helper Functions
# Function to decode reviews
def decode_revief(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

## Mapping of words index back to words(for understanding)
word_index=imdb.get_word_index()
word_index

# Functionn to preprocess user input
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
  return padded_review

### Prediction function

def predict_sentiments(review):
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)
  sentiments = 'Positive' if prediction > 0.5 else 'Negative'
  return sentiments, prediction[0][0]

# Step 4: user Input and prediction
# Example review for prediction

example_review = "This movie was fantastic! The acting was great and the plot was thrilling"

sentiments, score = predict_sentiments(example_review)
print(f"Sentiment: {sentiments}, Score: {score}")

import streamlit as st
##streamlit app

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Enter a movie review:')

# Prediction button
if st.button('Predict Sentiment'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiments = 'Positive' if prediction > 0.5 else 'Negative'
    
    # Display the result
    st.write(f"Sentiment: {sentiments}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please enter a movie review.')
