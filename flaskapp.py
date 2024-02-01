import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = load_model('language_model.h5')  # Replace with your model file path

# Tokenizer for text preprocessing
tokenizer = Tokenizer()

# Load vocabulary and set tokenizer
with open('vocabulary.txt', 'r', encoding='utf-8') as vocab_file:
    vocabulary = vocab_file.read().splitlines()
tokenizer.word_index = {word: index + 1 for index, word in enumerate(vocabulary)}
tokenizer.index_word = {index + 1: word for index, word in enumerate(vocabulary)}

# Define parameters
sequence_length = 50

# Function to generate text
def generate_text(seed_text, next_words):
    generated_text = seed_text  # Initialize generated text with the seed text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length - 1, padding='pre')
        predicted = model.predict(token_list)
        predicted_word_index = np.argmax(predicted, axis=-1)
        predicted_word = tokenizer.index_word.get(predicted_word_index[0], '')
        generated_text += ' ' + predicted_word
        seed_text = generated_text  # Update the seed text for the next iteration
    return generated_text

# Route for the chatbot webpage
@app.route('/', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = generate_text(user_input, 10)  # Adjust the number of words to generate
        print(f"User Input: {user_input}")
        print(f"Bot Response: {response}")
        return render_template('index.html', user_input=user_input, response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
