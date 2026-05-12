import random
import json
import pickle
import numpy as np
import nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide tensorflow spam

from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras

lemmatizer = WordNetLemmatizer()

# Load files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
   
    
    if not results:
        return []
        
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    

    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Try asking something else."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "I didn't get that."

print("Chatbot is running! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print("Bot: Bye!")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)