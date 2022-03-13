import cartlib

import os
import random

import nltk
import numpy as np
import tensorflow as tf

## DEFAULTS
INTENT_JSON=os.path.join(os.getcwd(), "data", "intent.json")
MODEL_CORE=os.path.join(os.getcwd(), "models", "saved", "core")

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [cartlib.lemmatizer.lemmatize(word) for word in tokens]
  return tokens


def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(model, text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result


# running the chatbot
if __name__ == "__main__":
    data = cartlib.intent_json(INTENT_JSON)
    (words, classes, doc_X, doc_y) = cartlib.intent_lists(data)
    cartman_model = tf.keras.models.load_model(MODEL_CORE)
    cartman_model.summary()
    print("\n~*~ knock knock...")
    while True:
        message = input("")
        intents = pred_class(cartman_model, message, words, classes)
        result = get_response(intents, data)
        print(result)

