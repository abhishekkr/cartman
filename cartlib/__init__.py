import json
import string

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()


def intent_json(jsonpath):
    with open(jsonpath, 'r') as fyl:
        fyldata=fyl.read()
    return json.loads(fyldata) 


def intent_lists(data):
# Each list to create
    words = []
    classes = []
    doc_X = []
    doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_y.append(intent["tag"])

        # add the tag to the classes if it's not there already
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
    words = sorted(set(words))
    classes = sorted(set(classes))
    return (words, classes, doc_X, doc_y)
