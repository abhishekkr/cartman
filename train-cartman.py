import cartlib

import random
import os

import nltk
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

## DEFAULTS
INTENT_JSON=os.path.join(os.getcwd(), "data", "intent.json")
CHECKPOINT_DIR=os.path.join(os.getcwd(), "models", "checkpoint")
CHECKPOINT_CORE=os.path.join(CHECKPOINT_DIR, "core-{epoch:04d}.ckpt")
MODEL_CORE=os.path.join(os.getcwd(), "models", "saved", "core")


core_model_dir = os.path.dirname(MODEL_CORE)
if not os.path.exists(core_model_dir):
    os.makedirs_p(core_model_dir)


# load intent data
data = cartlib.intent_json(INTENT_JSON)

(words, classes, doc_X, doc_y) = cartlib.intent_lists(data)

# list for training data
training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = cartlib.lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

# the deep learning model
def create_model(checkpoint_callback, batch_size):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation = "softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=["accuracy"])
    print(model.summary())
    model.fit(x=train_X,
              y=train_y,
              epochs=200,
              batch_size=batch_size,
              callbacks=[checkpoint_callback],
              verbose=1)
    return model

cp_batch_size   = 32
cp_callback     = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_CORE,
                    save_weights_only=True,
                    save_freq=5*cp_batch_size,
                    verbose=1)

core_model = create_model(cp_callback, cp_batch_size)
core_model.save(MODEL_CORE)

