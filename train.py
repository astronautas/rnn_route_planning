import glob
import json
from pathlib import Path
import numpy as np
from functools import reduce
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Activation, LSTM, Embedding, Flatten, TimeDistributed
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

FEATURES = 2
MAX_NEIGHBORS = 4

def encode_labels(episodes):
    for episode in episodes:
        labels = episode[1]
        episode[1] = to_categorical(labels)
    
    return episodes

def add_padding(episodes):
    max_len_x = 0
    max_len_y = 0

    for episode in episodes:
        for train_sample in episode[0]:
            if len(train_sample) > max_len_x:
                max_len_x = len(train_sample)

        for label in episode[1]:
            if len(label) > max_len_y:
                max_len_y = len(label) 
                
    # Add padding
    for episode in episodes:
        episode[0] = pad_sequences(episode[0], maxlen=max_len_x)
        episode[1] = pad_sequences(episode[1], maxlen=max_len_y)
    
    return episodes

def transform(episodes):
    transformed_episodes = []

    for episode in episodes:
        trans_episode = []
        
        train_data = list(map(
                            lambda step: reduce(list.__add__, 
                                map(
                                    lambda nbp: [nbp["best_travel_time"], nbp["length"], nbp["dist_to_goal"]], 
                                step["neighbour_props"])), 
                            episode["shortest_path"]))

        train_labels = np.array(list(map(lambda step: step["next_node_index"], episode["shortest_path"])))

        trans_episode = [train_data, train_labels]
        transformed_episodes.append(trans_episode)

    return transformed_episodes

class PrintDot(Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def train(episodes, model):
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    for episode in episodes:
        for idx in range(0, len(episode[0])):
            ex = episode[0][idx]
            label = np.array(episode[1][idx], ndmin=3)

            x = ex[np.newaxis, ...][np.newaxis, ...]

            model.fit(x, label, epochs=1, batch_size=1,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

        model.reset_states()

episodes = []

with open("episodes.json", "r") as file:
    episodes = json.load(file)

transformed_episodes = transform(episodes)

# Encode labels as vectors (1 if belongs to category, 0 if not, for all cats for each label)
print("[LOG] Encoding labels...")
transformed_episodes = encode_labels(transformed_episodes)

# Add padding (lets work with different size X, Y shapes)
print("[LOG] Adding padding...")
transformed_episodes = add_padding(transformed_episodes)

input_shape = transformed_episodes[0][0].shape[1]

n_batch = 1

model = Sequential()
model.add(LSTM(units=1000, dropout=0.7, recurrent_dropout=0.7, batch_input_shape=(1, 1, input_shape), return_sequences=True, stateful=True))
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(transformed_episodes[0][1].shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.summary()

train_episodes, test_episodes = train_test_split(transformed_episodes, train_size=0.7)

print("[LOG] We have train episodes count: ", len(train_episodes))

# Do training
print("[LOG] Training...")
train(train_episodes, model)

ex = test_episodes[0][0][0]
label = np.array(test_episodes[0][1][0], ndmin=3)
x = ex[np.newaxis, ...][np.newaxis, ...]

total_score = 0
total = 0 
for episode in test_episodes:
    for idx in range(0, len(episode[0])):
        ex = episode[0][idx]
        label = np.array(episode[1][idx], ndmin=3)

        x = ex[np.newaxis, ...][np.newaxis, ...]
        #y = label[..., np.newaxis][..., np.newaxis]

        print("Predicted: ", model.predict(x, batch_size=1, verbose=0), ". Actual: ", label)

        total_score += model.evaluate(x, label, batch_size=1, verbose=0)[1]
        total += 1

    model.reset_states()

print(float(total_score) / float(total))