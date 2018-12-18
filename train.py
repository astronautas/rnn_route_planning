import glob
import json
from pathlib import Path
import numpy as np
from functools import reduce
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Activation, LSTM, Embedding, Flatten, TimeDistributed, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from IPython import embed
import math
from tensorflow.keras.optimizers import RMSprop, SGD, Adadelta
import tensorflow as tf

import numpy as np

from tensorflow.keras.utils import multi_gpu_model
from DataGenerator import DataGenerator

import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy

from collections import defaultdict

print(tf.__version__)

# Make sure both validation and original dataset are divisable by batches
def make_divisable_by_batches(batch_size, data):
    if batch_size == 1:
        return data

    remainder = len(data) % batch_size

    return data[:-remainder]

def find_shortest_episode(episodes):
    shortest = math.inf

    for episode in episodes:
        if len(episode[0]) < shortest:
            shortest = len(episode[0])

    return shortest

def subsample_v2(episodes):
    episodes_subs = []
    
    for episode in episodes:
        for i in range(2, len(episode[0])):
            episodes_subs.append([episode[0][-i:], episode[1][-i:]])

    return episodes_subs

def subsample(episodes, window_size):
    subepisode_groups = []
    
    for episode in episodes:
        subepisode_group = []

        # Subsample each episode steps
        # for i, x_sequence in enumerate(episode[0]):
        #     len_seq = len(episode[0])

        len_seq = len(episode[0])

        for window_start in range(0, len_seq - window_size + 1):
            window_end = window_start + window_size
            x_window = episode[0][window_start:window_end]
            y_window = episode[1][window_start:window_end]

            if len(x_window) == window_size: subepisode_group.append([x_window, y_window])

        subepisode_groups.append(subepisode_group)

    return subepisode_groups

def encode_labels(episodes):
    labels = {}

    for episode in episodes:
        for label in episode[1]:
            labels[label] = 1

    for episode in episodes:
        episode[1] = to_categorical(episode[1] , len(labels.keys())).tolist()

def add_padding_x(episodes):
    max_len_x = 0

    for episode in episodes:
        for train_x in episode[0]:
            max_len_x = len(train_x) if len(train_x) > max_len_x else max_len_x          

    for episode in episodes:
        episode[0] = pad_sequences(episode[0], maxlen=max_len_x, dtype='float').tolist()

def transform(episodes):
    transformed_episodes = []

    for episode in episodes:
        trans_episode = []
        
        train_data = list(map(
                            lambda step: reduce(list.__add__, 
                                map(
                                    lambda nbp: [
                                        nbp["angle_to_goal"], 
                                        nbp["best_travel_time"], 
                                        nbp["not_oneway"], 
                                        nbp["dist_to_goal"]], 
                                step["neighbour_props"])), 
                            episode["shortest_path"]))

        train_labels = np.array(list(map(lambda step: step["next_node_index"], episode["shortest_path"])))

        trans_episode = [train_data, train_labels]
        transformed_episodes.append(trans_episode)

    return transformed_episodes

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.iterations = 0

    def on_batch_end(self, batch, logs={}):
        print(logs)

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def bucketing(bucket_size, episodes, placeholder_timestep):
    episode_batches = chunks(episodes, bucket_size)

    # Pad each chunk
    for episode_batch in episode_batches:
        max_t = len(max(map(lambda e: e[1], episode_batch), key=len))

        for episode in episode_batch:
            to_append = max_t - len(episode[1])

            for _ in range(to_append):
                episode[0].append(placeholder_timestep[0])
                episode[1].append(placeholder_timestep[1])

    return episodes

def train(episodes, validation_episodes, model, batch_size):
    training_generator = DataGenerator(list(map(lambda ep: ep[0], episodes)), list(map(lambda ep: ep[1], episodes)), 
                                        batch_size=batch_size, shuffle=False)

    validation_generator = DataGenerator(list(map(lambda ep: ep[0], validation_episodes)), list(map(lambda ep: ep[1], validation_episodes)), 
                                        batch_size=batch_size, shuffle=False)

    model.fit_generator(generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=6, epochs=10)

def test(test_episodes, model, batch_size):
    generator = DataGenerator(list(map(lambda ep: ep[0], test_episodes)), list(map(lambda ep: ep[1], test_episodes)), 
                                        batch_size=batch_size, shuffle=False)

    metrics = model.evaluate_generator(generator=generator,
            use_multiprocessing=True,
            workers=6)

    print(metrics)

# <coords> : <count>
timestep_prediction_coordinates = defaultdict(lambda: 0.0)

# Custom loss, that discounts already visited nodes to escape loops
def categorical_crossentropy_discounted_loops(yTrue, yPred):
    res = categorical_crossentropy(yTrue, yPred)

    # with tf.get_default_session().as_default() as df:
    #     res_eval = res.eval()
    
    occ_tensor = tf.map_fn(lambda timesteps: tf.map_fn(lambda timestep_pred: 
                            timestep_prediction_coordinates[timestep_pred._id], timesteps), yPred)

    range = res.shape[1]

    # Add to each sample of a batch, the loss of a loop
    #occ_tensor = tf.zeros_like(res)
    # occ_tensor = tf.Variable([], tf.float32)
    return tf.add(res, occ_tensor)

#### MAIN ####
x = tf.Session().__enter__()

episodes = []

with open("episodes.json", "r") as file:
    episodes = json.load(file)

# Squash json episode timestep properties
episodes = transform(episodes)[0:20]

# FLATTEN
# episodes_partial = [item for sublist in episodes_partial for item in sublist]
episodes = subsample_v2(episodes)

# ENCODE LABELS
# Encode labels as vectors (1 if belongs to category, 0 if not, for all cats for each label)
encode_labels(episodes)

# ADD X PADDING
# (lets work with different size X, Y shapes)
add_padding_x(episodes)

# SHUFFLE
np.random.shuffle(episodes)

# TEST, TRAIN SPLIT
train_episodes, test_episodes = train_test_split(episodes, train_size=0.8)

# TRAIN, VAL SPLIT
train_episodes, validation_episodes = train_test_split(train_episodes, train_size=0.8)

# ORDER BY LENGTH
train_episodes = sorted(train_episodes, key=lambda ep: len(ep[1]))
test_episodes = sorted(test_episodes, key=lambda ep: len(ep[1]), reverse=True)
validation_episodes = sorted(validation_episodes, key=lambda ep: len(ep[1]), reverse=True)

# BUCKET
batch_size = 64
placeholder_x = [0] * len(episodes[0][0][0])
placeholder_y = [0] * len(episodes[0][1][0])

train_episodes = bucketing(bucket_size=batch_size, episodes=train_episodes, placeholder_timestep=[placeholder_x, placeholder_y])
test_episodes = bucketing(bucket_size=batch_size, episodes=test_episodes, placeholder_timestep=[placeholder_x, placeholder_y])
validation_episodes = bucketing(bucket_size=batch_size, episodes=validation_episodes, placeholder_timestep=[placeholder_x, placeholder_y])

model = Sequential()
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, batch_input_shape=(batch_size, None, len(placeholder_x)), 
                return_sequences=True, stateful=False))
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, stateful=False))
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, stateful=False))
model.add(TimeDistributed(Dense(len(placeholder_y), activation='softmax')))

opt = RMSprop(lr=0.0008, rho=0.9, epsilon=None, decay=0.0)

#model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

train(train_episodes, validation_episodes, model, batch_size)

test(test_episodes, model, batch_size)

# SAVE MODEL
model.save('model.h5')