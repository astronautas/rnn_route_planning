import osmnx as ox
import networkx as nx
import numpy as np
import json
import datetime
import geopy.distance
from pathlib import Path
from scipy.spatial.distance import cosine
import math

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from osmnx_utils import add_time_to_roads
from osmnx_utils import build_max_speeds
from osmnx_utils import get_ng_data
from osmnx_utils import get_route_duration

from tensorflow.keras.layers import Dense, Activation, LSTM, Embedding, Flatten, TimeDistributed, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import collections
import time
from threading import Thread
import random

#from osmnx_utils import append_node_to_route

# RE-BUILD A MODEL
# Use diff. batch size than when training
old_model = load_model("models/model_5.h5")

old_weights = old_model.get_weights()

model = Sequential()
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, batch_input_shape=(1, None, 20), 
                return_sequences=True, stateful=True))
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, stateful=True))
model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(5, activation='softmax')))

opt = RMSprop(lr=0.0008, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()

model.set_weights(old_weights)

###### BUIILD A GRAPH ######
env = (55.917953, 21.066187, 8000)
name = f'{env[0]}, {env[1]}, {env[2]}.graphml'

if Path('data/', name).is_file():
    print("Pulling data from file...")
    G = ox.load_graphml(name)
else:
    print("Pulling data from OSM...")
    G = ox.graph_from_point((env[0], env[1]), distance=env[2], network_type='drive')
    ox.save_graphml(G, filename=name)

start_node_id = np.random.choice(G.nodes)
goal_node_id = np.random.choice(G.nodes)

# Add the speed feature to edges (we're gonna by time later)
build_max_speeds(G.edges(data=True))

# Add time to edge
add_time_to_roads(G.edges(data=True))

###### EVALUATION ######
truth_route = nx.shortest_path(G, start_node_id, goal_node_id, weight='best_travel_time')

curr_node_id = start_node_id
ml_route = []
visited = collections.deque(maxlen=7)

# Start node
ml_route.append(curr_node_id)
use_random_actions = False

while curr_node_id != goal_node_id:
    x0, ng_ids = get_ng_data(G, curr_node_id, goal_node_id)
    x = pad_sequences([x0], maxlen=20, dtype='float')[0]
    x = x[np.newaxis, ...][np.newaxis, ...]


    prediction = model.predict(x, batch_size=1)
    next_node_index = model.predict_classes(x, batch_size=1).tolist()[0][0]

    integer = random.randint(0, 100)

    if use_random_actions:
        print("Random")
        curr_node_id = np.random.choice(ng_ids)
    else:
        if len(ng_ids) <= next_node_index:
            curr_node_id = visited[-1]
            use_random_actions = True
        else:
            curr_node_id = ng_ids[next_node_index]
            x0_list = x0.tolist()[next_node_index * 4 + 3]
            print("Dist to goal: ", x0_list)


    visited.append(curr_node_id)
    
    if visited.count(curr_node_id) >= 4:
        use_random_actions = True
    else:
        use_random_actions = False

    ml_route.append(curr_node_id)

### TIME DIVERGENCE ###
gt_duration = get_route_duration(truth_route, G)
ml_duration = get_route_duration(ml_route, G)

print("------ METRICS ------")
print("GT duration: ", gt_duration, ". ML duration: ", ml_duration, ". Abs Diff: ", ml_duration - gt_duration, ". Ratio: ", gt_duration / ml_duration)
print("---------------------")

###### PLOT GT AND ML PATHS ######
ox.plot_graph_route(G, truth_route)
ox.plot_graph_route(G, ml_route)

input()