import argparse
import collections
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from threading import Thread
from functools import reduce

import geopy.distance
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

import osmnx as ox
from osmnx_utils import (add_time_to_roads, build_max_speeds, get_ng_data,
                         get_route_duration, random_network_modification)
import utils
from tensorflow.keras.layers import (GRU, LSTM, Activation, Dense, Embedding,
                                     Flatten, TimeDistributed)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences

import model_commons

from tensorflow.keras.utils import get_custom_objects

import tensorflow as tf
import tensorflow.keras.backend as K

# RE-BUILD A MODEL
# Use different batch size than when training
get_custom_objects().update({'softmax_with_temp': model_commons.softmax_with_temp})

old_model = load_model("model.h5")
old_weights = old_model.get_weights()

model = model_commons.get_model(batch_size=1, timesteps=None, X_dim=20, Y_dim=5, activation=model_commons.softmax_with_temp)
model.summary()

model.set_weights(old_weights)

### MODEL NAVIGATION WRAPPER ###
def navigation_wrapper(G, start_node_id, goal_node_id, stop_after_steps):
    curr_node_id = start_node_id
    ml_route = []
    visited = collections.deque(maxlen=7)

    # Start node
    ml_route.append(curr_node_id)
    use_random_actions = False

    while curr_node_id != goal_node_id:
        if stop_after_steps and len(ml_route) > stop_after_steps:
            return False

        # if stop_predicate()
        x0, ng_ids = get_ng_data(G, curr_node_id, goal_node_id)
        x = pad_sequences([x0], maxlen=20, dtype='float')[0]
        x = x[np.newaxis, ...][np.newaxis, ...]

        # prediction = model.predict(x, batch_size=1)
        next_node_index = model.predict_classes(x, batch_size=1).tolist()[0][0]

        integer = random.randint(0, 100)

        if use_random_actions:
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

    return ml_route

def predict_next_node_id(G, curr_node_id, goal_node_id, model, randomness=False):
    x0, ng_ids = get_ng_data(G, curr_node_id, goal_node_id)
    x = pad_sequences([x0], maxlen=20, dtype='float')[0]
    x = x[np.newaxis, ...][np.newaxis, ...]

    next_node_index = model.predict_classes(x, batch_size=1).tolist()[0][0]

    if len(ng_ids) >= next_node_index + 1:
        x0_list = x0.tolist()[next_node_index * 4 + 3]
        print("Dist to goal: ", x0_list)

        if randomness != False and np.random.rand() < randomness:
            return np.random.choice(ng_ids)
        else:
            return ng_ids[next_node_index]
    else:
        return np.random.choice(ng_ids)

### DYNAMIC NETWORK CHANGES ###
# Test how well model follows the optimal path
# under road network modification
def run_generalization_evaluation(G, model):
    experiment_count = 50

    experiment_node_pairs = [(np.random.choice(G.nodes), np.random.choice(G.nodes)) for i in range(1, experiment_count)]

    # Eval optimality and arrival rate to later check how
    # dynamic network changes affect these
    # optimalities = []

    # for start_goal_pair in experiment_node_pairs:
    #     ml_route = []
    #     optimal_route = []
    #     curr_node_id = start_goal_pair[0]
    #     goal_node_id = start_goal_pair[1]
    #     before_modification_node_id = curr_node_id

    #     while curr_node_id != start_goal_pair[1]:
    #         for i in range(0, 5):
    #             curr_node_id = predict_next_node_id(G, curr_node_id, goal_node_id, model)
    #             ml_route.append(curr_node_id)

    #         optimal_route = utils.split_list_at_value(nx.shortest_path(G, before_modification_node_id, 
    #                                                     goal_node_id, weight='best_travel_time'), curr_node_id)

    #         optimalities.append(get_route_duration(ml_route, G) / get_route_duration(optimal_route, G))
    #         ml_route = []
    #         optimal_route = []
    #         before_modification_node_id = curr_node_id
    #         # Modify graph

    # print("Optimality rate (dynamic): ", reduce(lambda a,b: a + b, optimalities) / len(optimalities))

    optimalities_static = []
    
    for start_goal_pair in experiment_node_pairs:
        ml_route = []
        optimal_route = []
        curr_node_id = start_goal_pair[0]
        goal_node_id = start_goal_pair[1]
        before_modification_node_id = curr_node_id

        while curr_node_id != start_goal_pair[1]:
            curr_node_id = predict_next_node_id(G, curr_node_id, goal_node_id, model)
            ml_route.append(curr_node_id)

        optimal_route = nx.shortest_path(G, start_goal_pair[0], goal_node_id, weight='best_travel_time')
        optimalities_static.append(get_route_duration(ml_route, G) / get_route_duration(optimal_route, G))

        random_network_modification(G)

    print("Optimality rate (dynamic): ", reduce(lambda a,b: a + b, optimalities_static) / len(optimalities_static))

    optimalities_static = []
    
    for start_goal_pair in experiment_node_pairs:
        ml_route = []
        optimal_route = []
        curr_node_id = start_goal_pair[0]
        goal_node_id = start_goal_pair[1]
        before_modification_node_id = curr_node_id

        while curr_node_id != start_goal_pair[1]:
            curr_node_id = predict_next_node_id(G, curr_node_id, goal_node_id, model)
            ml_route.append(curr_node_id)

        optimal_route = nx.shortest_path(G, start_goal_pair[0], goal_node_id, weight='best_travel_time')
        optimalities_static.append(get_route_duration(ml_route, G) / get_route_duration(optimal_route, G))

    print("Optimality rate (static): ", reduce(lambda a,b: a + b, optimalities_static) / len(optimalities_static))
    
### DIVERGION FROM OPTIMAL PATH RATE ###
def run_arrival_rate_evaluation(G):
    arrived = 0.0
    total = 0.0

    for i in range(0, 50):
        start_node_id = np.random.choice(G.nodes)
        goal_node_id = np.random.choice(G.nodes)

        truth_route = nx.shortest_path(G, start_node_id, goal_node_id, weight='best_travel_time')
        truth_route_steps = len(truth_route)
        
        predicted_route = navigation_wrapper(G, start_node_id, goal_node_id, truth_route_steps)
        total += 1.0

        if predicted_route:
            arrived += 1.0
        
        print("[LOG] Rate: ", arrived / total)

    return arrived / total    


### TIME DIVERGENCE ###
def run_optimality_evaluation(G):
    start_node_id = np.random.choice(G.nodes)
    goal_node_id = np.random.choice(G.nodes)

    truth_route = nx.shortest_path(G, start_node_id, goal_node_id, weight='best_travel_time')

    curr_node_id = start_node_id
    ml_route = []
    visited = collections.deque(maxlen=7)

    # Start node
    ml_route.append(curr_node_id)
    use_random_actions = False

    while curr_node_id != goal_node_id:
        curr_node_id = predict_next_node_id(G, curr_node_id, goal_node_id, model, randomness=0.2)
        ml_route.append(curr_node_id)    

    gt_duration = get_route_duration(truth_route, G)
    ml_duration = get_route_duration(ml_route, G)

    print("------ METRICS ------")
    print("GT duration: ", gt_duration, ". ML duration: ", ml_duration, ". Abs Diff: ", ml_duration - gt_duration, ". Ratio: ", gt_duration / ml_duration)
    print("---------------------")

    ###### PLOT GT AND ML PATHS ######
    ox.plot_graph_route(G, truth_route)
    ox.plot_graph_route(G, ml_route)

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='RNN route planning test.')

    parser.add_argument('--dynamic_network',
                    help='Run optimality change experiment under road network changes experiment (true/false)')

    parser.add_argument('--arrival_rate',
                    help='Run arrival rate experiment (true/false)')

    parser.add_argument('--optimality',
                    help='Run time optimality experiment (true/false)')
                    
    return parser

def prep_road_network():
    ###### BUIILD A GRAPH ######
    env = (55.916457, 21.424276, 5000)
    name = f'{env[0]}, {env[1]}, {env[2]}.graphml'

    if Path('data/', name).is_file():
        print("Pulling data from file...")
        G = ox.load_graphml(name)
    else:
        print("Pulling data from OSM...")
        G = ox.graph_from_point((env[0], env[1]), distance=env[2], network_type='drive')
        ox.save_graphml(G, filename=name)

    # Add the speed feature to edges (we're gonna by time later)
    build_max_speeds(G.edges(data=True))

    # Add time to edge
    add_time_to_roads(G.edges(data=True))

    return G

if __name__ == "__main__":
    G = prep_road_network()

    ### EXECUTION PART ###
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    if parsed_args.optimality == "true":
        print("[LOG] Running optimality...")
        run_optimality_evaluation(G)

    if parsed_args.arrival_rate == "true":
        run_arrival_rate_evaluation(G)

    if parsed_args.dynamic_network == "true":
        run_generalization_evaluation(G, model)


    input()