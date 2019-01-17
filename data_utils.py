from tensorflow.keras.preprocessing.sequence import pad_sequences

import osmnx_utils
import numpy as np

def make_single_step_data(G, curr_node_id, choice_id, goal_node_id):
    x_padding_len = 20
    y_padding_len = 5

    # Make example data
    x, x0, ng_ids = get_ng_data_formatted(G, curr_node_id, goal_node_id, x_padding_len)
    x = x.tolist()

    #x = x[np.newaxis, ...][np.newaxis, ...]

    # Make label
    y = [0] * y_padding_len
    y[ng_ids.index(choice_id)] = 1

    return x, y

def get_ng_data_formatted(G, curr_node_id, goal_node_id, max_padding_len):
    x0, ng_ids = osmnx_utils.get_ng_data(G, curr_node_id, goal_node_id)
    x = pad_sequences([x0], padding='post', maxlen=max_padding_len, dtype='float', value=0.0)[0]

    return x, x0, ng_ids