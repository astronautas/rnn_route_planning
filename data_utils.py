from tensorflow.keras.preprocessing.sequence import pad_sequences

import osmnx_utils
import numpy as np

def make_single_step_data(G, curr_node_id, choice_id, goal_node_id):
    x_padding_len = 20
    y_padding_len = 5

    # Make example data
    x0, ng_ids = osmnx_utils.get_ng_data(G, curr_node_id, goal_node_id)
    x = pad_sequences([x0], maxlen=x_padding_len, dtype='float', value=50000.0)[0].tolist()
    #x = x[np.newaxis, ...][np.newaxis, ...]

    # Make label
    y = [0] * y_padding_len
    y[ng_ids.index(choice_id)] = 1

    return x, y