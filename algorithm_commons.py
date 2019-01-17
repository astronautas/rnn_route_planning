import data_utils
import numpy as np

def predict_next_node_id(G, curr_node_id, goal_node_id, model, randomness=False):
    x, x0, ng_ids = data_utils.get_ng_data_formatted(G, curr_node_id, goal_node_id, 20)
    x = x[np.newaxis, ...][np.newaxis, ...]

    next_node_index = model.predict_classes(x, batch_size=1).tolist()[0][0]

    if len(ng_ids) >= next_node_index + 1:
        x0_list = x0.tolist()[next_node_index * 4 + 3]

        if randomness != False and np.random.rand() < randomness:
            return np.random.choice(ng_ids)
        else:
            return ng_ids[next_node_index]
    else:
        if len(ng_ids) > 0:
            return np.random.choice(ng_ids)
        else:
            return None