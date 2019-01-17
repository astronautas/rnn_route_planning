import algorithm_commons
import collections
import networkx as nx
import osmnx_utils
import numpy as np

def run_optimality_evaluation(G, model, randomness):
    model.reset_states()
    
    try:
        start_node_id = np.random.choice(G.nodes)
        goal_node_id = np.random.choice(G.nodes)

        try:
            truth_route = nx.shortest_path(G, start_node_id, goal_node_id, weight='best_travel_time')
        except nx.NetworkXNoPath:
            print("[ERROR] No path found")
            return None, None

        curr_node_id = start_node_id
        ml_route = []
        visited = collections.deque(maxlen=7)

        # Start node
        ml_route.append(curr_node_id)
        use_random_actions = False

        gt_duration = osmnx_utils.get_route_duration(truth_route, G)
        history = collections.deque(maxlen=5)

        while curr_node_id != goal_node_id:
            curr_node_id = algorithm_commons.predict_next_node_id(G, curr_node_id, goal_node_id, model, randomness=randomness)

            if curr_node_id:
                ml_route.append(curr_node_id)

            if osmnx_utils.get_route_duration(ml_route, G) > gt_duration * 1.5:
                print("x2 length, try again...")
                return None, None

        ml_duration = osmnx_utils.get_route_duration(ml_route, G)

        # print("------ METRICS ------")
        # print("GT duration: ", gt_duration, ". ML duration: ", ml_duration, ". Abs Diff: ", ml_duration - gt_duration, ". Ratio: ", gt_duration / ml_duration)
        # print("---------------------")

        # ##### PLOT GT AND ML PATHS ######
        # ox.plot_graph_route(G, truth_route)
        # ox.plot_graph_route(G, ml_route)
        
        if gt_duration == 0 or ml_duration == 0:
            return None, None
        else:
            return ml_duration, gt_duration

    except nx.NetworkXError:
        return None, None