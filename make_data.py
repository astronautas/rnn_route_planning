import osmnx as ox
import networkx as nx
import numpy as np
import json
import datetime
import geopy.distance
from pathlib import Path
from scipy.spatial.distance import cosine
import math

from osmnx_utils import isfloat

speeds = {
    "residential": 50,
    "secondary": 90,
    "primary": 90,
    "motorway": 120,
    "motorway_link": 120,
    "trunk": 110,
    "tertiary": 90,
    "default": 70
}

road_types = {
    "residential": 0,
    "secondary": 1,
    "primary": 2,
    "motorway": 3,
    "default": 6,
    "trunk": 4,
    "tertiary": 5,
}

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def build_max_speeds(edges):
    for edge in edges:
        edge_data = edge[-1]

        if not("maxspeed" in edge_data):
            road_type = edge_data["highway"][0] # TODO: there can be many types, fix this

            if road_type in speeds:
                edge_data["maxspeed"] = speeds[road_type]
            else:
                edge_data["maxspeed"] = speeds["default"]
        else:
            if isinstance(edge_data["maxspeed"], list):
                edge_data["maxspeed"] = np.mean(list(map(lambda speed: float(speed) if isfloat(speed) else 0, edge_data["maxspeed"])))
            else:
                if isfloat(edge_data["maxspeed"]):
                    edge_data["maxspeed"] = float(edge_data["maxspeed"])
                else:    
                    edge_data["maxspeed"] = speeds[edge_data["highway"][0]] if isinstance(edge_data["maxspeed"], list) else speeds[edge_data["highway"]]

def add_time_to_roads(edges):
    for edge in edges:
        # Speed in km/h, distance - in m, needs regularization
        edge[-1]["best_travel_time"] = float(edge[-1]["length"]) / (float(edge[-1]["maxspeed"]) / 3.6)

def build_shortest_path(graph, shortest_path, goal_node_id):
    steps_data = []
    goal_node_coords_ne_format = (graph.node[goal_node_id]["y"], graph.node[goal_node_id]["x"])

    for idx, val in enumerate(shortest_path):
        curr_node_name = shortest_path[idx]

        step_data = {}
        curr_node_neighbours_props = []

        next_node_name = shortest_path[idx+1] if idx+1 < len(shortest_path) else None

        if next_node_name != None:

            neighbours = graph.neighbors(curr_node_name)

            for idx, curr_neighbour_name in enumerate(neighbours):
                if curr_neighbour_name == next_node_name: step_data["next_node_index"] = idx

                neighbour_props = {}

                curr_neighbour_coords_ne_format = (graph.node[curr_neighbour_name]["y"], graph.node[curr_neighbour_name]["x"])

                edge_data = graph.get_edge_data(curr_node_name, curr_neighbour_name)[0] # there can be many edges, maybe take MIN(length)??

                curr_ng_props = graph.node[curr_neighbour_name]

                g_coords = (graph.node[goal_node_id]["x"], graph.node[goal_node_id]["y"])
                curr_coords = (graph.node[curr_node_name]["x"], graph.node[curr_node_name]["y"])
                curr_ng_coords = (graph.node[curr_neighbour_name]["x"], graph.node[curr_neighbour_name]["y"])

                angle_to_g = math.degrees(angle_between(np.subtract(curr_ng_coords, curr_coords), np.subtract(g_coords, curr_coords))) 
    
                if math.isnan(angle_to_g):
                    continue

                neighbour_props["angle_to_goal"] = angle_to_g

                neighbour_props["is_highway"] = 1 if edge_data["highway"] in ["motorway", "trunk"] else 0
                neighbour_props["best_travel_time"] = edge_data["best_travel_time"]
                neighbour_props["not_oneway"] = 1 if ("oneway" in edge_data) and (edge_data["oneway"] == False) else -1
                neighbour_props["dist_to_goal"] = geopy.distance.distance(curr_neighbour_coords_ne_format, goal_node_coords_ne_format).m

                curr_node_neighbours_props.append(neighbour_props)
            
            step_data["neighbour_props"] = curr_node_neighbours_props
            steps_data.append(step_data)

    return steps_data

def build_one_episode_in_env(G):
    episode = {}

    start_node_id = np.random.choice(G.nodes)
    goal_node_id = np.random.choice(G.nodes)

    # Add the speed feature to edges (we're gonna by time later)
    build_max_speeds(G.edges(data=True))

    # Add time to edge
    add_time_to_roads(G.edges(data=True))

    data = G.edges(data=True)

    print("Generating shortest path...")

    # Generate ground truth shortest path (Dijkstra)
    try:
        route = nx.shortest_path(G, start_node_id, goal_node_id, weight='best_travel_time')
        #ox.plot_graph_route(G, route)
    except nx.NetworkXNoPath:
        print("[ERROR] No path found")
        return None

    print("Building the data file...")

    episode["goal"] = {}
    episode["goal"]["x"] = G.node[goal_node_id]["x"]
    episode["goal"]["y"] = G.node[goal_node_id]["y"]

    episode["start"] = {}
    episode["start"]["x"] = G.node[start_node_id]["x"]
    episode["start"]["y"] = G.node[start_node_id]["y"]

    episode["shortest_path"] = build_shortest_path(G, route, goal_node_id)

    return episode


envs = [(55.917953, 21.066187, 500), (55.727253, 24.362339, 3000), (54.690272, 25.261696, 8000), (55.693357, 21.142402, 15000), (55.916457, 21.424276, 20000)]
episodes = []

for env in envs:
    name = f'{env[0]}, {env[1]}, {env[2]}.graphml'

    if Path('data/', name).is_file():
        print("Pulling data from file...")
        G = ox.load_graphml(name)
    else:
        print("Pulling data from OSM...")
        G = ox.graph_from_point((env[0], env[1]), distance=env[2], network_type='drive')
        ox.save_graphml(G, filename=name)

    for i in range(0, 200):
        print("Generating episode: ", i)
        episode = build_one_episode_in_env(G)
        if episode != None: episodes.append(episode)

# Shuffling reduces overfit
np.random.shuffle(episodes)

# Save data
episodes_json = json.dumps(episodes)

now = datetime.datetime.now()
with open("episodes.json", "w") as file:
    file.write(episodes_json)
    file.close()
