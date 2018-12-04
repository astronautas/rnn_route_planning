import numpy as np
from scipy.spatial.distance import cosine
import math
import geopy.distance

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def build_max_speeds(edges):
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

def get_ng_data(graph, node_id, goal_node_id):
    neighbours = graph.neighbors(node_id)
    ng_ids = []
    goal_node_coords_ne_format = (graph.node[goal_node_id]["y"], graph.node[goal_node_id]["x"])
    curr_node_neighbours_props = []

    for idx, curr_neighbour_name in enumerate(neighbours):
        ng_ids.append(curr_neighbour_name)

        neighbour_props = {}

        # Get curr neighbor coords
        curr_neighbour_coords_ne_format = (graph.node[curr_neighbour_name]["y"], graph.node[curr_neighbour_name]["x"])

        # Retrieve cheapest edge to curr NG
        # Min len edge - if there are multiple unequal cost (e.g. cost = min traversal time) edges between A -> B, always
        # traverse via the cheapest one
        edge_data = graph.get_edge_data(node_id, curr_neighbour_name)
        edge_data = min(edge_data.items(), key=lambda edge: edge[1]["best_travel_time"])[1] # there can be many edges, maybe take MIN(length)!

        curr_ng_props = graph.node[curr_neighbour_name]

        g_coords = (graph.node[goal_node_id]["x"], graph.node[goal_node_id]["y"])
        curr_coords = (graph.node[node_id]["x"], graph.node[node_id]["y"])
        curr_ng_coords = (graph.node[curr_neighbour_name]["x"], graph.node[curr_neighbour_name]["y"])
        
        neighbour_props["angle_to_goal"] = math.degrees(angle_between(np.subtract(curr_ng_coords, curr_coords), np.subtract(g_coords, curr_coords))) 

        neighbour_props["ne_coords"] = curr_neighbour_coords_ne_format
        neighbour_props["not_oneway"] = 1 if ("oneway" in edge_data) and (edge_data["oneway"] == False) else -1
        neighbour_props["is_highway"] = 1 if edge_data["highway"] in ["motorway", "trunk"] else 0
        neighbour_props["best_travel_time"] = edge_data["best_travel_time"]
        neighbour_props["dist_to_goal"] = geopy.distance.distance(curr_neighbour_coords_ne_format, goal_node_coords_ne_format).m

        # Encode properties
        curr_node_neighbours_props += [neighbour_props["angle_to_goal"], neighbour_props["best_travel_time"], neighbour_props["not_oneway"], neighbour_props["dist_to_goal"]]

    return np.array(curr_node_neighbours_props), ng_ids

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_route_duration(route, G):
    duration = 0
    prev_node = None

    for node in route:
        if prev_node:
            edge_data = G.get_edge_data(prev_node, node)
            edge_data = min(edge_data.items(), key=lambda edge: edge[1]["best_travel_time"])[1]
            duration += edge_data["best_travel_time"]
            
        prev_node = node

    return duration