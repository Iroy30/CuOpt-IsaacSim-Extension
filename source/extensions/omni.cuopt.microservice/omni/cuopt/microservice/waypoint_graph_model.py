# Copyright (c) 2021-2023, NVIDIA CORPORATION.

from .common import read_json


class WaypointGraphModel:
    def __init__(self):

        self.nodes = []
        self.offsets = []
        self.edges = []
        self.weights = []

        self.node_count = 0
        self.edge_count = 0

        self.node_path_map = {}
        self.path_node_map = {}
        self.node_edge_map = {}
        self.edge_path_map = {}
        self.path_edge_map = {}


def load_waypoint_graph_from_file(stage, waypoint_graph_json):

    model = WaypointGraphModel()

    waypoint_graph_data = read_json(waypoint_graph_json)

    model.nodes = waypoint_graph_data["node_locations"]

    graph = waypoint_graph_data["graph"]

    # Convert the graph to CSR and save it to the graph model
    offsets = []
    edges = []
    cur_offset = 0
    offset_node_lookup = {}
    ordered_keys = sorted([int(x) for x in graph.keys()])
    for i, node in enumerate(ordered_keys):
        offsets.append(cur_offset)
        cur_offset += len(graph[str(node)]["edges"])

        edges = edges + graph[str(node)]["edges"]
        offset_node_lookup[i] = node

    offsets.append(cur_offset)

    model.offsets = offsets
    model.edges = edges
    model.offset_node_lookup = offset_node_lookup

    return model
