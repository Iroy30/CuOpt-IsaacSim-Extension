# Copyright (c) 2021-2023, NVIDIA CORPORATION.

import json


class TransportOrders:
    def __init__(self):
        self.order_xyz_locations = None
        self.graph_locations = None
        self.order_demand = None
        self.order_time_windows = None
        self.order_service_times = None

        self.order_waypoint_material = None
        self.order_node_scale = [0.6, 0.6, 0.15]
        self.order_waypoint_color = [0.05, 0.5, 0.1]
        self.order_waypoint_intensity = 5000.0

    # Load Task info from json data
    def load_sample(self, orders_json):

        with open(orders_json) as orders_file:
            orders_data = json.load(orders_file)

        self.order_xyz_locations = orders_data["task_locations"]
        self.order_demand = orders_data["demand"]

        if "task_time_windows" in orders_data:
            self.order_time_windows = orders_data["task_time_windows"]
            self.order_service_times = orders_data["service_times"]

        self.graph_locations = orders_data["task_locations"]
