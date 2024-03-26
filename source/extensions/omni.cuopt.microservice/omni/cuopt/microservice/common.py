# Copyright (c) 2021-2023, NVIDIA CORPORATION.

import json
import requests
from .cuopt_thin_client import CuOptServiceClient

def read_json(json_file_path):

    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    return json_data


def show_vehicle_routes(routes):
    message = f"Solution found using {routes['num_vehicles']} vehicles \nSolution cost: {routes['solution_cost']} \n\n"
    for v_id, data in routes["vehicle_data"].items():
        message = message + "For vehicle -" + str(v_id) + " route is: \n"
        path = ""
        route_ids = data["route"]
        for index, route_id in enumerate(route_ids):
            path += str(route_id)
            if index != (len(route_ids) - 1):
                path += "-> "
        message = message + path + "\n\n"
    return message


def test_connection_microservice(ip, port):

    cuopt_url = f"http://{ip}:{port}/cuopt/"

    cuopt_status_info = f"working"

    try:
        cuopt_response = requests.get(cuopt_url + "health")
        if cuopt_response.status_code == 200:
            cuopt_status_info = "SUCCESS: cuOpt Microservice is Running"
        else:
            cuopt_status_info = (
                "FAILURE: cuOpt Microservice found but not running correctly"
            )

    except:
        cuopt_status_info = (
            f"FAILURE: cuOpt Microservice was not found running at {cuopt_url}"
        )
    return cuopt_status_info


def test_connection_managed_service(auth, function_name, function_id):
    print(auth, function_name, function_id)
    try:
        client = CuOptServiceClient(
            client_id = auth['id'],
            client_secret = auth['secret'],
            sak = auth['sak'],
            function_name = function_name,
            function_id = function_id
        )
        cuopt_status_info = "SUCCESS: cuOpt Managed Service is Accessible"
    except:
        client = None
        cuopt_status_info = (
            f"FAILURE: cuOpt Managed Service is not Accessible"
        )
    return cuopt_status_info, client

