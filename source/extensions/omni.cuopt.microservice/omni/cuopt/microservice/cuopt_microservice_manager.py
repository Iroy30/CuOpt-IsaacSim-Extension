# Copyright (c) 2021-2023, NVIDIA CORPORATION.

import requests


class cuOptRunner:
    def __init__(self, cuopt_url: str):
        """
        Note that a cuOpt server at a single url manages one problem at a time
        Initializing another instance of cuOptRunner at the same url will clear
        optimization data currently set on
        """
        self.cuopt_url = cuopt_url
        self.data_parameters = {"return_data_state": False}

        requests.delete(cuopt_url + "clear_optimization_data")
        print(f"\n - OPTIMIZATION DATA AT {cuopt_url} HAS BEEN CLEARED - \n")

    def get_routes(self, cuopt_problem_data):
        solver_response = requests.post(self.cuopt_url + "get_routes", json=cuopt_problem_data)
        print(f"SOLVER RESPONSE: {solver_response.json()}\n")

        return solver_response.json()["response"]["solver_response"]
