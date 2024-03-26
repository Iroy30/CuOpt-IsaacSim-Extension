# Copyright (c) 2021-2023, NVIDIA CORPORATION.

import omni.ext
import omni.ui as ui
from pxr import Gf
import requests

from omni.cuopt.microservice.cuopt_microservice_manager import cuOptRunner
from omni.cuopt.microservice.common import (
    show_vehicle_routes,
    test_connection_microservice,
    test_connection_managed_service,
)

from omni.kit.menu.utils import (
    add_menu_items,
    remove_menu_items,
    MenuItemDescription,
)
import omni.isaac.debug_draw as debug_draw
from omni.isaac.ui.ui_utils import (
    setup_ui_headers,
    get_style,
    btn_builder,
    str_builder,
    int_builder,
    float_builder,
)

from scipy.spatial.distance import pdist, squareform
import requests as req
import weakref
import random
import gc


EXTENSION_NAME = "Simple Cost Matrix"

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class cuOptSampleExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        self._ext_id = ext_id

        self._window = None
        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()

        self._cuopt_ip_prompt = "Enter IP"
        self._cuopt_port_prompt = "Enter Port"
        self._cuopt_id_prompt = "Enter ID"
        self._cuopt_secret_prompt = "Enter Secret"
        self._cuopt_sak_prompt = "Enter SAK"
        self._function_name_prompt = ""
        self._function_id_prompt = ""
        self._cuopt_ip = "Enter IP"
        self._cuopt_port = "Enter Port"
        self._cuopt_id = "Enter ID"
        self._cuopt_secret = "Enter Secret"
        self._cuopt_sak = "Enter SAK"
        self._function_name = ""
        self._function_id = ""
        self.client = None

        self._max_fleet_size = 100
        self._max_fleet_capacity = 100
        self._max_locations = 1000

        self._min_time_limit = 0.01
        self._max_time_limit = 30

        self.prim_data = {}
        self.clear_lines = False

        self._menu_items = [
            MenuItemDescription(header="Examples"),
            MenuItemDescription(
                name=EXTENSION_NAME,
                onclick_fn=lambda a=weakref.proxy(self): a._menu_callback(),
            ),
        ]

        add_menu_items(self._menu_items, "cuOpt")

        self._build_ui()

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_window(self, visible):
        if self._window.visible:
            self._sub_stage_event = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event
            )
        else:
            self._sub_stage_event = None

    def _on_stage_event(self, event):
        """
        Function for monitoring stage events
        """
        if event.type == 2:
            self.prim_data = {}

        # print(f"stage event type int: {event.type}{event.payload}")

    def _build_ui(self):
        if not self._window:
            self._window = ui.Window(
                title=EXTENSION_NAME,
                width=0,
                height=0,
                visible=False,
                dockPreference=ui.DockPreference.LEFT_BOTTOM,
            )
            self._window.set_visibility_changed_fn(self._on_window)

        Number_of_Locations_default = 10
        Fleet_Size_default = 3
        Fleet_Capacity_default = 4
        Solver_Time_Limit_default = 0.1

        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                title = "cuOpt Extension Code and Docs"
                doc_link = "https://docs.nvidia.com/cuopt/"

                overview = "This example demonstrates use of the NVIDIA cuOpt microservice "
                overview += "to solve routing optimization problems (via cost matrix) Isaac Sim"
                overview += "\n\nPress the 'Open Source Code' button to view the source code."

                setup_ui_headers(
                    self._ext_id, __file__, title, doc_link, overview
                )

                # Setting up the UI to connect to cuOpt Managed Service
                connect_cuOpt_frame = ui.CollapsableFrame(
                    title="Connect to cuOpt Managed Service",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )

                with connect_cuOpt_frame:

                    with ui.VStack(style=get_style(), spacing=5, height=0):

                        kwargs = {
                            "label": "cuOpt ID",
                            "type": "stringfield",
                            "default_val": self._cuopt_id_prompt,
                            "tooltip": "ID for cuOpt managed service",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._cuopt_id = str_builder(**kwargs)

                        kwargs = {
                            "label": "cuOpt Secret",
                            "type": "stringfield",
                            "default_val": self._cuopt_secret_prompt,
                            "tooltip": "Secret for cuOpt managed service",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._cuopt_secret = str_builder(**kwargs)

                        kwargs = {
                            "label": "cuOpt SAK",
                            "type": "stringfield",
                            "default_val": self._cuopt_sak_prompt,
                            "tooltip": "SAK for cuOpt managed service",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._cuopt_sak = str_builder(**kwargs)

                        kwargs = {
                            "label": "Function Name",
                            "type": "stringfield",
                            "default_val": self._function_name_prompt,
                            "tooltip": "Function name for cuOpt managed service",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._function_name = str_builder(**kwargs)

                        kwargs = {
                            "label": "Function Id",
                            "type": "stringfield",
                            "default_val": self._function_id_prompt,
                            "tooltip": "Function id for cuOpt managed service",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._function_id = str_builder(**kwargs)

                        kwargs = {
                            "label": "Test cuOpt Connection ",
                            "type": "button",
                            "text": "Test",
                            "tooltip": "Test to verify cuOpt managed service is reachable",
                            "on_clicked_fn": self._test_cuopt_connection_managed_service,
                        }
                        btn_builder(**kwargs)

                # Setting up the UI to connect to cuOpt Microservice
                connect_cuOpt_frame = ui.CollapsableFrame(
                    title="Connect to cuOpt Microservice",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with connect_cuOpt_frame:

                    with ui.VStack(style=get_style(), spacing=5, height=0):

                        kwargs = {
                            "label": "cuOpt IP",
                            "type": "stringfield",
                            "default_val": self._cuopt_ip_prompt,
                            "tooltip": "IP for cuOpt microservice",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._cuopt_ip = str_builder(**kwargs)

                        kwargs = {
                            "label": "cuOpt Port",
                            "type": "stringfield",
                            "default_val": self._cuopt_port_prompt,
                            "tooltip": "Port for cuOpt microservice",
                            "on_clicked_fn": None,
                            "use_folder_picker": False,
                            "read_only": False,
                        }
                        self._cuopt_port = str_builder(**kwargs)

                        kwargs = {
                            "label": "Test cuOpt Connection ",
                            "type": "button",
                            "text": "Test",
                            "tooltip": "Test to verify cuOpt microservice is reachable",
                            "on_clicked_fn": self._test_cuopt_connection_microservice
                        }
                        btn_builder(**kwargs)

                self._cuopt_status_info = ui.Label(" ")

                # Setting up the UI setup the optimization problem
                setup_frame = ui.CollapsableFrame(
                    title="Optimization Problem Setup",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with setup_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):

                        kwargs = {
                            "label": "Fleet Size",
                            "default_val": Fleet_Size_default,
                            "tooltip": "Set the number of vehicles available for this problem",
                        }
                        self.fleet_size = int_builder(**kwargs)

                        kwargs = {
                            "label": "Vehicle Capacity",
                            "default_val": Fleet_Capacity_default,
                            "tooltip": "Set the capacity of each vehicle",
                        }
                        self.fleet_capacity = int_builder(**kwargs)

                        kwargs = {
                            "label": "Number of Locations",
                            "default_val": Number_of_Locations_default,
                            "tooltip": "Set the number of locations to be serviced",
                        }
                        self.num_locations = int_builder(**kwargs)

                        kwargs = {
                            "label": "Solver Time Limit",
                            "default_val": Solver_Time_Limit_default,
                            "tooltip": "Scaling factor for the radii of the specified spheres",
                        }
                        self.time_limit = float_builder(**kwargs)

                        kwargs = {
                            "label": "Setup",
                            "type": "button",
                            "text": "Setup Problem",
                            "tooltip": "Setup ",
                            "on_clicked_fn": self.create_problem_geometry,
                        }
                        self.setup_problem = btn_builder(**kwargs)

                        self._cuopt_setup_status_info = ui.Label(" ")

                # Setting up the UI setup the optimization problem
                run_frame = ui.CollapsableFrame(
                    title="Run cuOpt",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with run_frame:
                    with ui.VStack(style=get_style(), spacing=5, height=0):

                        ui_data_style = {
                            "font_size": 14,
                            "color": 0xBBBBBBBB,
                            "alignment": ui.Alignment.LEFT,
                        }

                        args = {
                            "label": "Run cuOpt",
                            "type": "button",
                            "text": "Solve",
                            "tooltip": "Optimize routing for the current configuration",
                            "on_clicked_fn": self.run_cuopt,
                        }
                        self.run_optimization_btn = btn_builder(**args)

                        self._routes_ui_message = ui.Label(
                            "Run cuOpt for solution",
                            width=350,
                            word_wrap=True,
                            style=ui_data_style,
                        )

    def _form_cuopt_url(self):
        cuopt_ip = self._cuopt_ip.get_value_as_string()
        cuopt_port = self._cuopt_port.get_value_as_string()
        cuopt_url = f"http://{cuopt_ip}:{cuopt_port}/cuopt/"
        return cuopt_url

    # Test if cuopt microservice is up and running
    def _test_cuopt_connection_microservice(self):

        cuopt_ip = self._cuopt_ip.get_value_as_string()
        cuopt_port = self._cuopt_port.get_value_as_string()

        if (cuopt_ip == self._cuopt_ip_prompt) or (
            cuopt_port == self._cuopt_port_prompt
        ):
            self._cuopt_status_info.text = (
                "FAILURE: Please set both an IP and Port"
            )
            return
        self.client = None
        self._cuopt_status_info.text = test_connection_microservice(cuopt_ip, cuopt_port)

    # Test if cuopt managed service is up and running
    def _test_cuopt_connection_managed_service(self):
        cuopt_id = self._cuopt_id.get_value_as_string()
        cuopt_secret = self._cuopt_secret.get_value_as_string()
        cuopt_sak = self._cuopt_sak.get_value_as_string()
        function_name = self._function_name.get_value_as_string()
        function_id = self._function_id.get_value_as_string()

        cuopt_auth = {'id':None, 'secret':None, 'sak':None}

        if not cuopt_sak or cuopt_sak == self._cuopt_sak_prompt:
            print("HERE")
            if (cuopt_id == self._cuopt_id_prompt) or (
                cuopt_secret == self._cuopt_secret_prompt
            ):
                self._cuopt_status_info.text = (
                    "FAILURE: Please set SAK, or, both an ID and SECRET"
                )
                return
            else:
                cuopt_auth["id"] = cuopt_id
                cuopt_auth["secret"] = cuopt_secret
        else:
            cuopt_auth["sak"] = cuopt_sak

        self._cuopt_status_info.text, self.client = test_connection_managed_service(cuopt_auth, function_name, function_id)

    def clear_locations(self):
        locations = []
        for pr in self.prim_data:
            locations.append(self.prim_data[pr]["Path"])

        omni.kit.commands.execute(
            "DeletePrims",
            paths=locations,
        )

        self.prim_data = {}

    def update_location_position(self):
        stage = self._usd_context.get_stage()
        for pr in self.prim_data:
            pose = omni.usd.get_world_transform_matrix(
                stage.GetPrimAtPath(self.prim_data[pr]["Path"])
            )
            self.prim_data[pr]["Location"] = pose[-1][0:-1]

    def problem_setup_validation(
        self, n_vehicles, capacity_val, n_locations, time_limit
    ):
        message = ""

        auto_value_change = False

        # Check lower bounds
        if n_vehicles < 1:
            auto_value_change = True
            n_vehicles = 1
            self.fleet_size.set_value(n_vehicles)
            message += f"\n- Number of vehicles must be >= 1 "

        if capacity_val < 1:
            auto_value_change = True
            capacity_val = 1
            self.fleet_capacity.set_value(capacity_val)
            message += f"\n- Fleet capacity must be >= 1 "

        if n_locations < 1:
            auto_value_change = True
            n_locations = 1
            self.num_locations.set_value(n_locations)
            message += f"\n- Number of locations must be >= 1 "

        if time_limit < self._min_time_limit:
            auto_value_change = True
            time_limit = self._min_time_limit
            self.time_limit.set_value(time_limit)
            message += f"\n- Minimum time limit for this example is {self._min_time_limit}"

        # Check upper bounds
        if n_vehicles > self._max_fleet_size:
            auto_value_change = True
            n_vehicles = self._max_fleet_size
            self.fleet_size.set_value(n_vehicles)
            message += f"\n- Max fleet size for this example is set to {self._max_fleet_size}"

        if capacity_val > self._max_fleet_capacity:
            auto_value_change = True
            capacity_val = self._max_fleet_capacity
            self.fleet_capacity.set_value(capacity_val)
            message += f"\n- Max fleet capacity for this example is set to {self._max_fleet_capacity}"

        if n_locations > self._max_locations:
            auto_value_change = True
            n_locations = self._max_locations
            self.num_locations.set_value(n_locations)
            message += f"\n- Max locations for this example is set to {self._max_locations}"

        if time_limit > self._max_time_limit:
            auto_value_change = True
            time_limit = self._max_time_limit
            self.time_limit.set_value(time_limit)
            message += f"\n- Max time limit for this example is set to {self._max_time_limit}"

        # Check not using unnecessary number of vehicles
        if n_vehicles > n_locations:
            auto_value_change = True
            n_vehicles = n_locations
            self.fleet_size.set_value(n_vehicles)
            message += (
                f"\n- Number of vehicles should be <= number of locations"
            )

        if n_vehicles * capacity_val < n_locations:
            auto_value_change = True

            n_locations = n_vehicles * capacity_val
            self.num_locations.set_value(n_locations)

            message += f"\n- {n_vehicles} vehicles with capacity {capacity_val} can service a "
            message += (
                f"\nmaximum of {n_locations} locations each with demand 1 "
            )

        if auto_value_change:
            message = "NOTE : AUTOMATIC VALUE CHANGE" + message
            message += (
                f"\n\nFor advanced usage reference the cuOpt documentation"
            )

        return message

    def create_problem_geometry(self):

        self._cuopt_setup_status_info.text = self.problem_setup_validation(
            self.fleet_size.get_value_as_int(),
            self.fleet_capacity.get_value_as_int(),
            self.num_locations.get_value_as_int(),
            self.time_limit.get_value_as_float(),
        )

        if bool(self.prim_data):
            self.clear_locations()
            draw = debug_draw._debug_draw.acquire_debug_draw_interface()
            draw.clear_lines()

        min_pos = -40.0
        max_pos = 40.0
        stage = self._usd_context.get_stage()
        current_index = 0

        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_path="/World/Depot",
            prim_type="Cone",
            attributes={"radius": 2.5, "height": 5.0},
            select_new_prim=False,
        )
        self.prim_data[current_index] = {
            "Name": "Depot",
            "Path": "/World/Depot",
            "Prim": stage.GetPrimAtPath("/World/Depot"),
            "Location": (0, 0, 0),
        }

        current_index += 1

        for location in range(1, self.num_locations.get_value_as_int() + 1):
            location_prim_path = f"/World/Location_{location}"
            omni.kit.commands.execute(
                "CreatePrimWithDefaultXform",
                prim_path=location_prim_path,
                prim_type="Sphere",
                attributes={"radius": 1.0},
                select_new_prim=False,
            )

            rand_x = random.uniform(min_pos, max_pos)
            rand_y = random.uniform(min_pos, max_pos)
            omni.kit.commands.execute(
                "TransformPrimCommand",
                path=location_prim_path,
                old_transform_matrix=None,
                new_transform_matrix=Gf.Matrix4d().SetTranslateOnly(
                    Gf.Vec3d(rand_x, rand_y, 0)
                ),
            )
            self.prim_data[current_index] = {
                "Name": location,
                "Path": location_prim_path,
                "Prim": stage.GetPrimAtPath(location_prim_path),
                "Location": [rand_x, rand_y, 0],
            }

            current_index += 1

    def distance_matrix_from_point_list(self, point_list, scale):
        """
        Create a distance matrix from a point list
        """
        return scale * squareform(pdist(point_list, metric="euclidean"))

    def get_routes(self, raw_routes):
        routes = []
        cur_route = []
        writing = False
        for i in range(len(raw_routes)):
            stop_idx = raw_routes[str(i)]

            if (not writing) and (stop_idx == 0):
                writing = True
                cur_route.append(stop_idx)

            elif (writing) and (stop_idx == 0):
                writing = False
                cur_route.append(stop_idx)
                routes.append(cur_route)
                cur_route = []

            else:
                cur_route.append(stop_idx)

        return routes

    def run_cuopt(self):

        if bool(self.prim_data):
            self.update_location_position()
            num_locations = self.num_locations.get_value_as_int()
            num_vehicles = self.fleet_size.get_value_as_int()
            vehicle_capacity = self.fleet_capacity.get_value_as_int()

            distance_matrix = self.distance_matrix_from_point_list(
                [
                    self.prim_data[i]["Location"]
                    for i in range(len(self.prim_data))
                ],
                1,
            )

            print("Running cuOpt")

            self._stage = self._usd_context.get_stage()

            # Solver Settings
            solver_config = {
                "time_limit": self.time_limit.get_value_as_float(),
            }

            cost_data = {
                "cost_matrix": {0: distance_matrix.tolist()},
            }

            fleet_data = {
                "vehicle_locations": [[0, 0]] * num_vehicles,
                "capacities": [[vehicle_capacity] * num_vehicles],
            }

            task_data = {
                "task_locations": [i for i in range(1, num_locations + 1)],
                "demand": [[1] * (num_locations)],
            }

            environment_data = {
                "cost_matrix_data": cost_data,
                "fleet_data": fleet_data,
                "task_data": task_data,
                "solver_config": solver_config,
            }

            if self.client is None:
                cuopt_url = self._form_cuopt_url()
                cuopt_server = cuOptRunner(cuopt_url)

                cuopt_solution = cuopt_server.get_routes(environment_data)
                routes = cuopt_solution

            else:
                res = self.client.get_optimized_routes(environment_data)
                routes = res["response"]["solver_response"]

            self.draw_routes(routes["vehicle_data"])

            # Display the routes on UI
            self._routes_ui_message.text = show_vehicle_routes(routes)

    def draw_routes(self, routes):

        draw = debug_draw._debug_draw.acquire_debug_draw_interface()

        draw.clear_lines()

        for key, routes in routes.items():

            point_list_1 = []
            point_list_2 = []

            route = routes["route"]
            for idx, stop in enumerate(route[0:-1]):
                point_list_1.append(tuple(self.prim_data[stop]["Location"]))
                point_list_2.append(
                    tuple(self.prim_data[route[idx + 1]]["Location"])
                )

            N = len(point_list_1)
            r = random.uniform(0, 1)
            g = random.uniform(0, 1)
            b = random.uniform(0, 1)
            colors = [(r, g, b, 1) for _ in range(N)]
            sizes = [5 for _ in range(N)]

            draw.draw_lines(point_list_1, point_list_2, colors, sizes)

    def on_shutdown(self):
        self._editor_event_subscription = None
        remove_menu_items(self._menu_items, "cuOpt")
        self._window = None
        gc.collect()
