# @Time    : 2020/3/22 13:27
# @Author  : gzzang
# @File    : algorithm_route_msa
# @Project : traffic_assignment

import numpy as np
from preparation import read_data_plus_route
from preparation import set_parameter
import time


def algorithm_route_msa(data_with_route_variable, bool_display=True, bool_display_iteration=True):
    start_time = time.time()

    def cal_objective_value_from_route_flow(route_flow):
        link_flow = route_flow @ route_link_incidence
        objective_value = (link_free * link_flow + link_free * para_a / (para_b + 1) * link_capacity * (
                (link_flow / link_capacity) ** (para_b + 1))).sum()
        return objective_value


    link_node_pair = data_with_route_variable['link_node_pair']
    link_capacity = data_with_route_variable['link_capacity']
    link_free = data_with_route_variable['link_free_flow_time']
    od_node_pair = data_with_route_variable['od_node_pair']
    od_demand = data_with_route_variable['od_demand']
    od_number = data_with_route_variable['od_number']
    link_number = data_with_route_variable['link_number']
    node_number = data_with_route_variable['node_number']

    od_route_incidence = data_with_route_variable['od_route_incidence']
    route_link_incidence = data_with_route_variable['route_link_incidence']
    route_number = data_with_route_variable['route_number']

    parameter = set_parameter()
    para_a = parameter['a']
    para_b = parameter['b']

    od_route_number = od_route_incidence.sum(axis=1)
    current_route_flow = (od_demand / od_route_number) @ od_route_incidence
    optimal_value = cal_objective_value_from_route_flow(current_route_flow)

    tolerance = 1e-9
    iteration_number = 1000
    termination_bool = False
    optimum_bool = False
    iteration_index = 0
    while not (termination_bool or optimum_bool):
        link_flow = current_route_flow @ route_link_incidence
        link_time = link_free * (1 + para_a * (link_flow ** 4) / (link_capacity ** 4))
        route_time = route_link_incidence @ link_time

        od_route_time_incidence = np.inf * np.ones_like(od_route_incidence)
        od_route_time_incidence[od_route_incidence] = route_time

        od_minimal_route_time_index = od_route_time_incidence.argmin(axis=1)

        new_route_flow = np.zeros_like(current_route_flow)
        new_route_flow[od_minimal_route_time_index] = od_demand

        temp_route_flow = (iteration_index + 1) / (iteration_index + 2) * current_route_flow + 1 / (
                iteration_index + 2) * new_route_flow
        temp_value = cal_objective_value_from_route_flow(temp_route_flow)

        if iteration_index != 0:
            gap = np.abs((temp_value - optimal_value) / optimal_value)
        else:
            gap = 1
        if gap < tolerance:
            optimum_bool = True
        elif iteration_index == iteration_number:
            termination_bool = True

        current_route_flow = temp_route_flow
        optimal_value = temp_value
        iteration_index += 1
        if bool_display_iteration:
            print('--------')
            print(f"iteration_index:{iteration_index}")
            print(f"gap:{gap}")
            print(f"current_link_flow:{current_route_flow}")
            print(f"optimal_value:{optimal_value}")
            print('--------')

    if bool_display:
        print('----alg_route_msa----')
        print(f"iteration_index:{iteration_index}")
        print(f"gap:{gap}")
        print(f"tolerance:{tolerance}")
        print(f"optimum_bool:{optimum_bool}")
        print(f"current_route_flow:{current_route_flow}")
        print(f"optimal_value:{optimal_value}")
        print(f'runtime:{time.time() - start_time}')
        print('--------')

    return current_route_flow, optimal_value


if __name__ == '__main__':
    data_with_route_variable = read_data_plus_route('nguyen_dupuis')
    algorithm_route_msa(data_with_route_variable, bool_display=True, bool_display_iteration=False)
