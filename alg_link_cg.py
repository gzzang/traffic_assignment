# @Time    : 2020/3/2 22:01
# @Author  : gzzang
# @File    : alg_link_cg
# @Project : traffic_assignment

# 成功调用路径求解函数

import numpy as np
import igraph as ig
from preparation import read_data
from preparation import set_parameter
from algorithm_route_vi import algorithm_route_vi
from algorithm_route_mp import algorithm_route_mp
from algorithm_route_msa import algorithm_route_msa

import time
import pandas as pd
import os


def alg_link_cg(network_name, bool_display=True, bool_display_iteration=True, bool_route_result_output=False):
    data = read_data(network_name)
    link_node_pair = data['link_node_pair']
    link_capacity = data['link_capacity']
    link_free = data['link_free_flow_time']
    od_node_pair = data['od_node_pair']
    od_demand = data['od_demand']
    od_number = data['od_number']
    link_number = data['link_number']
    node_number = data['node_number']

    parameter = set_parameter()
    para_a = parameter['a']
    para_b = parameter['b']

    g = ig.Graph()
    g.add_vertices(node_number)
    g.add_edges(link_node_pair)

    def cal_sp_of_one_od(link_flow, node_list):
        link_time = link_free * (1 + para_a * (link_flow / link_capacity) ** para_b)
        g.es['weight'] = link_time
        node_incidence = np.zeros_like(link_flow, dtype=bool)
        node_incidence[g.get_shortest_paths(node_list[0], to=node_list[1], weights='weight', output='epath')[0]] = True
        return node_incidence

    start_time = time.time()
    initial_od_list_route_link_incidence = []
    zero_link_flow = np.zeros_like(link_free)
    for one_pair_list in od_node_pair:
        sp_link_incidence = cal_sp_of_one_od(zero_link_flow, one_pair_list)
        initial_od_list_route_link_incidence.append(sp_link_incidence.reshape([1, link_number]))

    initial_route_link_incidence = np.vstack(initial_od_list_route_link_incidence)

    current_link_flow = od_demand @ initial_route_link_incidence
    current_od_list_route_link_incidence = initial_od_list_route_link_incidence

    iteration_number = 20
    target_gap = 1e-20
    termination_bool = False
    optimum_bool = False
    iteration_index = 0
    while not (termination_bool or optimum_bool):
        for od_index, (one_route_link_incidence, one_pair_list) in enumerate(
                zip(current_od_list_route_link_incidence, od_node_pair)):
            sp_link_incidence = cal_sp_of_one_od(current_link_flow, one_pair_list)
            if not np.any(np.all(one_route_link_incidence == sp_link_incidence, axis=1)):
                current_od_list_route_link_incidence[od_index] = np.vstack(
                    (one_route_link_incidence, sp_link_incidence))

        route_link_incidence = np.vstack(current_od_list_route_link_incidence)
        od_route_number = np.array(
            [one_route_link_array.shape[0] for one_route_link_array in current_od_list_route_link_incidence])
        route_number = np.sum(od_route_number)
        od_route_incidence = np.zeros([od_number, route_number], dtype=bool)
        temp = 0
        for od_index, value in enumerate(od_route_number):
            od_route_incidence[od_index, temp:(temp + value)] = True
            temp += value

        data_with_route_variable = data
        data_with_route_variable['od_route_incidence'] = od_route_incidence
        data_with_route_variable['route_link_incidence'] = route_link_incidence
        data_with_route_variable['route_number'] = route_link_incidence.shape[0]
        # current_route_flow, temp_value = algorithm_route_vi(data_with_route_variable, bool_display=False,
        #                                                        bool_display_iteration=False)
        # current_route_flow, temp_value = algorithm_route_mp(data_with_route_variable, bool_display=False)
        current_route_flow, temp_value = algorithm_route_msa(data_with_route_variable, bool_display=False)

        current_link_flow = current_route_flow @ route_link_incidence

        if iteration_index != 0:
            gap = np.abs((temp_value - optimal_value) / optimal_value)
        else:
            gap = 1
        if gap < target_gap:
            optimum_bool = True
        elif iteration_index == iteration_number:
            termination_bool = True

        optimal_value = temp_value
        iteration_index += 1
        if bool_display:
            if bool_display_iteration:
                print('----iteration_alg_link_cg_default----')
                print(f"iteration_index:{iteration_index}")
                print(f'gap:{gap}')
                print(f"od_route_number:{od_route_number}")
                print(f"current_link_flow:{current_link_flow}")
                print(f"optimal_value:{optimal_value}")
                print('--------')

    if bool_display:
        print('----alg_link_cg_default----')
        print(f'optimum_bool:{optimum_bool}')
        print(f'gap:{gap}')
        print(f"iteration_index:{iteration_index}")
        print(f"od_route_number:{od_route_number}")
        print(f"current_link_flow:{current_link_flow}")
        print(f"optimal_value:{optimal_value}")
        print(f'runtime:{time.time() - start_time}')
        print('--------')

    if bool_route_result_output:
        print('--------')
        print(f'bool_route_result_output:{bool_route_result_output}')
        print('--------')

        output_path = 'output'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        pd.DataFrame({string: value for string, value in
                      zip(('route_index', 'link_index'), np.nonzero(route_link_incidence))}).to_csv(
            output_path + '/' + network_name + '_route_link_relation.csv', index=False)
        pd.DataFrame({string: value for string, value in
                      zip(('od_index', 'route_index'), np.nonzero(od_route_incidence))}).to_csv(
            output_path + '/' + network_name + '_od_route_relation.csv', index=False)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    alg_link_cg('nguyen_dupuis')
