# @Time    : 2020/3/4 12:20
# @Author  : gzzang
# @File    : alg_link_fw_sp
# @Project : tap_link_form


import numpy as np
import igraph as ig
from preparation import read_data
from preparation import set_parameter
from algorithm import algorithm_line_search_link
import time


# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def alg_link_fw_sp(network_name, bool_display=True, bool_display_iteration=True):
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

    def cal_direction_link_flow(link_flow):
        """
        根据路段时间计算基于最短路的全有全无路段流量
        :param link_flow:
        :return:
        """

        def cal_link_time(link_flow):
            return link_free * (1 + para_a * (link_flow / link_capacity) ** para_b)

        g.es['weight'] = cal_link_time(link_flow)
        optimal_link_flow = np.zeros_like(link_flow)
        for od, demand in zip(od_node_pair, od_demand):
            shortest_path_link_flow = np.zeros_like(link_flow)
            shortest_path_link_flow[g.get_shortest_paths(od[0], to=od[1], weights='weight', output='epath')[0]] = demand
            optimal_link_flow += shortest_path_link_flow
        return optimal_link_flow - link_flow

    start_time = time.time()
    current_link_flow = cal_direction_link_flow(np.zeros_like(link_free))

    target_gap = 1e-7
    iteration_number = 1000
    termination_bool = False
    optimum_bool = False
    iteration_index = 0
    while not (termination_bool or optimum_bool):
        direction_link_flow = cal_direction_link_flow(current_link_flow)
        current_link_flow, temp_value = algorithm_line_search_link(current_link_flow, direction_link_flow, link_free,
                                                                   para_a, para_b, link_capacity)

        if iteration_index == 0:
            gap = 1
        else:
            gap = np.abs((temp_value - optimal_value) / optimal_value)
        if gap < target_gap:
            optimum_bool = True
        elif iteration_index == iteration_number:
            termination_bool = True

        optimal_value = temp_value
        iteration_index += 1
        if bool_display_iteration:
            print()
            print(f"gap:{gap}")
            print(f"iteration_index:{iteration_index}")
    if bool_display:
        print('----alg_link_fw_sp----')
        print(f"direction_link_flow:{direction_link_flow}")
        print(f'final:{current_link_flow}')
        print(f'value:{optimal_value}')
        print('--------')

        print(f'runtime:{time.time() - start_time}')


if __name__ == '__main__':
    alg_link_fw_sp('simple_network')
