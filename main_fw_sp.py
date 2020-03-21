# @Time    : 2020/3/3 17:40
# @Author  : gzzang
# @File    : main-fw-shortest-path
# @Project : tap_link_form


import numpy as np
import igraph as ig
from preparation import *
from algorithm import algorithm_line_search

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

data = read_data()
link_node_pair = data['link_node_pair']
link_capacity = data['link_capacity']
link_free = data['link_free_flow_time']
od_node_pair = data['od_node_pair']
od_demand = data['od_demand']
od_number = data['od_number']
link_number = data['link_number']
node_number = data['node_number']

matrix = gen_matrix(data)
node_odlink_relation = matrix['node_odlink_relation']
node_oddemand = matrix['node_oddemand']
od_odlink_relation = matrix['od_odlink_relation']

parameter = set_parameter()
para_a = parameter['a']
para_b = parameter['b']

iteration_number = 500

g = ig.Graph()
g.add_vertices(node_number)
g.add_edges(link_node_pair)


def cal_direction_odlink_flow(odlink_flow):
    """
    根据路段时间计算基于最短路的全有全无路段流量
    :return:
    """

    def cal_link_time(odlink_flow):
        return link_free * (1 + para_a * (od_odlink_relation @ odlink_flow / link_capacity) ** para_b)

    g.es['weight'] = cal_link_time(odlink_flow)
    optimal_odlink_flow = []
    for od, demand in zip(od_node_pair, od_demand):
        shortest_route_link = np.zeros(link_number)
        shortest_route_link[g.get_shortest_paths(od[0], to=od[1], weights='weight', output='epath')[0]] = demand
        optimal_odlink_flow = np.hstack((optimal_odlink_flow, shortest_route_link))
    return optimal_odlink_flow - odlink_flow


current_odlink_flow = cal_direction_odlink_flow(np.zeros(od_number * link_number))

target_gap = 1e-7
iteration_number = 1000
termination_bool = False
optimum_bool = False
iteration_index = 0
while not (termination_bool or optimum_bool):
    direction_odlink_flow = cal_direction_odlink_flow(current_odlink_flow)

    current_odlink_flow, temp_value = algorithm_line_search(current_odlink_flow, direction_odlink_flow,
                                                            od_odlink_relation,
                                                            link_free, para_a, para_b, link_capacity)

    if iteration_index != 0:
        gap = np.abs((temp_value - optimal_value) / optimal_value)
        if gap < target_gap:
            optimum_bool = True
        elif iteration_index == iteration_number:
            termination_bool = True

    optimal_value = temp_value
    iteration_index += 1
    print(f"iteration_index:{iteration_index}")

print(f"direction:{direction_odlink_flow}")
print(f'final:{current_odlink_flow}')
print(f'value:{optimal_value}')
print('*********************')
