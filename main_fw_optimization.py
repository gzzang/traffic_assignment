# @Time    : 2020/3/3 14:06
# @Author  : gzzang
# @File    : main_fw_optimization
# @Project : tap_link_form

import cvxpy as cp
import numpy as np
from algorithm import algorithm_line_search

from preparation import read_data
from preparation import set_parameter
from preparation import gen_matrix

# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

data = read_data()
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

matrix = gen_matrix(data)
node_odlink_relation = matrix['node_odlink_relation']
node_oddemand = matrix['node_oddemand']
od_odlink_relation = matrix['od_odlink_relation']

iteration_number = 500


def cal_odlink_flow(current_odlink_flow):
    """
    根据路段时间计算基于最短路的全有全无路段流量
    :param link_time:
    :return:
    """

    def derivative(xx_odlink):
        return link_free * (
                1 + para_a * (od_odlink_relation @ xx_odlink / link_capacity) ** para_b) @ od_odlink_relation

    x_odlink_flow = cp.Variable(link_number * od_number)
    odlink_derivative = derivative(current_odlink_flow)
    objective = cp.Minimize(odlink_derivative @ x_odlink_flow)
    constraints = [x_odlink_flow >= 0, node_odlink_relation @ x_odlink_flow - node_oddemand == 0]
    cp.Problem(objective=objective, constraints=constraints).solve()
    return x_odlink_flow.value


current_odlink_flow = cal_odlink_flow(np.zeros(link_number * od_number))
print(f'x_initial:{current_odlink_flow}')

for i in range(iteration_number):
    optimal_vertex = cal_odlink_flow(current_odlink_flow)

    direction = optimal_vertex - current_odlink_flow

    current_odlink_flow, optimal_value = algorithm_line_search(current_odlink_flow, direction,
                                                               od_odlink_relation, link_free, para_a,
                                                               para_b, link_capacity)

    print(f"optimal_vertex:{optimal_vertex}")
    print(f"direction:{direction}")
    print(f'final:{current_odlink_flow}')
    print(f'value:{optimal_value}')
    print('*********************')
