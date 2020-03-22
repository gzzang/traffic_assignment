# @Time    : 2020/3/3 14:06
# @Author  : gzzang
# @File    : main_fw_optimization
# @Project : traffic_assignment

# 确定方向的两种方法：
# 优化问题和求解最短路
# 此外还可以分别对应路段流量和起讫点对相关路段流量
# 此方法是优化问题方式
# 对应的是起讫点对相关路段流量

# 对应路段流量的方法没有写

import cvxpy as cp
import numpy as np
from algorithm import algorithm_line_search

from preparation import read_data
from preparation import set_parameter
from preparation import gen_matrix


# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def alg_odlink_fw_optimization(network_name, bool_display=True, bool_display_iteration=True):
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

    matrix = gen_matrix(data)
    node_odlink_relation = matrix['node_odlink_relation']
    node_oddemand = matrix['node_oddemand']
    od_odlink_relation = matrix['od_odlink_relation']

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

    target_gap = 1e-7
    iteration_number = 1000
    termination_bool = False
    optimum_bool = False
    iteration_index = 0
    while not (termination_bool or optimum_bool):
        optimal_vertex = cal_odlink_flow(current_odlink_flow)

        direction = optimal_vertex - current_odlink_flow

        current_odlink_flow, temp_value = algorithm_line_search(current_odlink_flow, direction,
                                                                od_odlink_relation, link_free, para_a,
                                                                para_b, link_capacity)

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
            print('--------')
            print(f"gap:{gap}")
            print(f"iteration_index:{iteration_index}")
            print('--------')

    if bool_display:
        print('----alg_odlink_fw_optimization----')
        print(f"optimum_bool:{optimum_bool}")
        print(f"iteration_index:{iteration_index}")
        print(f"optimal_vertex:{optimal_vertex}")
        print(f"direction:{direction}")
        print(f'final:{current_odlink_flow}')
        print(f'value:{optimal_value}')
        print('--------')


if __name__ == '__main__':
    alg_odlink_fw_optimization('simple_network')
