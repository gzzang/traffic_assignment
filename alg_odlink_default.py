# @Time    : 2020/3/1 14:37
# @Author  : gzzang
# @File    : alg_odlink_default
# @Project : traffic_assignment

# 路段流量和OD相关路段流量

import time
import cvxpy as cp
import numpy as np
from preparation import read_data
from preparation import set_parameter
from preparation import gen_matrix


# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def alg_odlink_default(network_name, bool_display=True):
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

    start_time = time.time()

    x_odlink = cp.Variable(link_number * od_number)
    x_link = od_odlink_relation @ x_odlink
    objective = cp.Minimize(cp.sum(link_free * x_link) + cp.sum(
        link_free * para_a / (para_b + 1) * link_capacity * cp.power(x_link / link_capacity, para_b + 1)))

    constraints = [x_odlink >= 0, node_odlink_relation @ x_odlink - node_oddemand == 0]
    result = cp.Problem(objective=objective, constraints=constraints).solve()  # reltol=1e-15, abstol=1e-15

    target_value = objective.value
    link_flow = od_odlink_relation @ x_odlink.value
    runtime = time.time() - start_time

    if bool_display:
        print('----alg_odlink_default----')
        print(f'target_value: {target_value}')
        print(f'link_flow: {link_flow}')
        print(f'runtime: {runtime}')
        print('--------')


if __name__ == '__main__':
    alg_odlink_default('simple_network')
