# @Time    : 2020/3/22 14:37
# @Author  : gzzang
# @File    : alg_default_based_on_route
# @Project : traffic_assignment

import numpy as np
import cvxpy as cp
from preparation import read_data_plus_route
from preparation import set_parameter

import time

# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def alg_route_default(network_name, bool_display=True):
    start_time = time.time()

    data = read_data_plus_route(network_name)
    link_node_pair = data['link_node_pair']
    link_capacity = data['link_capacity']
    link_free = data['link_free_flow_time']
    od_node_pair = data['od_node_pair']
    od_demand = data['od_demand']
    od_number = data['od_number']
    link_number = data['link_number']
    node_number = data['node_number']

    od_route_incidence = data['od_route_incidence']
    route_link_incidence = data['route_link_incidence']
    route_number = data['route_number']

    parameter = set_parameter()
    para_a = parameter['a']
    para_b = parameter['b']

    x_route_flow = cp.Variable(route_number)
    x_link_flow = x_route_flow @ route_link_incidence
    objective = cp.Minimize(cp.sum(link_free * x_link_flow) + cp.sum(
        link_free * para_a / (para_b + 1) * link_capacity * cp.power(x_link_flow / link_capacity, para_b + 1)))
    constraints = [x_route_flow >= 0, od_route_incidence @ x_route_flow == od_demand]
    result = cp.Problem(objective=objective, constraints=constraints).solve()
    current_link_flow = x_link_flow.value
    optimal_value = objective.value

    if bool_display:
        print('----alg_route_default----')
        print(f"current_link_flow:{current_link_flow}")
        print(f"optimal_value:{optimal_value}")
        print(f'runtime:{time.time() - start_time}')
        print('----')


if __name__ == '__main__':
    alg_route_default('simple_network')
