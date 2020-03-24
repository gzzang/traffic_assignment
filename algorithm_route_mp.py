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

def algorithm_route_mp(data_with_route_variable, bool_display=True):
    start_time = time.time()
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

    x_route_flow = cp.Variable(route_number)
    x_link_flow = x_route_flow @ route_link_incidence
    objective = cp.Minimize(cp.sum(link_free * x_link_flow) + cp.sum(
        link_free * para_a / (para_b + 1) * link_capacity * cp.power(x_link_flow / link_capacity, para_b + 1)))
    constraints = [x_route_flow >= 0, od_route_incidence @ x_route_flow == od_demand]
    result = cp.Problem(objective=objective, constraints=constraints).solve()
    current_link_flow = x_link_flow.value
    current_route_flow = x_route_flow.value
    optimal_value = objective.value

    if bool_display:
        print('----alg_route_default----')
        print(f"current_link_flow:{current_link_flow}")
        print(f"current_route_flow:{current_route_flow}")
        print(f"optimal_value:{optimal_value}")
        print(f'runtime:{time.time() - start_time}')
        print('----')

    return current_route_flow, optimal_value

if __name__ == '__main__':
    data_with_route_variable = read_data_plus_route('simple_network')
    algorithm_route_mp(data_with_route_variable,bool_display=False)
