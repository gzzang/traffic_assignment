# @Time    : 2020/3/22 11:36
# @Author  : gzzang
# @File    : algorithm_route_vi
# @Project : traffic_assignment



import numpy as np
import gurobipy as gp
from gurobipy import GRB

from preparation import read_data_plus_route
from preparation import set_parameter


def algorithm_route_vi(data_with_route_variable, bool_display=True, bool_display_iteration=True):
    gp.setParam('OutputFlag', 0)

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

    def cal_objective_value_from_route_flow(route_flow):
        link_flow = route_flow @ route_link_incidence
        objective_value = (link_free * link_flow + link_free * para_a / (para_b + 1) * link_capacity * (
                (link_flow / link_capacity) ** (para_b + 1))).sum()
        return objective_value

    def cal_route_time(cf):
        rf = cf / 1
        lf = rf @ route_link_incidence
        lt = link_free * (1 + para_a * (lf ** para_b) / (link_capacity ** para_b))
        rt = route_link_incidence @ lt
        ct = rt + 0
        return ct

    od_route_number = od_route_incidence.sum(axis=1)
    initial_route_flow = (od_demand / od_route_number) @ od_route_incidence
    optimal_value = cal_objective_value_from_route_flow(initial_route_flow)

    current_route_flow = initial_route_flow

    tolerance = 1e-15
    iteration_number = 5000
    termination_bool = False
    optimum_bool = False
    iteration_index = 0
    while not (termination_bool or optimum_bool):
        current_route_cost = cal_route_time(current_route_flow)

        matrix_h = np.eye(route_number)
        rho = 0.1
        matrix_quadratic = matrix_h / 2
        vector_quadratic = rho * current_route_cost.reshape([1, route_number]) - matrix_h @ current_route_flow

        m = gp.Model()
        var = m.addMVar(shape=route_number, lb=0.0)
        m.addConstr(od_route_incidence @ var == od_demand)
        m.setObjective(var @ matrix_quadratic @ var + vector_quadratic @ var, GRB.MINIMIZE)
        m.optimize()
        new_route_flow = var.x

        temp_value = cal_objective_value_from_route_flow(new_route_flow)

        if iteration_index != 0:
            gap = np.abs((temp_value - optimal_value) / optimal_value)
        else:
            gap = 1
        if gap < tolerance:
            optimum_bool = True
        elif iteration_index == iteration_number:
            termination_bool = True

        optimal_value = temp_value
        current_route_flow = new_route_flow
        iteration_index += 1
        if bool_display_iteration:
            print('----iteration_of_algorithm_route_vi----')
            print(f'iteration_index:{iteration_index}')
            print(f'gap:{gap}')
            print(f'optimal_value:{optimal_value}')
            print('--------')

    if bool_display:
        print('----algorithm_route_vi----')
        print(f'iteration_index:{iteration_index}')
        print(f'optimum_bool:{optimum_bool}')
        print(f'gap:{gap}')
        print(f'current_route_flow:{current_route_flow}')
        print(f'cost:{cal_route_time(current_route_flow)}')
        print(f'optimal_value:{optimal_value}')
        print('--------')

    return current_route_flow, optimal_value


if __name__ == '__main__':
    gp.setParam('OutputFlag', 0)
    network_name = 'simple_network'
    data_with_route_variable = read_data_plus_route(network_name)
    current_route_flow, optimal_value = algorithm_route_vi(data_with_route_variable, bool_display=True, bool_display_iteration=True)
