# @Time    : 2020/3/1 14:37
# @Author  : gzzang
# @File    : main_default
# @Project : cvxpy

import cvxpy as cp
import numpy as np
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

x_odlink = cp.Variable(link_number * od_number)
x_link = od_odlink_relation @ x_odlink
objective = cp.Minimize(cp.sum(link_free * x_link) + cp.sum(
    link_free * para_a / (para_b + 1) * link_capacity * cp.power(x_link / link_capacity, para_b + 1)))

constraints = [x_odlink >= 0, node_odlink_relation @ x_odlink - node_oddemand == 0]
result = cp.Problem(objective=objective, constraints=constraints).solve(reltol=1e-15, abstol=1e-15)

print(od_odlink_relation)
print(x_odlink.value)

aaa= od_odlink_relation @ x_odlink.value

print(f'optimal objective value: {objective.value}')
print(f'optimal value: {od_odlink_relation @ x_odlink.value}')

print(np.sum(link_free * aaa) + np.sum(
    link_free * para_a / (para_b + 1) * link_capacity * np.power(aaa / link_capacity, para_b + 1)))