# @Time    : 2020/3/10 16:12
# @Author  : gzzang
# @File    : verification
# @Project : traffic_assignment
from preparation import read_data
from preparation import set_parameter
import pickle
import numpy as np

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

with open('flow.pkl', 'rb') as f:
    current_link_flow = pickle.load(f)

function_value = np.sum(link_free * current_link_flow) + np.sum(
    link_free * para_a / (para_b + 1) * link_capacity * np.power(current_link_flow / link_capacity, para_b + 1))

print(function_value)
