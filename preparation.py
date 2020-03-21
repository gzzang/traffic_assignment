# @Time    : 2020/3/3 22:09
# @Author  : gzzang
# @File    : preparation
# @Project : tap_link_form

import numpy as np
import pandas as pd
import pickle
import os


def set_parameter():
    para = dict()
    para['a'] = 0.15
    para['b'] = 4
    return para


def read_from_pickle(network_name, folder_path):
    with open(folder_path + network_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def read_from_csv(network_name, folder_path):
    link_file_path = folder_path + network_name + '_link.csv'
    od_file_path = folder_path + network_name + '_od.csv'

    link_df = pd.read_csv(link_file_path)
    od_df = pd.read_csv(od_file_path)

    link_node_pair = link_df.iloc[:, :2].to_numpy()
    link_free_flow_time = link_df.iloc[:, 2].to_numpy()
    link_capacity = link_df.iloc[:, 3].to_numpy()

    od_node_pair = od_df.iloc[:, :2].to_numpy()
    od_demand = od_df.iloc[:, 2].to_numpy()

    data = {'od_node_pair': od_node_pair,
            'od_demand': od_demand,
            'link_node_pair': link_node_pair,
            'link_free_flow_time': link_free_flow_time,
            'link_capacity': link_capacity}

    return data


# def read_data(network_name='simple_network'):
def read_data(network_name='siouxfalls'):
# def read_data(network_name='nguyen_dupuis'):
# def read_data(network_name='nguyen_dupuis'):
    """
    :return: ['link_node_pair', 'link_capacity', 'link_free_flow_time',
        'od_node_pair', 'od_demand', 'od_number', 'link_number', 'node_number']
    """
    folder_path = f'data/'

    if os.path.exists(folder_path + network_name + '.pkl'):
        print('pickle')
        data = read_from_pickle(network_name, folder_path)
    else:
        data = read_from_csv(network_name, folder_path)
        with open(folder_path + network_name + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    od_node_pair = data['od_node_pair']
    link_node_pair = data['link_node_pair']

    od_number = od_node_pair.shape[0]
    link_number = link_node_pair.shape[0]
    node_number = np.max(link_node_pair) + 1

    data['od_number'] = od_number
    data['link_number'] = link_number
    data['node_number'] = node_number
    return data


def gen_matrix(data):
    link_node_pair = data['link_node_pair']
    link_capacity = data['link_capacity']
    link_free = data['link_free_flow_time']
    od_node_pair = data['od_node_pair']
    od_demand = data['od_demand']
    od_number = data['od_number']
    link_number = data['link_number']
    node_number = data['node_number']

    node_link_incidence = np.zeros((node_number, link_number))
    for i, edge in enumerate(link_node_pair):
        node_link_incidence[edge[0], i] = 1
        node_link_incidence[edge[1], i] = -1

    node_od_incidence = np.zeros((node_number, od_number))
    for i, (od, demand) in enumerate(zip(od_node_pair, od_demand)):
        node_od_incidence[od[0], i] = demand
        node_od_incidence[od[1], i] = -demand

    node_odlink_relation = np.kron(np.eye(od_number), node_link_incidence)
    node_oddemand = node_od_incidence.flatten(order='F')
    od_odlink_relation = np.tile(np.eye(link_number), (1, od_number))

    matrix = {}
    matrix['node_odlink_relation'] = node_odlink_relation
    matrix['node_oddemand'] = node_oddemand
    matrix['od_odlink_relation'] = od_odlink_relation
    return matrix


def read_matrix_data():
    od_node, od_demand, link_list, free_time_list, capacity_list, multiple_od_node_link_incidence, multiple_od_node_od_incidence, multiple_od_link_flow_matrix, parameter_alpha, parameter_beta, od_number, link_number, node_number = read_relation_and_matrix_data()
    return free_time_list, capacity_list, multiple_od_node_link_incidence, multiple_od_node_od_incidence, multiple_od_link_flow_matrix, parameter_alpha, parameter_beta, od_number, link_number, node_number


if __name__ == '__main__':
    print(read_data('nguyen_dupuis'))
