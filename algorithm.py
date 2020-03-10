# @Time    : 2020/3/3 23:22
# @Author  : gzzang
# @File    : algorithm_fw
# @Project : tap_link_form
import cvxpy as cp


def func(link_free, y_link_flow, para_a, para_b, link_capacity, y_step):
    objective2 = cp.Minimize(cp.sum(link_free * y_link_flow) + cp.sum(
        link_free * para_a / (para_b + 1) * link_capacity * cp.power(y_link_flow / link_capacity, para_b + 1)))
    constraints2 = [y_step >= 0, y_step <= 1]
    cp.Problem(objective=objective2, constraints=constraints2).solve()
    return objective2.value


def algorithm_line_search(odlink_flow, odlink_direction, od_odlink_relation, link_free, para_a, para_b, link_capacity):
    y_step = cp.Variable(1)
    y_odlink_flow = odlink_flow + y_step * odlink_direction
    y_link_flow = od_odlink_relation @ y_odlink_flow
    objective_value = func(link_free, y_link_flow, para_a, para_b, link_capacity, y_step)
    return odlink_flow + y_step.value * odlink_direction, objective_value


def algorithm_line_search_link(link_flow, link_direction, link_free, para_a, para_b, link_capacity):
    y_step = cp.Variable(1)
    y_link_flow = link_flow + y_step * link_direction
    objective_value = func(link_free, y_link_flow, para_a, para_b, link_capacity, y_step)
    return link_flow + y_step.value * link_direction, objective_value
