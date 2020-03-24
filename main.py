# @Time    : 2020/3/22 17:12
# @Author  : gzzang
# @File    : main
# @Project : traffic_assignment

from re.alg_route_msa import *
from re.alg_route_default import *
from re.alg_route_vi_2 import *

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

network_name='nguyen_dupuis'


# alg_link_cg(network_name, bool_display_iteration=False)
# alg_odlink_default(network_name)
# alg_link_fw_sp(network_name, bool_display_iteration=False)
# alg_odlink_fw_optimization(network_name, bool_display_iteration=False)
# alg_odlink_fw_sp(network_name, bool_display_iteration=False)
alg_route_vi(network_name, bool_display_iteration=False)
# alg_route_msa(network_name, bool_display=True, bool_display_iteration=False)
# alg_route_default(network_name)
