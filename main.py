# @Time    : 2020/3/22 17:12
# @Author  : gzzang
# @File    : main
# @Project : traffic_assignment

from alg_link_cg import *
from alg_link_fw_sp import *
from alg_odlink_default import *
from alg_odlink_fw_optimization import *
from alg_odlink_fw_sp import *
from alg_route_msa import *
from alg_route_default import *


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

alg_route_msa('simple_network', bool_display=True, bool_display_iteration=False)
alg_route_default('simple_network')
alg_link_cg('simple_network', bool_display_iteration=False)
alg_odlink_default('simple_network')
alg_link_fw_sp('simple_network', bool_display_iteration=False)
alg_odlink_fw_optimization('simple_network', bool_display_iteration=False)
alg_odlink_fw_sp('simple_network', bool_display_iteration=False)
