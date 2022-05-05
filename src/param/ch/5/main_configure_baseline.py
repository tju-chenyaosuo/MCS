import common_base_funs

baseline_types = ['swarm', 'hicond']
baseline_type = baseline_types[0]

# common baselines configure
run_time_limit = 10 * 24 * 60 * 60
n_thread = 10
program_max_num = 40 * 10000
test_dir = None
if baseline_type == baseline_types[0]:    # swarm
    test_dir = common_base_funs.add_sep('swarm_baseline_test')
if baseline_type == baseline_types[1]:    # hicond
    test_dir = common_base_funs.add_sep('hicond_baseline_test')
assert test_dir is not None

use_seed = True

log_prefix = None
if baseline_type == baseline_types[0]:    # swarm
    log_prefix = 'log/swarm_p'
if baseline_type == baseline_types[1]:    # hicond
    log_prefix = 'log/hicond_p'
assert log_prefix is not None

# hicond
if baseline_type == baseline_types[1]:
    hicond_conf_dir = common_base_funs.add_sep('hicond-conf')
    hicond_confs = [hicond_conf_dir + 'config' + str(_) for _ in range(1, 11)]
