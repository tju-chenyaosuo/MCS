import os
import common_base_funs

baseline_types = ['random', 'default', 'hicond']
baseline_type = baseline_types[2]

# common baselines configure
run_time_limit = 10 * 24 * 60 * 60
n_thread = 10
program_max_num = 40 * 10000
test_dir = None
if baseline_type == baseline_types[0]:    # random
    test_dir = common_base_funs.add_sep('random_baseline_test')
if baseline_type == baseline_types[1]:    # default
    test_dir = common_base_funs.add_sep('default_baseline_test')
if baseline_type == baseline_types[2]:    # hicond
    test_dir = common_base_funs.add_sep('hicond_baseline_test')
assert test_dir is not None

use_seed = True

log_prefix = None
if baseline_type == baseline_types[0]:    # random
    log_prefix = 'log/random_p'
if baseline_type == baseline_types[1]:    # default
    log_prefix = 'log/default_p'
if baseline_type == baseline_types[2]:    # hicond
    log_prefix = 'log/hicond_p'
assert log_prefix is not None

# hicond
if baseline_type == baseline_types[2]:
    hicond_conf_dir = common_base_funs.add_sep('hicond-conf')
    hicond_confs = [hicond_conf_dir + 'config' + str(_) for _ in range(1, 11)]
