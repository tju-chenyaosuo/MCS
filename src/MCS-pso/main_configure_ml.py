import os
import common_base_funs
from main_configure_approach import MainProcessConf

tmp_csmith_conf_dir = common_base_funs.add_sep('tmp-csmith-dir')
tmp_csmith_configuration_prefix = tmp_csmith_conf_dir + 'config'
candidate_configuration_num = 2000

log_tmp_conf_seed = MainProcessConf.log_dir + 'sim_ml_conf_seed.txt'
log_conf_predict = MainProcessConf.log_dir + 'sim_ml_conf_predict.txt'
log_state_reward = MainProcessConf.log_dir + 'sim_ml_sr.txt'

model_dir = common_base_funs.add_sep('sim_ml_models')
model_prefix = model_dir + 'model'
