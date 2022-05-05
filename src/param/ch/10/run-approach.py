# -*- encoding=utf-8 -*-
import os
import time
import torch
import random
import numpy as np
from a2c import A2CClu

import common_base_funs
from environment import Environment
from main_configure_approach import *
from program_feature import ProgramFeature
"""
This module only do main process
"""

if __name__ == '__main__':
    # initialize all method
    assert os.path.exists(GeneralOperationConf.limit_memory_script)
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[0]:    # gcc
        assert os.path.exists(GCCTestConf.csmith_lib)
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[1]:    # llvm
        assert os.path.exists(LLVMTestConf.csmith_lib)
    assert os.path.exists(CsmithConfigurationConf.csmith)
    for _ in CsmithConfigurationConf.hicond_confs:
        assert os.path.exists(_)
    common_base_funs.mkdir_if_not_exists(A2cConfigure.model_dir)
    if MainProcessConf.use_seed:
        rl_seed = 91122391
        random.seed(rl_seed)
        np.random.seed(rl_seed)
        torch.manual_seed(rl_seed)
        assert os.path.exists(MainProcessConf.seed_file)
        MainProcessConf.seeds = common_base_funs.get_file_lines(MainProcessConf.seed_file)
    if EnvironmentConf.feature_standardization:
        assert os.path.exists(EnvironmentConf.standardization_file)
        EnvironmentConf.standardization_data = ProgramFeature.load_features(EnvironmentConf.standardization_file)
        EnvironmentConf.standardization_mu = np.mean(EnvironmentConf.standardization_data, axis=0)
        EnvironmentConf.standardization_std = np.std(EnvironmentConf.standardization_data, axis=0)
        EnvironmentConf.standardization_std = [1 if _ == 0 else _ for _ in EnvironmentConf.standardization_std]
    common_base_funs.rm_file(MainProcessConf.csmith_conf_dir)
    common_base_funs.mkdir_if_not_exists(MainProcessConf.csmith_conf_dir)
    common_base_funs.rm_file(MainProcessConf.test_dir_prefix)
    common_base_funs.mkdir_if_not_exists(MainProcessConf.test_dir_prefix)
    common_base_funs.rm_file(MainProcessConf.log_dir)
    common_base_funs.mkdir_if_not_exists(MainProcessConf.log_dir)

    clu = A2CClu()
    # tobe tested
    if MainProcessConf.retrain:
        last_model_dir = common_base_funs.get_last_dir(A2cConfigure.model_dir)
        if last_model_dir is not None:
            clu.load_nets()
    env = Environment()
    s = env.env_vector()
    s = torch.tensor(s, dtype=torch.float32)
    s = s / 100
    buffer_a = []
    buffer_r = []
    buffer_s = []
    tm_spt = 0
    iteration_cnt = 0
    model_generation_cnt = 0
    method_start_time = time.time()
    while tm_spt <= MainProcessConf.run_time_limit:
        env_log_content = str(env.iteration_cnt) + '-env:' + str(s) + '\n'
        common_base_funs.log(MainProcessConf.log_env, env_log_content)

        # Choose action
        start_time = time.time()
        action_ids = clu.choose_actions(env.iteration_cnt, s)    # choose action index
        end_time = time.time()
        choose_action_time = end_time - start_time

        action_log_content = str(env.iteration_cnt) + '-actions:' + str([a.item() for a in action_ids]) + '\n'
        time_log_content = str(env.iteration_cnt) + '-choose_actions_time:' + str(choose_action_time) + '\n'
        common_base_funs.log(MainProcessConf.log_action_ids, action_log_content)
        common_base_funs.log(MainProcessConf.log_time, time_log_content)

        # Do a step, and evaluate it.
        s_, r, reset, env_step_time, info = env.step(action_ids)

        reward_log_content = str(env.iteration_cnt) + '-reward:' + str(r) + '\n'
        time_log_content = str(env.iteration_cnt) + '-step_time:' + str(env_step_time) + '\n'
        common_base_funs.log(MainProcessConf.log_reward, reward_log_content)
        common_base_funs.log(MainProcessConf.log_time, time_log_content)

        # Save or update model
        start_time = time.time()
        s_ = torch.tensor(s_, dtype=torch.float32)
        s_ = s_ / 100
        buffer_a.append(action_ids)
        buffer_s.append(s)
        buffer_r.append(r)
        s = s_
        iteration_cnt += 1
        # save model
        save_model = model_generation_cnt % 50 == 0
        if save_model:
            clu.save_nets()
        model_generation_cnt += 1
        # update model
        update_model = reset or iteration_cnt == A2cConfigure.update_iterations
        if update_model:
            common_base_funs.log(MainProcessConf.log_reward, str(env.iteration_cnt) + '-update' + '\n')
            buffer_r = np.array(buffer_r)
            clu.update(buffer_a, buffer_s, buffer_r, s, reset)
            buffer_a = []
            buffer_r = []
            buffer_s = []
            iteration_cnt = 0
        end_time = time.time()
        model_time = end_time - start_time

        if save_model or update_model:
            log_model_content = ''
            if save_model:
                log_model_content += str(env.iteration_cnt) + '-save_model\n'
            if update_model:
                log_model_content += str(env.iteration_cnt) + '-update_model\n'
            log_time_content = str(env.iteration_cnt) + '-model_op_time:' + str(model_time) + '\n'
            common_base_funs.log(MainProcessConf.log_model, log_model_content)
            common_base_funs.log(MainProcessConf.log_time, log_time_content)

        # Do reset
        reset_time = 0
        if reset:
            start_time = time.time()
            s_ = env.reset()
            end_time = time.time()
            reset_time = end_time - start_time

            time_log_content = str(env.iteration_cnt) + '-reset_time:' + str(env_step_time) + '\n'
            common_base_funs.log(MainProcessConf.log_time, time_log_content)

        iteration_spt = choose_action_time + env_step_time + reset_time + model_time
        tm_spt += iteration_spt
        log_time_content = str(env.iteration_cnt) + '-total_time:' + str(iteration_spt) + '\n'
        log_time_content += str(env.iteration_cnt) + '-accumulate_time:' + str(tm_spt) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_content)

    method_end_time = time.time()
    log_time_content = 'total_run_time:' + str(method_end_time - method_start_time)\
                       + ',configure_time:' + str(MainProcessConf.run_time_limit) + '\n'
    common_base_funs.log(MainProcessConf.log_time, log_time_content)

