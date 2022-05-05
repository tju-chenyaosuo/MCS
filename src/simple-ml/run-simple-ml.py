import os

import time
import numpy as np
import random
import xgboost as xgb
import test_base_funs
import common_base_funs
import main_configure_baseline
from csmith_configure import CsmithConfiguration
from main_configure_approach import *
from test_base_funs import HicondSrcGenerateStatus, TestResult
from program_feature import ProgramFeature
import main_configure_ml

"""
All related configurations:
    + main_configure_approach.EnvironmentConf.program_cnt
    + main_configure_approach.EnvironmentConf.n_thread
    + main_configure_approach.EnvironmentConf.timeout_percentages
    + main_configure_approach.EnvironmentConf.enable_all_program_feature
    + main_configure_approach.MainProcessConf.test_compiler_type
    + main_configure_approach.MainProcessConf.test_dir_prefix
    + main_configure_approach.MainProcessConf.use_seed
    + main_configure_approach.MainProcessConf.seeds
    + main_configure_approach.MainProcessConf.csmith_conf_file_prefix
    + main_configure_approach.MainProcessConf.compiler_types
    + main_configure_approach.GCCTestConf.csmith_lib
    + main_configure_approach.LLVMTestConf.csmith_lib
    + main_configure_approach.A2cConfigure.update_iterations
    
    + main_configure_ml [all the configurations]
    
"""


class MlEnvironment:
    """
    Collect all the information about testing.(including do testing)
    """
    def __init__(self):
        self.iteration_cnt = 0
        self.fv_history = []
        if EnvironmentConf.enable_all_program_feature:
            self.untune_feature_idxs = []
        else:
            self.untune_feature_idxs = ProgramFeatureConf.default_discount_feature_inx + \
                                       ProgramFeatureConf.untune_discount_feature_inx
        self.n_single_agent = EnvironmentConf.n_single_agent
        self.n_group_agent = EnvironmentConf.n_group_agent
        self.n_agent = self.n_group_agent + self.n_single_agent
        self.configuration_quality_history = []
        self.base_itm_single, self.base_itm_group = CsmithConfiguration.default_conf_f()
        pass

    def predict_score(self, xgb_clf, feature):
        x = np.array(feature)
        res = xgb_clf.predict(x)

        print(res)

        return [[res[i], i] for i in range(len(res))]

    def recommend_conf(self, model, ml_features, configuration_paths):
        TOP_N = 1
        predict_scores = self.predict_score(model, ml_features)
        predict_scores = sorted(predict_scores, reverse=True)
        predict_scores = [predict_scores[_] for _ in range(TOP_N)]
        configuration_path = [configuration_paths[predict_scores[_][1]] for _ in range(TOP_N)]
        return configuration_path, predict_scores

    def gen_conf(self, model):
        """
        This function will generate Csmith configuration in ${MainProcessConf.csmith_conf_file_prefix + str(self.iteration_cnt)}
        Take care about parameter "model"!!!
        """
        t1 = time.time()
        if model is None:
            conf_file = MainProcessConf.csmith_conf_file_prefix + str(self.iteration_cnt)
            seed = str(random.randint(0, 80000000))
            self.base_itm_single, self.base_itm_group = CsmithConfiguration.gen_random_conf(conf_file, seed)
            t2 = time.time()
        # a non-random step
        else:
            # randomly generate xxxx configurations
            configuration_features = []
            configuration_paths = []
            log_content_seeds = []
            for _ in range(main_configure_ml.candidate_configuration_num):
                print('genconf: ' + str(_))
                conf_file = main_configure_ml.tmp_csmith_configuration_prefix + str(_)
                common_base_funs.rm_file(conf_file)
                seed = str(random.randint(0, 80000000))
                base_itm_single, base_itm_group = CsmithConfiguration.gen_random_conf(conf_file, seed)
                configuration_features.append(CsmithConfiguration.get_vector(base_itm_single, base_itm_group))
                configuration_paths.append(conf_file)
                log_content_seeds.append(seed)
            log_content = str(self.iteration_cnt) + '-seeds: ' + ','.join(log_content_seeds) + '\n'
            common_base_funs.log(main_configure_ml.log_tmp_conf_seed, log_content)

            # predict
            configuration_path, predict_prob = self.recommend_conf(model, np.array(configuration_features), configuration_paths)

            # select the best one
            configuration_path = configuration_path[0]
            predict_prob = predict_prob[0]
            log_content = str(self.iteration_cnt) + '-predict: ' + str(predict_prob) + '\n'
            common_base_funs.log(main_configure_ml.log_conf_predict, log_content)

            # prepare for the testing and evaluation
            target_conf = MainProcessConf.csmith_conf_file_prefix + str(self.iteration_cnt)
            cp_cmd = ' '.join(['cp', configuration_path, target_conf])
            common_base_funs.execmd(cp_cmd)
            self.base_itm_single, self.base_itm_group = CsmithConfiguration.load_conf(target_conf)
            t2 = time.time()

        t = t2 - t1
        time_log_content = str(self.iteration_cnt) + '-gen conf time: ' + str(t) + '\n'
        common_base_funs.log(MainProcessConf.log_time, time_log_content)

        return t

    def step(self, model=None):
        self.iteration_cnt += 1
        # a random step
        gen_conf_time = self.gen_conf(model)
        # get socre
        rewards, if_reset, actual_fail_percent, total_time = self.score()
        return rewards, if_reset, actual_fail_percent, total_time + gen_conf_time

    def do_test(self):
        """
        This function is changed a lot comparing to old one.
        Because the heavy implementation of multi-threading in previous code,
            I reduced the code size and make control flow clearer.
        :return:
        """
        # test_info=[program_id, feature, res, time_spt, seed]
        # Do test
        test_info, timeout_cnt = test_base_funs.multi_process_test_via_conf(self.iteration_cnt)

        log_test_info_content = str(self.iteration_cnt) + '-test_len:' + str(len(test_info)) + '\n'
        common_base_funs.log(MainProcessConf.log_test_info, log_test_info_content)
        test_info_sorted = sorted(
            [[int(test_info[_][0].split('-')[1]), test_info[_][1], test_info[_][2], test_info[_][3], _]
             for _ in range(len(test_info))], reverse=False)
        log_test_info_content = []
        log_time_content = []
        for _ in range(len(test_info)):
            test_info_detail = test_info[test_info_sorted[_][-1]]
            program_id = test_info_detail[0]
            feature = test_info_detail[1]
            result = test_info_detail[2]
            time_spt = test_info_detail[3]
            seed = test_info_detail[4]
            log_test_info_content.append(','.join([program_id + '-result:' + result, program_id + '-seed:' + seed,
                                                   program_id + '-feature:' + str(feature)]))
            log_time_content.append('program-' + program_id + '-time:' + str(time_spt))
        common_base_funs.log(MainProcessConf.log_time, '\n'.join(log_time_content) + '\n')
        common_base_funs.log(MainProcessConf.log_detailed_test_info, '\n'.join(log_test_info_content) + '\n')

        # Calculate: features, labels, fail_features, avg_predict_prob, avg_feature...
        start_time = time.time()
        features = [t[1] for t in test_info if len(t[1]) != 0]
        labels = []
        for _ in test_info:
            l = None
            if _[2] in [HicondSrcGenerateStatus.timeout, HicondSrcGenerateStatus.crash]:
                continue
            if _[2] in TestResult.invalid_results:
                assert l is None
                l = -1
            if _[2] in TestResult.normal_results:
                assert l is None
                l = 0
            if _[2] in TestResult.bug_results:
                assert l is None
                l = 1
            assert l is not None
            labels.append(l)
        # program crash will be omit when using this code.
        # labels = [0 if t[2] == TestState.success else -1 if 'timeout' in t[2] or 'undefined' in t[2] or 'program' in t[2] else 1 for t in test_info if len(t[1]) != 0]
        assert len(features) == len(labels)
        fail_features = [features[_] for _ in range(len(labels)) if labels[_] == 1]
        do_reset = timeout_cnt >= EnvironmentConf.program_cnt * EnvironmentConf.timeout_percentages
        no_additional_elements = timeout_cnt < EnvironmentConf.program_cnt * EnvironmentConf.timeout_percentages + 1
        assert len(test_info) == EnvironmentConf.program_cnt or (do_reset and no_additional_elements), str(
            len(test_info)) + ':' + str(timeout_cnt)
        config_run_time = sum([t[3] for t in test_info])

        if len(features) == 0:
            end_time = time.time()
            calculate_time = end_time - start_time
            return np.nan, [], True, [], config_run_time + calculate_time
        else:
            actual_fail_percent = len([vl for vl in labels if vl == 1]) / len(test_info)
            end_time = time.time()
            calculate_time = end_time - start_time

            return actual_fail_percent, np.mean(np.array(features), axis=0), do_reset, fail_features, config_run_time + calculate_time

    def score(self):
        # predict timeout and do test
        actual_fail_percent = 0
        avg_feature = []
        if_reset = True
        fail_features = []
        do_test_time = 0
        if (EnvironmentConf.use_timeout_model and not self.predict_timeout()) or not EnvironmentConf.use_timeout_model:
            actual_fail_percent, avg_feature, if_reset, fail_features, do_test_time = self.do_test()
            if EnvironmentConf.feature_standardization and len(avg_feature) != 0:
                avg_feature = (avg_feature-EnvironmentConf.standardization_mu) / EnvironmentConf.standardization_std

        log_detailed_test_info = str(self.iteration_cnt) + '-avg_feature:' + str(avg_feature) + '\n'
        log_detailed_test_info += str(self.iteration_cnt) + '-fails:' + str([len(f) for f in fail_features]) + '\n'
        log_detailed_test_info += str(self.iteration_cnt) + '-reset:' + str(if_reset) + '\n'
        log_detailed_test_info += str(self.iteration_cnt) + '-actual_fail_percent:' + str(actual_fail_percent) + '\n'
        common_base_funs.log(MainProcessConf.log_detailed_test_info, log_detailed_test_info)

        # evaluate environment
        rewards, evaluate_time = self.evaluate_step(avg_feature, if_reset, fail_features)

        total_time = do_test_time + evaluate_time

        log_time_content = str(self.iteration_cnt) + '-evaluate_time:' + str(evaluate_time) + '\n'
        log_time_content += str(self.iteration_cnt) + '-score_time:' + str(total_time) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_content)

        return rewards, if_reset, actual_fail_percent, total_time

    def get_configuration_quality(self, avg_feature):
        diversity_quality = ProgramFeature.k_cosine_distance(x=avg_feature, y=self.fv_history, default_val=0,
                                                             untune_feature_inx=self.untune_feature_idxs,
                                                             k=EnvironmentConf.diversity_k,
                                                             compensation_factor=EnvironmentConf.diversity_compensation_factor)
        quality_component = [diversity_quality]
        assert len(quality_component) == len(EnvironmentConf.quality_weight)
        quality = sum(
            [quality_component[_] * EnvironmentConf.quality_weight[_] for _ in range(len(quality_component))])
        return quality

    def get_k_pre_mean_quality(self, k):
        if len(self.configuration_quality_history) == 0:
            return 0
        truncate = 0 if k > len(self.configuration_quality_history) else -k
        calculate_qualities = self.configuration_quality_history[truncate:]
        return sum(calculate_qualities) / len(calculate_qualities)

    def get_buggy_reward(self, fail_features):
        return EnvironmentConf.buggy_reward * len(fail_features)

    def get_boundary_punish(self):
        punish = 0
        vec = self.env_vector()
        boundary_option_num = sum([1 for _ in vec if
                                   _ > 100 - EnvironmentConf.boundary_threshold or _ < EnvironmentConf.boundary_threshold])
        if boundary_option_num > EnvironmentConf.boundary_num_threshold:
            punish = EnvironmentConf.boundary_punish
        return punish

    def get_delta_configuration_quality(self, avg_feature, fail_features, reset):
        if reset:
            if len(avg_feature) != 0:
                self.fv_history.append(avg_feature)
            return EnvironmentConf.reset_reward, 0
        if self.iteration_cnt == 1:
            quality = self.get_configuration_quality(avg_feature)
            self.fv_history.append(avg_feature)
            self.configuration_quality_history.append(quality)
            return 0, 0
        # get quality
        t = time.time()
        quality = self.get_configuration_quality(avg_feature)
        # get delta quality
        delta_quality = quality - self.get_k_pre_mean_quality(EnvironmentConf.dv_avg_n)
        # add to history
        self.fv_history.append(avg_feature)
        self.configuration_quality_history.append(quality)
        # calculate buggy and boundary reward
        buggy_reward = self.get_buggy_reward(fail_features)
        # if buggy_reward != 0:
        #     print('catch it!')
        #     exit(1)
        boundary_punish = self.get_boundary_punish()
        delta_quality += buggy_reward
        delta_quality += boundary_punish
        t = time.time() - t
        # log
        reward_component = [quality, delta_quality, buggy_reward, boundary_punish]
        content = str(self.iteration_cnt)+'-reward_component:'+str(reward_component)
        common_base_funs.log(MainProcessConf.log_reward, content+'\n')
        # reward rate
        return delta_quality * EnvironmentConf.delta_rate, t

    def evaluate_step(self, avg_feature, reset, fail_features):
        rated_delta_quality, time_cost = self.get_delta_configuration_quality(avg_feature, fail_features, reset)
        rewards = [rated_delta_quality] * self.n_agent
        return rewards, time_cost

    def env_vector(self):
        """
        This function has been tested
        :return:
        """
        return CsmithConfiguration.get_vector(self.base_itm_single, self.base_itm_group)

    def predict_timeout(self):
        """
        Deprecated function in MCS, originally used to predict whether current configuration is timeout.
        Now we do not do such a prediction.
        """
        return False


def run_ml():
    # initialize
    ml_env = MlEnvironment()
    time_spt = 0
    buffer_r = []
    buffer_s = []
    model = xgb.XGBRegressor()    # default params

    # random setup
    for _ in range(A2cConfigure.update_iterations):
        rewards, if_reset, actual_fail_percent, total_time = ml_env.step(model=None)
        s = ml_env.env_vector()
        r = rewards[0]
        buffer_s.append(s)
        buffer_r.append(r)
        log_content = str(ml_env.iteration_cnt) + '-s: ' + str(s) + '\n'
        log_content += str(ml_env.iteration_cnt) + '-r: ' + str(r) + '\n'
        common_base_funs.log(main_configure_ml.log_state_reward, log_content)
        time_spt += total_time

    # fit the model
    train_start_time = time.time()
    model = model.fit(np.array(buffer_s), np.array(buffer_r))
    train_end_time = time.time()
    time_spt += train_end_time - train_start_time
    buffer_s = []
    buffer_r = []
    initialize_train_log_content = 'initialize train time: ' + str(time_spt) + '\n'
    common_base_funs.log(MainProcessConf.log_time, initialize_train_log_content)

    # start retrain process
    iterations_2_update = A2cConfigure.update_iterations
    while time_spt <= MainProcessConf.run_time_limit:
        rewards, if_reset, actual_fail_percent, total_time = ml_env.step(model=model)
        time_spt += total_time
        s = ml_env.env_vector()
        r = rewards[0]
        buffer_s.append(s)
        buffer_r.append(r)

        log_content = str(ml_env.iteration_cnt) + '-s: ' + str(s) + '\n'
        log_content += str(ml_env.iteration_cnt) + '-r: ' + str(r) + '\n'
        common_base_funs.log(main_configure_ml.log_state_reward, log_content)

        iterations_2_update -= 1
        if iterations_2_update == 0:
            tar_model = main_configure_ml.model_prefix + str(ml_env.iteration_cnt)
            model.save_model(tar_model)
            train_start_time = time.time()
            model = model.fit(np.array(buffer_s), np.array(buffer_r), xgb_model=model)
            train_end_time = time.time()
            time_spt += train_end_time - train_start_time

            initialize_train_log_content = str(ml_env.iteration_cnt) + '-initialize train time: ' + str(time_spt) + '\n'
            common_base_funs.log(MainProcessConf.log_time, initialize_train_log_content)

            buffer_s = []
            buffer_r = []
            iterations_2_update = A2cConfigure.update_iterations
        log_time_content = str(ml_env.iteration_cnt) + '-total_run_time:' + str(time_spt) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_content)


if __name__ == '__main__':
    assert os.path.exists(GeneralOperationConf.limit_memory_script)
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[0]:  # gcc
        assert os.path.exists(GCCTestConf.csmith_lib)
    if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[1]:  # llvm
        assert os.path.exists(LLVMTestConf.csmith_lib)
    assert os.path.exists(CsmithConfigurationConf.csmith)
    common_base_funs.mkdir_if_not_exists(A2cConfigure.model_dir)
    if MainProcessConf.use_seed:
        rl_seed = 91122391
        random.seed(rl_seed)
        np.random.seed(rl_seed)
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
    common_base_funs.mkdir_if_not_exists(MainProcessConf.log_dir_counter_action)
    common_base_funs.mkdir_if_not_exists(MainProcessConf.log_dir_config_cnt)

    common_base_funs.rm_file(main_configure_ml.tmp_csmith_conf_dir)
    common_base_funs.mkdir_if_not_exists(main_configure_ml.tmp_csmith_conf_dir)
    common_base_funs.rm_file(main_configure_ml.model_dir)
    common_base_funs.mkdir_if_not_exists(main_configure_ml.model_dir)

    run_ml()

