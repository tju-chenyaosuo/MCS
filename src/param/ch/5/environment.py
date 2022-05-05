# -*- encoding=utf-8 -*-
import time
from main_configure_approach import CsmithConfigurationConf, EnvironmentConf, MainProcessConf
from csmith_configure import CsmithConfiguration
from program_feature import ProgramFeature
from test_base_funs import TestResult, HicondSrcGenerateStatus
import common_base_funs
import test_base_funs
import numpy as np
import copy

class Environment:
    def __init__(self):
        # Each environment contains a Csmith configuration.
        # In initialization, this value should be Csmith Default Configuration
        self.iteration_cnt = 0
        self.base_itm_single, self.base_itm_group = CsmithConfiguration.default_conf_f()

        self.s_dim = 71
        self.action_single = EnvironmentConf.action_single
        self.action_group = EnvironmentConf.action_group

        # Test variables
        self.n_thread = EnvironmentConf.n_thread
        self.program_cnt = EnvironmentConf.program_cnt
        self.timeout_percentages = EnvironmentConf.timeout_percentages

        # Evaluate variables
        self.fv_history = []
        self.fail_fv_history = []
        # Ga variables
        self.fail_cfg_history = []
        self.fail_cfg_history_cnt = []
        self.children = []

        self.fail_cfg_history_cnt = [1] * (len(CsmithConfigurationConf.hicond_confs) + 1)

        self.n_single_agent = EnvironmentConf.n_single_agent
        self.n_group_agent = EnvironmentConf.n_group_agent

        self.untune_feature_idxs = []

        self.n_agent = self.n_group_agent + self.n_single_agent
        self.configuration_quality_history = []

    def env_vector(self):
        return CsmithConfiguration.get_vector(self.base_itm_single, self.base_itm_group)

    def action_mapping(self, action_ids):
        assert len(action_ids) == self.n_agent
        assert len(action_ids) == len(EnvironmentConf.action_mapping)
        actions = [self.action_single[action_ids[inx].item()]
                   if inx < self.n_single_agent else self.action_group[action_ids[inx].item()]
                   for inx in range(len(action_ids))]
        actions_71 = [0 for _ in range(self.s_dim)]
        for _ in range(len(actions)):
            actions_71[EnvironmentConf.action_mapping[_]] = actions[_]
        return actions_71

    @staticmethod
    def tune_group_overflow(probability_values):
        bias = 0
        overflow = sum(probability_values) - 100
        while overflow > 0:
            pos = (overflow + bias) % len(probability_values)
            pos = int(pos)
            if probability_values[pos] > 1:
                probability_values[pos] -= 1
                overflow -= 1
            else:  # sum > 100 but val[pos]-1 = 0, pos -> pos+1
                bias -= 1

    @staticmethod
    def tune_group_underflow(probability_values, fix_options_idxs):
        bias = 0
        overflow = sum(probability_values) - 100
        while overflow < 0:
            pos = (overflow + bias) % len(probability_values)
            pos = int(pos)
            if pos not in fix_options_idxs:
                probability_values[pos] += 1
                overflow += 1  # sum > 100 and val[pos]-1 >0 , from pos minus 1 by turn
            else:
                bias += 1

    @staticmethod
    def tune_group_configure(values, actions, fixed_options_indexes):
        values_inx = [[values[inx], inx] for inx in range(len(values))]
        sorted_values_inx = sorted(values_inx)
        unordered_probability_values = [int(sorted_values_inx[inx][0]) if inx == 0 else
                                        int(sorted_values_inx[inx][0]-sorted_values_inx[inx-1][0])
                                        for inx in range(len(sorted_values_inx))]
        probability_indexes = [v_i[1] for v_i in sorted_values_inx]
        probability_values = [-1 for _ in probability_indexes]
        for inx in range(len(probability_indexes)):
            probability_values[probability_indexes[inx]] = unordered_probability_values[inx]
        assert len([1 for _ in probability_values if _ == -1]) == 0

        for inx in range(len(probability_values)):
            if inx not in fixed_options_indexes:
                probability_values[inx] = np.clip(probability_values[inx] + actions[inx], 1, 100)
        Environment.tune_group_overflow(probability_values)
        Environment.tune_group_underflow(probability_values, fixed_options_indexes)

        # convert group probability to csmith values
        unordered_probability_values = [probability_values[_] for _ in probability_indexes]
        unordered_probability_values = [unordered_probability_values[0] if inx == 0 else int(sum(unordered_probability_values[:inx+1]))
                                        for inx in range(len(probability_indexes))]
        values = [-1 for _ in values]
        for inx in range(len(probability_indexes)):
            values[probability_indexes[inx]] = unordered_probability_values[inx]
        assert len([1 for _ in values if _ == -1]) == 0
        return values

    def init_configure_via_list(self, values):
        self.base_itm_single, self.base_itm_group = CsmithConfiguration.get_items(values)

    def tune_configure(self, actions):
        # Get tuned single configuration values
        single_vals = [np.clip(self.base_itm_single[CsmithConfigurationConf.single_keys[_]] + actions[_], 0, 100)
                       for _ in range(len(CsmithConfigurationConf.single_keys))]

        # Get fixed option indexes in each group
        fix_option_indexes_in_group = {group_name: [] for group_name in CsmithConfigurationConf.group_keys}
        for option in CsmithConfigurationConf.untune_options:
            if ':' in option:
                option_info = option.split(':')
                group_name = option_info[0]
                option_name = option_info[1]
                fix_option_indexes_in_group[group_name].append(CsmithConfigurationConf.group_items_keys[group_name].index(option_name))

        # Get tuned group configuration values
        group_vals = []
        start = len(single_vals)
        for group_name in CsmithConfigurationConf.group_keys:
            end = start + len(CsmithConfigurationConf.group_items_keys[group_name])
            values = [self.base_itm_group[group_name][option_name] for option_name in CsmithConfigurationConf.group_items_keys[group_name]]
            values = Environment.tune_group_configure(values, actions[start:end], fix_option_indexes_in_group[group_name])
            start = end
            group_vals.extend(values)

        # change configuration via value list
        values = single_vals + group_vals
        self.init_configure_via_list(values)

    def do_test(self):
        # test_info=[program_id, feature, res, time_spt]
        # Do test
        test_info, timeout_cnt = test_base_funs.multi_process_test_via_conf(self.iteration_cnt)
        test_info_sorted = sorted([[int(test_info[_][0].split('-')[1]), test_info[_][1], test_info[_][2], test_info[_][3], _]
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
        assert len(features) == len(labels)
        fail_features = [features[_] for _ in range(len(labels)) if labels[_] == 1]
        do_reset = timeout_cnt >= EnvironmentConf.program_cnt * EnvironmentConf.timeout_percentages
        no_additional_elements = timeout_cnt < EnvironmentConf.program_cnt * EnvironmentConf.timeout_percentages + 1
        assert len(test_info) == EnvironmentConf.program_cnt or (do_reset and no_additional_elements), str(len(test_info)) + ':' + str(timeout_cnt)
        config_run_time = sum([t[3] for t in test_info])

        if len(features) == 0:
            end_time = time.time()
            calculate_time = end_time - start_time
            return np.nan, [], True, [], config_run_time + calculate_time
        else:
            actual_fail_percent = len([vl for vl in labels if vl == 1]) / len(test_info)
            end_time = time.time()
            calculate_time = end_time - start_time

            return actual_fail_percent, list(
                np.mean(np.array(features), axis=0)), do_reset, fail_features, config_run_time + calculate_time

    def get_buggy_reward(self, fail_features):
        return EnvironmentConf.buggy_reward * len(fail_features)

    def update_qt_data(self, avg_feature, fail_features):
        self.fv_history.append(avg_feature)
        self.fail_fv_history.extend(fail_features)

    def get_boundary_punish(self):
        punish = 0
        vec = self.env_vector()
        boundary_option_num = sum([1 for _ in vec if
                                   _ > 100 - EnvironmentConf.boundary_threshold or _ < EnvironmentConf.boundary_threshold])
        if boundary_option_num > EnvironmentConf.boundary_num_threshold:
            punish = EnvironmentConf.boundary_punish
        return punish

    def get_k_pre_mean_quality(self, k):
        if len(self.configuration_quality_history) == 0:
            return 0
        truncate = 0 if k > len(self.configuration_quality_history) else -k
        calculate_qualities = self.configuration_quality_history[truncate:]
        return sum(calculate_qualities) / len(calculate_qualities)

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

    def score(self):
        # write conf
        start_time = time.time()
        conf_file = MainProcessConf.csmith_conf_file_prefix + str(self.iteration_cnt)
        common_base_funs.rm_file(conf_file)
        CsmithConfiguration.write_conf(conf_file, self.base_itm_single, self.base_itm_group)
        end_time = time.time()
        write_conf_time = end_time - start_time

        # do test
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

        total_time = write_conf_time + do_test_time + evaluate_time

        log_time_content = str(self.iteration_cnt) + '-evaluate_time:' + str(evaluate_time) + '\n'
        log_time_content += str(self.iteration_cnt) + '-score_time:' + str(total_time) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_content)

        return rewards, if_reset, actual_fail_percent, total_time

    def step(self, action_ids):
        # Mapping actions
        self.iteration_cnt += 1
        start_time = time.time()
        actions = self.action_mapping(action_ids)
        end_time = time.time()
        action_mapping_time = end_time - start_time

        log_mapped_action_content = str(self.iteration_cnt) + '-mapped_actions:' + str(actions) + '\n'
        common_base_funs.log(MainProcessConf.log_action_ids, log_mapped_action_content)
        log_time_mapping_action = str(self.iteration_cnt) + '-mapping_aciton_time:' + str(action_mapping_time) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_mapping_action)

        # tune configuration
        start_time = time.time()
        self.tune_configure(actions)
        end_time = time.time()
        tune_configure_time = end_time - start_time

        log_time_tune = str(self.iteration_cnt) + '-tune_configuration_time:' + str(tune_configure_time) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_tune)

        # Do test and evaluation.
        rewards, reset, actual_fail_percent, score_time = self.score()

        vals = self.env_vector()

        total_time = action_mapping_time + tune_configure_time + score_time
        return np.array(vals), rewards, reset, total_time, {}

    def constrain_check(self):
        no_action = [0] * 71
        self.tune_configure(no_action)

    def reset_sample_default_hicond(self):
        probs = [1 / self.fail_cfg_history_cnt[_] for _ in range(len(self.fail_cfg_history_cnt)) if
                 _ <= len(CsmithConfigurationConf.hicond_confs)]
        sum_prob = sum(probs)
        probs = [_ / sum_prob for _ in probs]
        config_idx = np.random.choice(len(CsmithConfigurationConf.hicond_confs) + 1, size=1, replace=False, p=probs)[0]
        if config_idx == 0:  # default reset
            self.base_itm_single, self.base_itm_group = CsmithConfiguration.gen_default_conf(
                MainProcessConf.csmith_tmp_conf)
            common_base_funs.rm_file(MainProcessConf.csmith_tmp_conf)
        else:  # hicond reset
            single_items, group_items = CsmithConfiguration.load_conf(
                CsmithConfigurationConf.hicond_confs[config_idx - 1])
            self.init_configure_via_list(CsmithConfiguration.get_vector(single_items, group_items))

    def reset(self):
        # reset
        old_itm_single = copy.deepcopy(self.base_itm_single)
        old_itm_group = copy.deepcopy(self.base_itm_group)
        self.reset_sample_default_hicond()

        # untune options
        for op in CsmithConfigurationConf.untune_options:
            if ':' not in op:
                self.base_itm_single[op] = old_itm_single[op]
            else:
                g_name = op.split(':')[0]
                ele_name = op.split(':')[1]
                self.base_itm_group[g_name][ele_name] = old_itm_group[g_name][ele_name]
        self.constrain_check()

        return self.env_vector()


if __name__ == '__main__':
    # import copy
    # # test deep copy
    # dict1 = {'a': {'a1': 1, 'a2': 2, 'a3': 3}}
    # dict2 = copy.deepcopy(dict1)
    # dict1['a']['a2'] = 100
    # print(dict2)
    pass

