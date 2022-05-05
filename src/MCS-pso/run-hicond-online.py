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
from test_base_funs import HicondSrcGenerateStatus, TestResult, HicondSrcGenerator
from program_feature import ProgramFeature
import main_configure_ml
import main_configure_hicond_online

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

single_tune_num = CsmithConfigurationConf.tune_single_cnt
group_tune_num = CsmithConfigurationConf.tune_group_cnt

group_nums = [len(CsmithConfigurationConf.group_items_keys[_]) for _ in CsmithConfigurationConf.group_keys]
un_tune_group_nums = [1, 0, 0, 2, 0]

tune_group_nums = [group_nums[_] - un_tune_group_nums[_] for _ in range(len(group_nums))]
all_tune_group_num = sum(tune_group_nums)


class PSOSearcher:
    def __init__(self):
        self.particles = []
        self.velocities = []
        self.global_best_position = []
        self.particle_best_position = np.zeros((main_configure_hicond_online.particle_number, 71))
        self.particle_best_scores = np.zeros((main_configure_hicond_online.particle_number,))
        self.global_best_score = -1000000    # score can be a negative number
        self.iteration_cnt = 0
        self.conf_cnt = 0

        self.if_reset = [False for _ in range(main_configure_hicond_online.particle_number)]

        # initialize for all the variables in original environment class
        self.fv_history = []
        if EnvironmentConf.enable_all_program_feature:
            self.untune_feature_idxs = []
        else:
            self.untune_feature_idxs = ProgramFeatureConf.default_discount_feature_inx + \
                                       ProgramFeatureConf.untune_discount_feature_inx
        self.configuration_quality_history = []
        self.n_single_agent = EnvironmentConf.n_single_agent
        self.n_group_agent = EnvironmentConf.n_group_agent
        self.n_agent = self.n_group_agent + self.n_single_agent

        self.initialize()

    def initialize_particles(self):
        """
        Warning: this part is different from the original hicond
        Warning: the difference of original hicond and hicond-online is as following
        Warning: 1) the way of generate configuration, original hicond uses native python, while MCS uses Csmith command option
        Warning: 2) the option value of generated configuration, MCS fixes some of Csmith configuration options, while original hicond seems not.
        Warning: 3) actually, original hicond fixes more configuration options in its' probability files.
        Return a list of randomly constructed configuration vector.
        """
        conf_seed_idx = 0    # this variable is only use once in initialize stage, no need to be a field of class.
        for _ in range(main_configure_hicond_online.particle_number):
            tar_file = MainProcessConf.csmith_conf_file_prefix + '-init-' + str(_)
            while True:
                seed = MainProcessConf.seeds[conf_seed_idx]
                conf_seed_idx += 1
                base_single_itm, base_group_itm = CsmithConfiguration.gen_random_conf(tar_file, seed)
                # this is a point that may generate different experiment result, but so do original hicond.
                if PSOSearcher.test_conf(tar_file):
                    break
            feature_vec = CsmithConfiguration.get_vector(base_single_itm, base_group_itm)
            prob_vec = CsmithConfiguration.get_prob_vec_from_vec(feature_vec)
            self.particles.append(np.array(prob_vec))

    def initialize_particle(self, idx):
        tar_file = MainProcessConf.csmith_conf_file_prefix + '-init-' + str(idx)
        while True:
            seed = str(random.randint(0, 80000000))
            base_single_itm, base_group_itm = CsmithConfiguration.gen_random_conf(tar_file, seed)
            # this is a point that may generate different experiment result, but so do original hicond.
            if PSOSearcher.test_conf(tar_file):
                break
        feature_vec = CsmithConfiguration.get_vector(base_single_itm, base_group_itm)
        prob_vec = CsmithConfiguration.get_prob_vec_from_vec(feature_vec)
        self.particles[idx] = np.array(prob_vec)

    def initialize(self):
        # initilization of pset
        self.initialize_particles()
        # initilization of vset
        self.velocities = np.array([[int(random.uniform(main_configure_hicond_online.velocity_limitations[_][0], main_configure_hicond_online.velocity_limitations[_][1]))
                                     if __ not in EnvironmentConf.un_tune_indexes else 0
                                     for __ in range(len(self.particles[0]))]
                                    for _ in range(main_configure_hicond_online.particle_number)])
        # print('test initialize')
        # output_particle_info = [[str(__) for __ in _] for _ in self.particles]
        # output_particle_info = '\n'.join([','.join(_) for _ in output_particle_info])
        # print('initialize particles: \n' + output_particle_info)
        # output_velocities_info = [[str(__) for __ in _] for _ in self.velocities]
        # output_velocities_info = '\n'.join([','.join(_) for _ in output_velocities_info])
        # print('initialize velocities: \n' + output_velocities_info)

    @staticmethod
    def test_conf(conf_file):
        """
        For testing the generated Csmith configuration, and is usually called after update or random construct particles.
        Some of generate Configuration lead Csmith to crash when generating test programs.
        This configuration should at least generate one valid program.
        This function directly comes from original hicond.
        (make sure that our experiment uses same strategy as original hicond)
        """
        feature_file = main_configure_hicond_online.tmp_test_program_dir + 'a.csv'
        src = main_configure_hicond_online.tmp_test_program_dir + 'a.c'
        hsg = HicondSrcGenerator(src=src, feature_file=feature_file, conf=conf_file)
        while True:
            res, t = hsg.gen_src()
            if res == HicondSrcGenerateStatus.crash:
                return False
            if res == HicondSrcGenerateStatus.success:
                return True
            if res == HicondSrcGenerateStatus.timeout:
                continue

    def convert_configurations(self):
        """
        Convert all of particles into Csmith configurations
        We do not test configuration in this function
        When this function is called, we assume all particles have been checked through 'test_conf'
        """
        time_spt = []
        self.iteration_cnt += 1
        assert self.conf_cnt == (self.iteration_cnt - 1) * main_configure_hicond_online.particle_number
        for par in self.particles:

            gen_start_time = time.time()

            feature_vec = CsmithConfiguration.get_vec_from_prob(par)
            base_single_itm, base_group_itm = CsmithConfiguration.get_items(feature_vec)
            tar_file = MainProcessConf.csmith_conf_file_prefix + str(self.conf_cnt)
            CsmithConfiguration.write_conf(tar_file, base_single_itm, base_group_itm)
            self.conf_cnt += 1

            gen_end_time = time.time()
            time_spt.append(gen_end_time - gen_start_time)

            log_env_content = str(self.iteration_cnt) + '-' + str(self.conf_cnt) + '-particle(prob): ' + str(par.tolist()) + '\n'
            log_env_content += str(self.iteration_cnt) + '-' + str(self.conf_cnt) + '-feature(value): ' + str(feature_vec)
            common_base_funs.log(MainProcessConf.log_env, log_env_content + '\n')
        return sum(time_spt)

    def do_test(self, config_id):
        """
        This function is changed a lot comparing to old one.
        Because the heavy implementation of multi-threading in previous code,
            I reduced the code size and make control flow clearer.
        :return:
        """
        # test_info=[program_id, feature, res, time_spt, seed]
        # Do test
        test_info, timeout_cnt = test_base_funs.multi_process_test_via_conf(config_id)

        log_test_info_content = str(config_id) + '-config-test_len:' + str(len(test_info)) + '\n'
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

            return actual_fail_percent, np.mean(np.array(features),
                                                axis=0), do_reset, fail_features, config_run_time + calculate_time

    def score(self):
        # predict timeout and do test
        actual_fail_percents = []
        avg_features = []
        if_resets = []
        fail_features_list = []
        do_total_time = 0

        # if (EnvironmentConf.use_timeout_model and not self.predict_timeout()) or not EnvironmentConf.use_timeout_model:
        for config_id in range((self.iteration_cnt - 1) * 10, self.iteration_cnt * 10, 1):

            actual_fail_percent, avg_feature, if_reset, fail_features, do_test_time = self.do_test(config_id)
            do_total_time += do_test_time
            if EnvironmentConf.feature_standardization and len(avg_feature) != 0:
                avg_feature = (avg_feature - EnvironmentConf.standardization_mu) / EnvironmentConf.standardization_std

            actual_fail_percents.append(actual_fail_percent)
            avg_features.append(avg_feature)
            if_resets.append(if_reset)
            fail_features_list.append(fail_features)

            log_detailed_test_info = str(config_id) + '-config-avg_feature:' + str(avg_feature) + '\n'
            log_detailed_test_info += str(config_id) + '-config-fails:' + str([len(f) for f in fail_features]) + '\n'
            log_detailed_test_info += str(config_id) + '-config-reset:' + str(if_reset) + '\n'
            log_detailed_test_info += str(config_id) + '-config-actual_fail_percent:' + str(actual_fail_percent) + '\n'
            common_base_funs.log(MainProcessConf.log_detailed_test_info, log_detailed_test_info)

        # evaluate environment
        rewards, evaluate_time = self.evaluate_step(avg_features, if_resets, fail_features_list)

        total_time = do_total_time + evaluate_time

        log_time_content = str(self.iteration_cnt) + '-evaluate_time:' + str(evaluate_time) + '\n'
        log_time_content += str(self.iteration_cnt) + '-score_time:' + str(total_time) + '\n'
        common_base_funs.log(MainProcessConf.log_time, log_time_content)

        return rewards, if_resets, actual_fail_percents, total_time

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
        return EnvironmentConf.buggy_reward * (1 if len(fail_features) != 0 else 0)

    def get_boundary_punish(self, vec):
        punish = 0
        boundary_option_num = sum([1 for _ in vec if
                                   _ > 100 - EnvironmentConf.boundary_threshold or _ < EnvironmentConf.boundary_threshold])
        if boundary_option_num > EnvironmentConf.boundary_num_threshold:
            punish = EnvironmentConf.boundary_punish
        return punish

    def get_delta_configuration_quality(self, avg_features, fail_features_list, resets):
        time_spt = []
        rewards = []
        qualities = []
        for _ in range(main_configure_hicond_online.particle_number):
            reset = resets[_]
            avg_feature = avg_features[_]
            fail_features = fail_features_list[_]
            if reset:
                # if len(avg_feature) != 0:    history modification
                #     self.fv_history.append(avg_feature)
                rewards.append(EnvironmentConf.reset_reward)
                time_spt.append(0)
                continue    # return EnvironmentConf.reset_reward, 0
            if self.iteration_cnt == 1:
                quality = self.get_configuration_quality(avg_feature)
                # self.fv_history.append(avg_feature)    # history modification
                # self.configuration_quality_history.append(quality)
                qualities.append(quality)
                rewards.append(0)
                time_spt.append(0)
                continue    # return 0, 0
            # get quality
            t = time.time()
            quality = self.get_configuration_quality(avg_feature)
            delta_quality = quality - self.get_k_pre_mean_quality(EnvironmentConf.dv_avg_n)
            # self.fv_history.append(avg_feature)    # history modification
            # self.configuration_quality_history.append(quality)
            qualities.append(quality)
            buggy_reward = self.get_buggy_reward(fail_features)
            boundary_punish = self.get_boundary_punish(CsmithConfiguration.get_vec_from_prob(self.particles[_]))
            delta_quality += buggy_reward
            delta_quality += boundary_punish
            t = time.time() - t
            # log
            reward_component = [quality, delta_quality, buggy_reward, boundary_punish]

            content = str(self.iteration_cnt) + '-' + str(_) + '-reward_component:'+str(reward_component)
            common_base_funs.log(MainProcessConf.log_reward, content+'\n')

            # reward rate
            rewards.append(delta_quality * EnvironmentConf.delta_rate)
            time_spt.append(t)
            continue
            # return delta_quality * EnvironmentConf.delta_rate, t

        # modify history after all the particles evaluated.
        for _ in avg_features:
            if len(_) != 0:
                self.fv_history.append(_)
        for _ in qualities:
            self.configuration_quality_history.append(_)

        return rewards, sum(time_spt)

    def evaluate_step(self, avg_features, resets, fail_features_list):
        rewards, time_cost = self.get_delta_configuration_quality(avg_features, fail_features_list, resets)
        # rewards = [rated_delta_quality] * self.n_agent
        return rewards, time_cost

    def evaluate(self):
        """
        Do test and evaluate these configuration particles.
        """
        # do test
        rewards, if_reset, actual_fail_percent, total_time = self.score()
        return rewards, if_reset, actual_fail_percent, total_time

    def update_best(self, scores):
        """
        Update the partial best position and group best position.
        """
        old_particle_best_positions = self.particle_best_position[:]
        old_particle_best_scores = self.particle_best_scores[:]
        old_global_best_positions = self.global_best_position[:]
        old_global_best_score = self.global_best_score

        update_start_time = time.time()
        for _ in range(main_configure_hicond_online.particle_number):    # operation code
            if scores[_] > self.particle_best_scores[_]:
                self.particle_best_scores[_] = scores[_]
                self.particle_best_position[_] = self.particles[_][:]
            max_idx = np.argmax(scores)
            if scores[max_idx] > self.global_best_score:
                self.global_best_score = scores[max_idx]
                self.global_best_position = self.particles[max_idx][:]
        update_end_time = time.time()

        for _ in range(main_configure_hicond_online.particle_number):
            log_env_content = str(self.iteration_cnt) + '-' + str(_) + '-best_self_particle-before_update: ' + str(old_particle_best_scores[_]) + '-' + str(old_particle_best_positions[_]) + '\n'
            log_env_content += str(self.iteration_cnt) + '-' + str(_) + '-best_self_particle-after_update: ' + str(self.particle_best_scores[_]) + '-' + str(self.particle_best_position[_]) + '\n'

        log_env_content = str(self.iteration_cnt) + '-best_global_particle-before_update: ' + str(old_global_best_score) + '-' + str(old_global_best_positions) + '\n'
        log_env_content += str(self.iteration_cnt) + '-best_global_particle-after_update: ' + str(self.global_best_position) + '-' + str(self.global_best_position) + '\n'
        common_base_funs.log(MainProcessConf.log_env, log_env_content)

        return update_end_time - update_start_time

    def update_particles(self):
        time_spt = []
        for particle_num in range(main_configure_hicond_online.particle_number):
            start_time = time.time()
            vshape = np.shape(self.velocities[particle_num])
            rand_v1 = np.random.random_sample(vshape)
            rand_v2 = np.random.random_sample(vshape)
            # convert vec into probabilities...
            self.velocities[particle_num] = self.velocities[particle_num] + \
                                 main_configure_hicond_online.c * rand_v1 * (self.global_best_position - self.particles[particle_num]) + \
                                 main_configure_hicond_online.c * rand_v2 * (self.particle_best_position[particle_num] - self.particles[particle_num])

            # limit the velocity
            if main_configure_hicond_online.limit_velocity:
                for __ in range(len(self.velocities[particle_num])):
                    if self.velocities[particle_num][__] > main_configure_hicond_online.velocity_limitations[particle_num][1]:
                        self.velocities[particle_num][__] = main_configure_hicond_online.velocity_limitations[particle_num][1]
                    if self.velocities[particle_num][__] < main_configure_hicond_online.velocity_limitations[particle_num][0]:
                        self.velocities[particle_num][__] = main_configure_hicond_online.velocity_limitations[particle_num][0]

            for __ in range(len(self.velocities[particle_num])):
                if __ in EnvironmentConf.un_tune_indexes:
                    self.velocities[particle_num][__] = 0

            old_particle = self.particles[particle_num][:]
            self.particles[particle_num] = self.particles[particle_num] + self.velocities[particle_num]

            # limit range
            for __ in range(len(self.particles[particle_num])):
                if self.particles[particle_num][__] > 100:
                    self.particles[particle_num][__] = 100
                if self.particles[particle_num][__] < 0:
                    self.particles[particle_num][__] = 0
            # min max
            start_idx = len(CsmithConfigurationConf.single_keys)
            for group_name in CsmithConfigurationConf.group_keys:
                group_itm_num = len(CsmithConfigurationConf.group_items_keys[group_name])
                end_idx = start_idx + group_itm_num

                sum_prob = sum(self.particles[particle_num][start_idx:end_idx])
                sum_prob = 1 if sum_prob == 0 else sum_prob
                for __ in range(start_idx, end_idx):
                    self.particles[particle_num][__] = self.particles[particle_num][__] / sum_prob * 100

                start_idx = end_idx

            # random init particles
            if main_configure_hicond_online.if_reset:
                boundary_option_num = sum([1 for val in self.particles[particle_num] if
                                           val > 100 - EnvironmentConf.boundary_threshold or val < EnvironmentConf.boundary_threshold])
                if boundary_option_num > EnvironmentConf.boundary_num_threshold or self.if_reset[particle_num]:
                    self.initialize_particle(particle_num)
                    common_base_funs.log(MainProcessConf.log_env,
                                         str(self.iteration_cnt) + '-' + str(particle_num) + '-reset\n')

            end_time = time.time()
            time_spt.append(end_time - start_time)

            for __ in EnvironmentConf.un_tune_indexes:
                self.particles[particle_num][__] = 0

            assert start_idx == 71

            log_env_content = str(self.iteration_cnt) + '-' + str(particle_num) + '-particles_before_update: ' + str(old_particle.tolist()) + '\n'
            log_env_content += str(self.iteration_cnt) + '-' + str(particle_num) + '-velocity: ' + str(self.velocities[particle_num].tolist()) + '\n'
            log_env_content += str(self.iteration_cnt) + '-' + str(particle_num) + '-particles_after_update: ' + str(self.particles[particle_num].tolist()) + '\n'
            common_base_funs.log(MainProcessConf.log_env, log_env_content)

        return sum(time_spt)


def run_hicond_online():
    # initialize
    time_spt = 0
    init_start_time = time.time()
    pso_sear = PSOSearcher()    # operation
    init_end_time = time.time()
    init_time = init_end_time - init_start_time
    time_spt += init_time
    log_time_content = 'initialize time: ' + str(init_time)
    common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')

    # start searching
    while time_spt <= MainProcessConf.run_time_limit:
        # convert the particles into Csmith configurations
        convert_time = pso_sear.convert_configurations()    # operation
        time_spt += convert_time
        log_time_content = str(pso_sear.iteration_cnt) + '-construct_time: ' + str(convert_time)
        common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')

        # evaluation (testing and calculating score)
        """
        We use a mapping method (self.iteration-1)*10+self.conf_cnt to do handle the test process.
        and in reward calculation, we move history update related code to the end of the calculation process.
        """
        rewards, if_reset, actual_fail_percent, total_time = pso_sear.evaluate()    # operation
        time_spt += total_time
        log_time_content = str(pso_sear.iteration_cnt) + '-evaluate_time: ' + str(total_time)
        common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')

        pso_sear.if_reset = if_reset[:]

        update_best_time = pso_sear.update_best(rewards)    # operation
        time_spt += update_best_time
        log_time_content = str(pso_sear.iteration_cnt) + '-update_best_time: ' + str(update_best_time)
        common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')

        udpate_particles_time = pso_sear.update_particles()    # operation
        time_spt += udpate_particles_time
        log_time_content = str(pso_sear.iteration_cnt) + '-udpate_particles_time: ' + str(udpate_particles_time)
        common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')

        log_time_content = str(pso_sear.iteration_cnt) + '-total_time_spent: ' + str(time_spt)
        common_base_funs.log(MainProcessConf.log_time, log_time_content + '\n')


if __name__ == '__main__':
    # Some other experiments also need same initialization process as MCS. (e.g., ensure their run under same setting)
    # initialize for main approach experiment
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

    # initialize for hicond online with MCS's reward experiment
    common_base_funs.rm_file(main_configure_hicond_online.tmp_test_program_dir)
    common_base_funs.mkdir_if_not_exists(main_configure_hicond_online.tmp_test_program_dir)

    run_hicond_online()
    # CsmithConfVecGenerator.test_()

