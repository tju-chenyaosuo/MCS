"""
This file is to research the relationship between the number of boundary values and time spent on running each test
  program.
The experiment will split into several steps:
    1. Determine the number of boundary values.
    2. Random generate 100 configuration,
      for each configuration, run 100 test programs and record the time spent on running each test program.
"""

import random

import common_base_funs
from csmith_configure import CsmithConfiguration
from main_configure_approach import CsmithConfigurationConf, EnvironmentConf
from test_base_funs import HicondSrcGenerator, HicondSrcGenerateStatus

import boundary_config_generator_conf


class BoundaryConfigGenerator:
    def __init__(self, boundary_value_num, mode):
        assert mode in boundary_config_generator_conf.modes
        self.boundary_value_num = boundary_value_num
        self.mode = mode

        # get limit info via mode
        self.single_limit = None
        self.group_limit = None
        self.limit = None
        if self.mode == boundary_config_generator_conf.modes[0]:  # upper_bound
            self.single_limit = boundary_config_generator_conf.upper_single_limit[:]
            self.group_limit = boundary_config_generator_conf.upper_group_limit[:]
            self.limit = boundary_config_generator_conf.upper_limit[:]
        if self.mode == boundary_config_generator_conf.modes[1]:  # lower_bound
            self.single_limit = boundary_config_generator_conf.lower_single_limit[:]
            self.group_limit = boundary_config_generator_conf.lower_group_limit[:]
            self.limit = boundary_config_generator_conf.lower_limit[:]
        if self.mode == boundary_config_generator_conf.modes[2]:  # mix
            self.single_limit = boundary_config_generator_conf.mix_single_limit[:]
            self.group_limit = boundary_config_generator_conf.mix_group_limit[:]
            self.limit = boundary_config_generator_conf.mix_limit[:]
        assert self.single_limit is not None
        assert self.group_limit is not None
        assert self.limit is not None

        self.all_group_limit = [sum([_[0] for _ in self.group_limit]), sum([_[1] for _ in self.group_limit])]

        assert self.limit[0] <= self.boundary_value_num <= self.limit[1]

    def random_gen_config(self, config):
        print('**********************************************')
        print('start construct:')
        print('mode: ' + self.mode)

        # Arrange boundary num into single and group
        min_single = max(self.boundary_value_num - self.all_group_limit[1], self.single_limit[0])
        max_single = min(self.boundary_value_num - self.all_group_limit[0], self.single_limit[1])
        assert self.single_limit[0] <= min_single and max_single <= self.single_limit[1], \
            '['+str(self.single_limit[0])+','+str(min_single)+','+str(max_single)+str(self.single_limit[1])+']'
        single_num = random.randint(min_single, max_single) if max_single > min_single else min_single
        all_group_num = self.boundary_value_num - single_num
        assert self.all_group_limit[0] <= all_group_num <= self.all_group_limit[1]
        print('end arrange')
        print('single: '+str(single_num))
        print('group: ' + str(all_group_num))

        # Calculate additional boundary value
        # limit[0] is the boundary value by default, so all we need to calculate additional boundary value is "-limit[0]".
        add_single_num = single_num-self.single_limit[0]
        add_all_group_num = all_group_num-self.all_group_limit[0]
        print('end calculate additional boundary values')
        print('add single: '+str(add_single_num))
        print('add group: '+str(add_all_group_num))

        # randomly construct boundary value vector
        single_vec = self.random_single_vec(num=add_single_num)
        group_vecs = self.all_random_group_vecs(num=add_all_group_num)
        vec = single_vec[:]
        for _ in group_vecs:
            vec.extend(_)
        assert len(vec) == 71 - len(CsmithConfigurationConf.untune_options)

        # map the tune vector to all vector
        assert len(vec) == len(EnvironmentConf.action_mapping)
        vec_71 = [0 for _ in range(71)]
        for _ in range(len(EnvironmentConf.action_mapping)):
            vec_71[EnvironmentConf.action_mapping[_]] = vec[_]

        # write the conf
        single_itm, group_itm = CsmithConfiguration.get_items(vec_71)
        CsmithConfiguration.write_conf(config, single_itm, group_itm)

    def random_single_vec(self, num):
        single_idx = random.sample(boundary_config_generator_conf.tune_single_idx, num)
        vec = [self.boundary_value() if _ in single_idx else random.randint(5, 95)
               for _ in range(boundary_config_generator_conf.tune_single_num)]
        return vec

    def all_random_group_vecs(self, num):
        # To understand the following code:
        # One should know that, limit means the num of boundary values can be specified, including constrains and untune
        # When mode is lower bound, the number of boundary values would -untune_num, because untune_num is always 0
        # When mode is upper bound, the number of boundary values would -1, because constrains we do not count.
        # When mode is mix bound, both of above meets.
        group_nums = None
        if self.mode == boundary_config_generator_conf.modes[1]:    # lower bound
            group_nums = self.simulate_dice(box_cnt=len(CsmithConfigurationConf.group_keys),
                                            box_sizes=[self.group_limit[_][1] - boundary_config_generator_conf.untune_group_nums[_]
                                                       for _ in range(len(self.group_limit))], game_time=num)
        if self.mode == boundary_config_generator_conf.modes[0]:    # upper bound
            group_nums = self.simulate_dice(box_cnt=len(CsmithConfigurationConf.group_keys),
                                            box_sizes=[self.group_limit[_][1] - 1
                                                       for _ in range(len(self.group_limit))], game_time=num)
        if self.mode == boundary_config_generator_conf.modes[2]:    # mix bound
            group_nums = self.simulate_dice(box_cnt=len(CsmithConfigurationConf.group_keys),
                                            box_sizes=[self.group_limit[_][1] - boundary_config_generator_conf.untune_group_nums[_] - 1
                                                       for _ in range(len(self.group_limit))], game_time=num)
        assert group_nums is not None
        return [self.random_group_vec(num_of_option=boundary_config_generator_conf.tune_group_nums[_],
                                      num_of_bound=group_nums[_])
                for _ in range(len(CsmithConfigurationConf.group_keys))]

    def random_group_vec(self, num_of_option, num_of_bound):
        vec = None
        if self.mode == boundary_config_generator_conf.modes[0]:    # upper_bound
            vec = BoundaryConfigGenerator.simulate_dice(box_cnt=num_of_option - num_of_bound - 1,
                                                        box_sizes=[95 for _ in range(num_of_option)],
                                                        game_time=95)
            vec = BoundaryConfigGenerator.sum_up_vec(vec)
            vec.extend(BoundaryConfigGenerator.gen_upper_vec(num_of_bound))
            random.shuffle(vec)
        if self.mode == boundary_config_generator_conf.modes[1]:    # lower_bound
            vec = BoundaryConfigGenerator.simulate_dice(box_cnt=num_of_option - num_of_bound,
                                                        box_sizes=[100 for _ in range(num_of_option)],
                                                        game_time=100, default_cnt=5)
            vec = BoundaryConfigGenerator.sum_up_vec(vec)
            vec = BoundaryConfigGenerator.gen_lower_vec(num_of_bound) + vec
            random.shuffle(vec)
            pass
        if self.mode == boundary_config_generator_conf.modes[2]:    # mix
            vec = BoundaryConfigGenerator.simulate_dice(box_cnt=num_of_option - num_of_bound - 1,
                                                        box_sizes=[95 for _ in range(num_of_option)],
                                                        game_time=95, default_cnt=5)
            vec = BoundaryConfigGenerator.sum_up_vec(vec)
            max_upper_num = min(num_of_bound - 1, 4)
            min_upper_num = 0
            upper_num = random.randint(min_upper_num, max_upper_num) if max_upper_num > min_upper_num else min_upper_num
            lower_num = num_of_bound - upper_num
            upper_vec = BoundaryConfigGenerator.gen_upper_vec(upper_num)
            lower_vec = BoundaryConfigGenerator.gen_lower_vec(lower_num)
            vec = lower_vec + vec + upper_vec
            random.shuffle(vec)
        assert vec is not None
        return vec

    @staticmethod
    def gen_upper_vec(num):
        assert num <= 4
        return random.sample([96, 97, 98, 99], num)+[100]

    @staticmethod
    def gen_lower_vec(num):
        zero_min = max(0, num - len([1, 2, 3, 4]))
        zero_num = random.randint(zero_min, num) if num > zero_min else zero_min
        non_zero_num = num - zero_num
        zeros = [0 for _ in range(zero_num)]
        non_zeros = random.sample([1, 2, 3, 4], non_zero_num)
        return zeros + non_zeros

    def boundary_value(self):
        """
        Get a random value according to self.mode
        """
        random_num = None
        if self.mode == boundary_config_generator_conf.modes[0]:    # upper_bound
            random_num = random.randint(96, 100)
        if self.mode == boundary_config_generator_conf.modes[1]:    # lower_bound
            random_num = random.randint(0, 4)
        if self.mode == boundary_config_generator_conf.modes[2]:    # mix
            random_num = random.randint(0, 4) if random.randint(0, 1) == 0 else random.randint(96, 100)
        assert random_num is not None
        return random_num

    @staticmethod
    def simulate_dice(box_cnt, box_sizes, game_time, default_cnt=0):
        # Assume that there are several boxes, each box has different size, and our task is to random put a peach into
        #   one of them, repeat this action game_time times.
        if box_cnt == 0:
            return []
        boxes = [default_cnt for _ in range(box_cnt)]
        game_time -= sum(boxes)
        while game_time > 0:
            random_idx = random.randint(0, box_cnt - 1) if box_cnt != 1 else 0
            if boxes[random_idx] < box_sizes[random_idx]:
                boxes[random_idx] += 1
                game_time -= 1
        return boxes

    @staticmethod
    def sum_up_vec(vec):
        vec = sorted(vec)
        vec.append(0)
        for _ in range(len(vec)):
            vec[_] += vec[_-1]
        return vec[:-1]


if __name__ == '__main__':
    for _ in range(boundary_config_generator_conf.upper_limit[0], boundary_config_generator_conf.upper_limit[1] + 1):
        for i in range(1):
            print(str(_)+'-'+str(i))
            bcg_upper = BoundaryConfigGenerator(boundary_value_num=_, mode=boundary_config_generator_conf.modes[0])
            upper_name = 'config/'+str(_) + 'upper.conf'
            bcg_upper.random_gen_config(upper_name)
            single, group = CsmithConfiguration.load_conf(upper_name)
            vec = CsmithConfiguration.get_vector(single, group)
            assert len([1 for __ in vec if 96 <= __ <= 100]) == _
            hsg = HicondSrcGenerator(src='test_dir/a.c', feature_file='test_dir/a.csv', conf=upper_name,
                                     err_file='test_dir/a.err')
            res, t = hsg.gen_src()
            if res == HicondSrcGenerateStatus.crash:
                msg = common_base_funs.get_file_content('test_dir/a.err')
                if 'assert' in msg:
                    exit(1)

    for _ in range(boundary_config_generator_conf.lower_limit[0], boundary_config_generator_conf.lower_limit[1] + 1):
        for i in range(1):
            print(str(_)+'-'+str(i))
            bcg_lower = BoundaryConfigGenerator(boundary_value_num=_, mode=boundary_config_generator_conf.modes[1])
            lower_name = 'config/'+str(_) + 'lower.conf'
            bcg_lower.random_gen_config(lower_name)
            single, group = CsmithConfiguration.load_conf(lower_name)
            vec = CsmithConfiguration.get_vector(single, group)
            assert len([1 for __ in vec if 0 <= __ <= 4]) == _
            hsg = HicondSrcGenerator(src='test_dir/a.c', feature_file='test_dir/a.csv', conf=lower_name,
                                     err_file='test_dir/a.err')
            res, t = hsg.gen_src()
            if res == HicondSrcGenerateStatus.crash:
                msg = common_base_funs.get_file_content('test_dir/a.err')
                if 'assert' in msg:
                    exit(1)

    for _ in range(boundary_config_generator_conf.mix_limit[0], boundary_config_generator_conf.mix_limit[1] + 1):
        for i in range(1):
            print(str(_)+'-'+str(i))
            bcg_mix = BoundaryConfigGenerator(boundary_value_num=_, mode=boundary_config_generator_conf.modes[2])
            mix_name = 'config/' + str(_) + 'mix.conf'
            bcg_mix.random_gen_config(mix_name)
            single, group = CsmithConfiguration.load_conf(mix_name)
            vec = CsmithConfiguration.get_vector(single, group)
            assert len([1 for __ in vec if 0 <= __ <= 4 or 96 <= __ <= 100]) == _
            hsg = HicondSrcGenerator(src='test_dir/a.c', feature_file='test_dir/a.csv', conf=mix_name,
                                     err_file='test_dir/a.err')
            res, t = hsg.gen_src()
            if res == HicondSrcGenerateStatus.crash:
                msg = common_base_funs.get_file_content('test_dir/a.err')
                if 'assert' in msg:
                    exit(1)
