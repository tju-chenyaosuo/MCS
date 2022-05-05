from main_configure_approach import CsmithConfigurationConf

# construct mode
modes = ["upper_bound", "lower_bound", "mix"]

# number of options
# single
single_num = len(CsmithConfigurationConf.single_keys)
untune_single_num = 1
tune_single_num = single_num - untune_single_num
# group
group_nums = [len(CsmithConfigurationConf.group_items_keys[_]) for _ in CsmithConfigurationConf.group_keys]
untune_group_nums = [1, 0, 0, 2, 0]

tune_group_nums = [group_nums[_] - untune_group_nums[_] for _ in range(len(group_nums))]
all_tune_group_num = sum(tune_group_nums)

# limitation of boundary values
# lower boundary values limitation
lower_single_limit = [untune_single_num, single_num]
lower_group_limit = [[untune_group_nums[_], group_nums[_] - 1] for _ in range(len(group_nums))]
lower_limit = [lower_single_limit[0] + sum([_[0] for _ in lower_group_limit]),
               lower_single_limit[1] + sum([_[1] for _ in lower_group_limit])]
# upper boundary values limitation
upper_single_limit = [0, single_num - untune_single_num]
upper_group_limit = [[1, min(group_nums[_] - untune_group_nums[_], 5)] for _ in range(len(group_nums))]
upper_limit = [upper_single_limit[0] + sum([_[0] for _ in upper_group_limit]),
               upper_single_limit[1] + sum([_[1] for _ in upper_group_limit])]
# mix boundary values limitation
mix_single_limit = [untune_single_num, single_num]
mix_group_limit = [[untune_group_nums[_] + 1, group_nums[_]] for _ in range(len(group_nums))]
mix_limit = [mix_single_limit[0] + sum([_[0] for _ in mix_group_limit]),
             mix_single_limit[1] + sum([_[1] for _ in mix_group_limit])]

# index information
tune_idx = [_ for _ in range(tune_single_num + all_tune_group_num)]
tune_single_idx = tune_idx[:tune_single_num]
tune_group_idx = tune_idx[tune_single_num:]
assert len(tune_group_idx) == all_tune_group_num
assert tune_single_num + all_tune_group_num == 71 - len(CsmithConfigurationConf.untune_options)