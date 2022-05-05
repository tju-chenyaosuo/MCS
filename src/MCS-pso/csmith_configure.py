import numpy as np

from main_configure_approach import CsmithConfigurationConf
from main_configure_approach import MainProcessConf
import common_base_funs
import random


class CsmithConfiguration:
    '''
    Csmith Configuration:
    base_itm_single={single-opt1: val1, single-opt2: val2, ..., }
    base_itm_group={group-key: {group1: val1, group2: val2, ..., }, ..., }
    '''

    @staticmethod
    def gen_default_conf(conf_file):
        """
        from csmith --dump-default-probabilities
        :return: base_itm_single, base_itm_group
        """
        common_base_funs.rm_file(conf_file)
        cmd = ' '.join([CsmithConfigurationConf.csmith, '--dump-default-probabilities', conf_file])
        common_base_funs.execmd(cmd)
        base_itm_single, base_itm_group = CsmithConfiguration.load_conf(conf_file)
        return base_itm_single, base_itm_group

    @staticmethod
    def gen_random_conf(conf_file, seed):
        """
        from csmith --dump-default-probabilities
        :return: base_itm_single, base_itm_group
        """
        common_base_funs.rm_file(conf_file)
        cmd = ' '.join([CsmithConfigurationConf.csmith, '--seed', seed, '--dump-random-probabilities', conf_file])
        common_base_funs.execmd(cmd)
        base_itm_single, base_itm_group = CsmithConfiguration.load_conf(conf_file)
        return base_itm_single, base_itm_group

    @staticmethod
    def load_conf(conf_file):
        base_itm_single = {}
        base_itm_group = {}
        conf_lines = common_base_funs.get_file_lines(conf_file)
        for c in conf_lines:
            if len(c) == 0 or '#' in c:
                continue
            # single
            if not c.startswith('['):
                c_info = c.split('=')
                c_key = c_info[0]
                c_val = c_info[1]
                assert c_key not in base_itm_single, 'There should not be two same single key in a config file!'
                base_itm_single[c_key] = float(c_val)
                continue
            # group
            if c.startswith('['):
                c = c.replace('[', '').replace(']', '').strip()
                c_info = c.split(',')
                group_key = c_info[0]
                del c_info[0]
                group_dir = {}
                for c_k_v in c_info:
                    c_k_v = c_k_v.split('=')
                    assert c_k_v[0] not in group_dir, 'There should not be two same key in same group!'
                    group_dir[c_k_v[0]] = float(c_k_v[1])
                base_itm_group[group_key] = group_dir
                continue
        return base_itm_single, base_itm_group

    @staticmethod
    def write_conf(conf_file, base_itm_single, base_itm_group):
        common_base_funs.rm_file(conf_file)
        single_key_values = '\n\n'.join([key + '=' + str(base_itm_single[key]) for key in CsmithConfigurationConf.single_keys])
        group_key_values = {key: ','.join([item_key + '=' + str(base_itm_group[key][item_key])
                                           for item_key in CsmithConfigurationConf.group_items_keys[key]])
                            for key in CsmithConfigurationConf.group_keys}
        group_key_values = '\n\n'.join(['[' + ','.join([key, group_key_values[key]]) + ']'
                                      for key in CsmithConfigurationConf.group_keys])
        common_base_funs.put_file_content(conf_file, single_key_values + '\n\n' + group_key_values)

    @staticmethod
    def default_conf_f():
        conf_file = 'default.config'
        CsmithConfiguration.gen_default_conf(conf_file)
        return CsmithConfiguration.load_conf(conf_file)

    @staticmethod
    def get_vector(base_itm_single, base_itm_group):
        state = []
        for key in CsmithConfigurationConf.single_keys:
            state.append(base_itm_single[key])
        for key in CsmithConfigurationConf.group_keys:
            for item_key in CsmithConfigurationConf.group_items_keys[key]:
                state.append(base_itm_group[key][item_key])
        return state

    @staticmethod
    def get_items(vec):
        # TODO: to be tested
        base_itm_single = {}
        base_itm_group = {}
        index = 0
        for key in CsmithConfigurationConf.single_keys:
            base_itm_single[key] = vec[index]
            index += 1
        for group in CsmithConfigurationConf.group_keys:
            base_itm_group[group] = {}
            for key in CsmithConfigurationConf.group_items_keys[group]:
                base_itm_group[group][key] = vec[index]
                index += 1
        return base_itm_single, base_itm_group

    @staticmethod
    def get_prob_vec_from_vec(vec):    # tested
        """
        Return probability vector from csmith configuration vector.
        """
        # extract single probabilities
        prob_vec = [vec[_] for _ in range(len(CsmithConfigurationConf.single_keys))]
        # extract group probabilities
        start_idx = len(CsmithConfigurationConf.single_keys)
        for group_name in CsmithConfigurationConf.group_keys:
            group_itm_num = len(CsmithConfigurationConf.group_items_keys[group_name])
            end_idx = start_idx + group_itm_num

            group_vec = vec[start_idx:end_idx]
            group_vec = [[group_vec[_], _] for _ in range(len(group_vec))]
            sorted_group_vec = sorted(group_vec)
            sorted_group_vec.append([0, -1])
            group_prob_s = sorted([[sorted_group_vec[_][1], sorted_group_vec[_][0] - sorted_group_vec[_-1][0]]
                                   for _ in range(len(sorted_group_vec))][:-1])
            group_prob_s = [_[1] for _ in group_prob_s]

            assert sum(group_prob_s) == 100
            prob_vec.extend(group_prob_s)

            start_idx = end_idx
        assert len(prob_vec) == 71
        return prob_vec

    @staticmethod
    def get_vec_from_prob(prob_vec):
        vec = []
        # single
        for _ in range(len(CsmithConfigurationConf.single_keys)):
            vec.append(prob_vec[_])
        # group
        start_idx = len(CsmithConfigurationConf.single_keys)
        for group_name in CsmithConfigurationConf.group_keys:
            group_num = len(CsmithConfigurationConf.group_items_keys[group_name])
            end_idx = start_idx + group_num

            asce_group_data = prob_vec[start_idx:end_idx]
            asce_group_data_idx = sorted([[asce_group_data[_], _] for _ in range(len(asce_group_data))])
            asce_group_data = [_[0] for _ in asce_group_data_idx]
            asce_group_idx = [_[1] for _ in asce_group_data_idx]

            asce_conf_data = [asce_group_data[0] if _ == 0 else int(sum(asce_group_data[:_+1])) for _ in range(len(asce_group_data_idx))]
            asce_conf_data[-1] = 100
            conf_data = [-1 for _ in range(group_num)]
            for _ in range(group_num):
                conf_data[asce_group_idx[_]] = asce_conf_data[_]
            vec.extend(conf_data)

            start_idx = end_idx
        assert len(vec) == 71
        assert start_idx == 71
        return vec




if __name__ == '__main__':
    # test code
    # test_conf = 'test.conf'
    # CsmithConfiguration.gen_default_conf(test_conf)
    # single_conf, group_conf = CsmithConfiguration.load_conf(test_conf)
    # print(len(single_conf))
    # assert len(single_conf) == 24, 'Shape error!'
    # for i in range(10):
    #     CsmithConfiguration.gen_random_conf(test_conf, 0)
    #     single_conf, group_conf = CsmithConfiguration.load_conf(test_conf)
    #     print(single_conf)
    #     print(group_conf)
    #     assert len(single_conf) == 24, 'Shape error!'
    #     print(''.ljust(20, '*'))
    pass

    tar_file = 'tmp/file'
    seed = str(random.randint(0, 8000000000))
    base_itm_single, base_itm_group = CsmithConfiguration.gen_random_conf(tar_file, seed)
    feature_vec = CsmithConfiguration.get_vector(base_itm_single, base_itm_group)
    prob_vec = CsmithConfiguration.get_prob_vec_from_vec(feature_vec)

    for _ in range(len(feature_vec)):
        print([feature_vec[_], prob_vec[_]])
