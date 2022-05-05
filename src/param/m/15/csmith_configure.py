from main_configure_approach import CsmithConfigurationConf
import common_base_funs


class CsmithConfiguration:
    @staticmethod
    def gen_default_conf(conf_file):
        common_base_funs.rm_file(conf_file)
        cmd = ' '.join([CsmithConfigurationConf.csmith, '--dump-default-probabilities', conf_file])
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
