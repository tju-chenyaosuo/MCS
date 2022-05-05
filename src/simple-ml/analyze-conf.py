import os
from csmith_configure import CsmithConfiguration
import common_base_funs
import numpy as np
from scipy.spatial.distance import pdist
import random
from main_configure_approach import CsmithConfigurationConf


def get_conf_vecs(dir_name):
    vecs = []
    cocnf_num = len(os.listdir(dir_name))
    conf_fs = [dir_name+os.sep+'config'+str(_) for _ in range(1, cocnf_num+1)]
    for _ in conf_fs:
        if not os.path.exists(_):
            break
        single_items, group_items = CsmithConfiguration.load_conf(_)
        vec = CsmithConfiguration.get_vector(single_items, group_items)
        vec = [int(_) for _ in vec]
        vecs.append(vec)
    return vecs


def get_manhattan(vec1, vec2):
    return sum([abs(vec1[_]-vec2[_]) for _ in range(len(vec1))])


def get_diff(vec1, vec2):
    return sum([1 for _ in range(len(vec1)) if vec1[_] != vec2[_]])


def get_cosine(vec1, vec2):
    vec2 = vec2
    m = np.vstack([vec1, vec2])
    dist2 = pdist(m, 'cosine')
    # rely on the order of pdist' return value.
    cosine_ds = dist2[:len(vec2)]
    rt = np.mean(sorted(cosine_ds)[:len(vec2)])
    return rt


def get_each_diff(vec1, vec2):
    return [abs(vec1[_] - vec2[_]) for _ in range(len(vec1))]


def get_diff_info(vecs, step):
    inx = 0
    print(['iter1', 'iter2', 'diff', 'manhattan', 'cosine', 'max', 'min', 'avg'])
    while inx+step < len(vecs):
        each_diff = get_each_diff(vecs[inx], vecs[inx+step])
        print(str([inx, inx+step]) + '-summary:' + str([get_diff(vecs[inx], vecs[inx+step]), sum(each_diff),
                                                max(each_diff), min(each_diff), sum(each_diff) / len(each_diff)]))
        print(str([inx, inx+step]) + '-vecs-' + str(inx) + '-:' + str(vecs[inx]))
        print(str([inx, inx+step]) + '-vecs-' + str(inx+step) + '-:' + str(vecs[inx+step]))
        print(str([inx, inx+step]) + '-each_diff:' + str(each_diff))
        inx += 1


def get_single_diff(vecs1, vecs2, compare_names):
    each_diff = get_each_diff(vecs1, vecs2)
    prefix = str([compare_names[0], compare_names[1]])
    print(prefix + '-summary:' + str([get_diff(vecs1, vecs2), sum(each_diff), max(each_diff), min(each_diff),
                                      sum(each_diff) / len(each_diff)]))
    print(str(compare_names[0]) + '-vecs:' + str(vecs1))
    print(str(compare_names[1]) + '-vecs:' + str(vecs2))
    print(prefix + '-each_diff:' + str(each_diff))


def get_score_times(log_file):
    log_content = common_base_funs.get_file_lines(log_file)
    score_times = []
    for _ in log_content:
        if '-score_time:' in _:
            score_times.append(_)
    return score_times


def get_reset_iters(log_file):
    log_content = common_base_funs.get_file_lines(log_file)
    vec = []
    for _ in log_content:
        if 'reset' in _ and 'True' in _:
            vec.append(int(_.split('-')[0]))
    return vec


def get_buggy_iters(test_dir):
    com_cra_cmd = ' '.join(['find', test_dir, '-name', "'*com*cra*'"])
    exe_cra_cmd = ' '.join(['find', test_dir, '-name', "'*exe*cra*'"])
    mis_cmd = ' '.join(['find', test_dir, '-name', "'*mis*'"])
    com_cra_iters = common_base_funs.execmd(com_cra_cmd).split('\n')[:-1]
    exe_cra_iters = common_base_funs.execmd(exe_cra_cmd).split('\n')[:-1]
    mis_iters = common_base_funs.execmd(mis_cmd).split('\n')[:-1]
    com_cra_iters = [int(_.split('/')[-2].split('-')[0]) for _ in com_cra_iters]
    exe_cra_iters = [int(_.split('/')[-2].split('-')[0]) for _ in exe_cra_iters]
    mis_iters = [int(_.split('/')[-2].split('-')[0]) for _ in mis_iters]
    return com_cra_iters + exe_cra_iters + mis_iters


def analyze_boundary_options(log_file, conf_dir, test_dir, factor):
    score_times = get_score_times(log_file)
    score_times = [float(_.split(':')[1]) / factor for _ in score_times]

    vecs = get_conf_vecs(conf_dir)
    bound_idxes = []
    reset_points = get_reset_iters(log_file)
    bug_points = get_buggy_iters(test_dir)

    for v in range(len(vecs)):
        bound_idxes.append([_ for _ in range(len(vecs[v])) if vecs[v][_] > 95 or vecs[v][_] < 5])
    print('iter_num,boundary_value_num,reset,bug,time')
    for _ in range(len(bound_idxes)):
        content = [str(_), str(len(bound_idxes[_]))]
        if _ in reset_points:
            content.append('0')
        else:
            content.append('')
        if _ in bug_points:
            content.append('0')
        else:
            content.append('')
        if _ < len(score_times):
            content.append(str(score_times[_]))
        print(','.join(content))


def get_timeout_conf_vec(log_file, csmith_conf_dir):
    reset_iters = get_reset_iters(log_file)
    reset_iters = [str(_) for _ in reset_iters]
    reset_confs = [common_base_funs.add_sep(csmith_conf_dir)+'config'+_ for _ in reset_iters]
    vecs = []
    # load config
    for _ in reset_confs:
        single_item, group_item = CsmithConfiguration.load_conf(_)
        vecs.append(CsmithConfiguration.get_vector(single_item, group_item)[:len(single_item)])
    return vecs


def analyze_timeout_conf_small(csmith_conf_dir, log_file, threshold, rate, normal_threshold, normal_rate):
    reset_vecs = get_timeout_conf_vec(log_file, csmith_conf_dir)
    normal_vecs = get_conf_vecs(csmith_conf_dir)
    # reset small values
    reset_cols = [0 for _ in CsmithConfigurationConf.single_keys]
    for c in range(len(CsmithConfigurationConf.single_keys)):
        counter = 0
        for i in range(len(reset_vecs)):
            if reset_vecs[i][c] <= threshold or reset_vecs[i][c] >= 100-threshold:
                counter += 1
        if counter >= rate * len(reset_vecs):
            reset_cols[c] = 1
    print(reset_cols)
    # normal small values
    normal_cols = [0 for _ in CsmithConfigurationConf.single_keys]
    for c in range(len(CsmithConfigurationConf.single_keys)):
        counter = 0
        for i in range(len(normal_vecs)):
            if normal_vecs[i][c] <= normal_threshold or normal_vecs[i][c] >= 100-normal_threshold:
                counter += 1
        if counter >= normal_rate * len(normal_vecs):
            normal_cols[c] = 1
    print(normal_cols)
    # print result
    print([CsmithConfigurationConf.single_keys[_] for _ in range(len(normal_cols)) if normal_cols[_] == 0 and reset_cols[_]] == 1)


def analyze_timeout_conf(csmith_conf_dir, log_file):
    reset_iters = get_reset_iters(log_file)
    reset_iters = [str(_) for _ in reset_iters]
    reset_confs = [common_base_funs.add_sep(csmith_conf_dir)+'config'+_ for _ in reset_iters]
    vecs = []
    for _ in reset_confs:
        single_item, group_item = CsmithConfiguration.load_conf(_)
        vecs.append(CsmithConfiguration.get_vector(single_item, group_item))
    content = ['iter']
    for _ in CsmithConfigurationConf.single_keys:
        content.append(_)
    for _ in CsmithConfigurationConf.group_keys:
        for item in CsmithConfigurationConf.group_items_keys[_]:
            content.append(_+':'+item)
    print(','.join(content))
    for i in range(len(vecs)):
        print(reset_iters[i]+','+','.join([str(_) for _ in vecs[i]]))


def analyze_single_boundary(log_file, factor, conf_dir, bug_points):
    score_times = get_score_times(log_file)
    score_times = [float(_.split(':')[1]) / factor for _ in score_times]

    vecs = get_conf_vecs(conf_dir)
    vecs = [_[:len(CsmithConfigurationConf.single_keys)] for _ in vecs]
    bound_idxes = []
    reset_points = get_reset_iters(log_file)

    for v in range(len(vecs)):
        bound_idxes.append([_ for _ in range(len(vecs[v])) if vecs[v][_] > 95 or vecs[v][_] < 5])
    print('iter_num,boundary_value_num,reset,bug,time')
    for _ in range(len(bound_idxes)):
        content = [str(_), str(len(bound_idxes[_]))]
        if _ in reset_points:
            content.append('0')
        else:
            content.append('')
        if _ in bug_points:
            content.append('0')
        else:
            content.append('')
        if _ < len(score_times):
            content.append(str(score_times[_]))
        print(','.join(content))


def analyze_group_boundary(log_file, factor, conf_dir, bug_points):
    score_times = get_score_times(log_file)
    score_times = [float(_.split(':')[1]) / factor for _ in score_times]
    vecs = get_conf_vecs(conf_dir)
    vecs = [_[len(CsmithConfigurationConf.single_keys):] for _ in vecs]
    bound_idxes = []
    reset_points = get_reset_iters(log_file)

    for v in range(len(vecs)):
        bound_idxes.append([_ for _ in range(len(vecs[v])) if vecs[v][_] > 95 or vecs[v][_] < 5])
    print('iter_num,boundary_value_num,reset,bug,time')
    for _ in range(len(bound_idxes)):
        content = [str(_), str(len(bound_idxes[_]))]
        if _ in reset_points:
            content.append('0')
        else:
            content.append('')
        if _ in bug_points:
            content.append('0')
        else:
            content.append('')
        if _ < len(score_times):
            content.append(str(score_times[_]))
        print(','.join(content))


if __name__ == '__main__':
    analyze_boundary_options(log_file='log/total.txt', conf_dir='csmith-config', test_dir='test_dir', factor=50)

    # analyze_single_boundary(log_file='log/total.txt', factor=50, conf_dir='csmith-config',
    #                         bug_points=[206, 394, 771, 283, 104, 236, 288, 911, 58, 36, 262, 217, 566, 230, 279, 107, 32, 769, 10])
    # analyze_group_boundary(log_file='log/total.txt', factor=50, conf_dir='csmith-config',
    #                             bug_points=[206, 394, 771, 283, 104, 236, 288, 911, 58, 36, 262, 217, 566, 230, 279, 107, 32, 769, 10])

