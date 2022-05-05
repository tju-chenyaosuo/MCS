"""
This file is to study the relationship between the hicond_crash, hicond_timeout, compile_timeout, execute_timeout and
  the number of boundary values.
"""
from multiprocessing import Queue, Process

import common_base_funs
from test_base_funs import HicondSrcGenerator, HicondSrcGenerateStatus, GCCTestProcess, GCCTestStatus
from main_configure_approach import GCCTestConf

from config_boundary_limitation import BoundaryConfigGenerator
import boundary_config_generator_conf

core_cnt = 10

def test_process(todo_queue, core_num):
    # {type}-{boundary_values}-{config}-{program}:[gen_res, com_res, exe_res]:[gen_time, com_time, exe_time]
    log_file = 'log/log-'+str(core_num)+'.txt'
    while not todo_queue.empty():
        # get info from id
        itm_id = todo_queue.get()
        itm_id_info = itm_id.split('-')
        mode = itm_id_info[0]
        boundary_value_num = itm_id_info[1]
        config_num = itm_id_info[2]

        # gen configure
        bcg_mix = BoundaryConfigGenerator(boundary_value_num=int(boundary_value_num), mode=mode)
        config_name = 'config/' + mode + '-' + boundary_value_num + '-' + config_num + 'mix.conf'
        bcg_mix.random_gen_config(config_name)

        for _ in range(100):
            # default result values
            results = ['None', 'None', 'None']
            times = [-1, -1, -1]

            # create environment
            work_dir = 'test_dir/'+itm_id+'-'+str(_)+'/'
            common_base_funs.rm_file(work_dir)
            common_base_funs.mkdir_if_not_exists(work_dir)
            src = work_dir+'a.c'
            feature_file = work_dir+'a.csv'
            err_file = work_dir+'a.err'

            # gen program
            hsg = HicondSrcGenerator(src=src, feature_file=feature_file, conf=config_name, err_file=err_file)
            res, t = hsg.gen_src()
            # record result
            results[0] = res
            times[0] = t
            if res != HicondSrcGenerateStatus.success:
                log_content = itm_id+'-'+str(_)+':'+str(results)+':'+str(times)
                common_base_funs.log(log_file, log_content+'\n')
                continue

            # compile program
            tp = GCCTestProcess(libs=[GCCTestConf.csmith_lib], src=src, work_dir=work_dir)
            res, t = tp.test_compile('-O0')
            # record result
            results[1] = res
            times[1] = t
            if res != GCCTestStatus.success:
                log_content = itm_id + '-' + str(_) + ':' + str(results) + ':' + str(times)
                common_base_funs.log(log_file, log_content + '\n')
                continue

            # execute program
            res, t = tp.test_execute('-O0', work_dir+'res')
            # record result
            results[2] = res
            times[2] = t

            log_content = itm_id + '-' + str(_) + ':' + str(results) + ':' + str(times)
            common_base_funs.log(log_file, log_content + '\n')


def main():
    # prepare for all environment
    common_base_funs.rm_file('log')
    common_base_funs.mkdir_if_not_exists('log')
    common_base_funs.rm_file('test_dir')
    common_base_funs.mkdir_if_not_exists('test_dir')
    common_base_funs.rm_file('config')
    common_base_funs.mkdir_if_not_exists('config')

    # add the task into todo_queue
    todo_queue = Queue()
    # add mix
    step = 71 / 10
    boundary_value_nums = [int(_ * step) for _ in range(0, 11)]
    boundary_value_nums = [_ for _ in boundary_value_nums if boundary_config_generator_conf.mix_limit[0] <= _ <= boundary_config_generator_conf.mix_limit[1]]
    if boundary_config_generator_conf.mix_limit[0] not in boundary_value_nums:
        boundary_value_nums = [boundary_config_generator_conf.mix_limit[0]] + boundary_value_nums
    if boundary_config_generator_conf.mix_limit[1] not in boundary_value_nums:
        boundary_value_nums = boundary_value_nums + [boundary_config_generator_conf.mix_limit[1]]
    for boundary_value_num in boundary_value_nums:
        for config_num in range(100):
            mode = boundary_config_generator_conf.modes[2]
            itm_id = mode+'-'+str(boundary_value_num)+'-'+str(config_num)
            todo_queue.put(itm_id)
    # start the process
    ps = []
    for _ in range(core_cnt):
        p = Process(target=test_process, args=(todo_queue, _, ))
        p.daemon = True
        ps.append(p)
        p.start()
    for _ in ps:
        _.join()


if __name__ == '__main__':
    main()
