import random
from test_base_funs import *
import main_configure_baseline
from multiprocessing import Queue
from multiprocessing import Process
from main_configure_approach import MainProcessConf


"""
Before run this file, please check:
    1. main_configure_baseline
    2. main_configure_approach.MainProcessConf.test_compiler_type
"""


def do_test(run_time_limit, program_queue, core_num):
    time_spt = 0
    log_file = main_configure_baseline.log_prefix + str(core_num)
    while time_spt < run_time_limit:
        # generate test program
        assert not program_queue.empty()
        program_num = program_queue.get()
        conf_file = None
        program_id = None
        if main_configure_baseline.baseline_type == main_configure_baseline.baseline_types[0]:    # swarm
            program_id = 'swarm-' + str(program_num)
        if main_configure_baseline.baseline_type == main_configure_baseline.baseline_types[1]:    # hicond
            conf_num = random.randint(0, 9)
            conf_file = main_configure_baseline.hicond_confs[conf_num]
            program_id = str(conf_num) + '-' + str(program_num)
        assert program_id is not None
        work_dir = main_configure_baseline.test_dir + str(program_id)
        work_dir = common_base_funs.add_sep(work_dir)
        common_base_funs.mkdir_if_not_exists(work_dir)
        seed = MainProcessConf.seeds[program_num]
        src = work_dir + 'a.c'
        feature_file = work_dir + 'a.csv'
        gen_err_file = work_dir + 'gen-err.txt'
        hsg = None
        if main_configure_baseline.baseline_type == main_configure_baseline.baseline_types[0]:  # swarm
            hsg = HicondSrcGenerator(src=src, feature_file=feature_file, err_file=gen_err_file, seed=seed, random=True)
        if main_configure_baseline.baseline_type == main_configure_baseline.baseline_types[1]:    # hicond
            assert conf_file is not None
            hsg = HicondSrcGenerator(src=src, feature_file=feature_file, err_file=gen_err_file, seed=seed,
                                     conf=conf_file)
        assert hsg is not None
        # do test
        res, gen_program_time = hsg.gen_src()
        time_spt += gen_program_time
        if res != HicondSrcGenerateStatus.success:
            common_base_funs.log(log_file,
                                 str([program_id, conf_file, [], res, gen_program_time]) + '\n')
            continue
        tp = None
        if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[0]:    # gcc
            tp = GCCTestProcess(libs=[GCCTestConf.csmith_lib], src=src, work_dir=work_dir)
        if MainProcessConf.test_compiler_type == MainProcessConf.compiler_types[1]:    # llvm
            tp = LLVMTestProcess(libs=[LLVMTestConf.csmith_lib], src=src, work_dir=work_dir)
        assert tp is not None
        res, test_program_time = tp.do_test()
        time_spt += test_program_time
        common_base_funs.log(log_file,
                             str([program_id, conf_file, ProgramFeature.load_feature(feature_file), res,
                                  gen_program_time + test_program_time]) + '\n')
        if res in TestResult.normal_results:
            common_base_funs.rm_file(work_dir)


def multi_thread_test_limit_time(run_time_limit):
    # add program
    program_queue = Queue()
    for _ in range(main_configure_baseline.program_max_num):
        program_queue.put(_)
    # create and run programs
    ts = []
    for core_num in range(main_configure_baseline.n_thread):
        t = Process(target=do_test, args=(run_time_limit, program_queue, core_num, ))
        t.daemon = True
        t.start()
        ts.append(t)
    # wait
    while True:
        old_size = program_queue.qsize()
        time.sleep(GeneralOperationConf.time_limit * 3)
        new_size = program_queue.qsize()
        if old_size == new_size:
            break
    # release resource
    for _ in ts:
        _.terminate()
    program_queue.close()


if __name__ == '__main__':
    # initialize and check
    common_base_funs.rm_file(main_configure_baseline.test_dir)
    common_base_funs.mkdir_if_not_exists(main_configure_baseline.test_dir)
    required_log_dir = '/'.join(main_configure_baseline.log_prefix.split('/')[:-1])
    common_base_funs.rm_file(required_log_dir)
    common_base_funs.mkdir_p_if_not_exists(required_log_dir)
    if main_configure_baseline.use_seed:
        assert os.path.exists(MainProcessConf.seed_file)
        MainProcessConf.seeds = common_base_funs.get_file_lines(MainProcessConf.seed_file)
    if main_configure_baseline.baseline_type == main_configure_baseline.baseline_types[1]:    # hicond
        assert os.path.exists(main_configure_baseline.hicond_conf_dir)
        for _ in main_configure_baseline.hicond_confs:
            assert os.path.exists(_)

    # test
    core_time_limit = main_configure_baseline.run_time_limit / main_configure_baseline.n_thread
    multi_thread_test_limit_time(core_time_limit)
