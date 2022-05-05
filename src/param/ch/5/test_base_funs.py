import time

from main_configure_approach import GeneralOperationConf, EnvironmentConf, CsmithConfigurationConf
from main_configure_approach import GCCTestConf, LLVMTestConf, MainProcessConf
from program_feature import ProgramFeature
from multiprocessing import Process
from multiprocessing import Queue
import common_base_funs
import os


# test program generation
class SrcGenerator(object):
    def __init__(self, src):
        assert isinstance(src, str), 'source code path should be str'
        self.src = src

    def gen_src(self):
        raise NotImplementedError


class CpSrcGenerator(SrcGenerator):
    def __init__(self, source, src):
        """
        To generate source code for testing via cp command
        :param source: source code
        :param src: target
        """
        super().__init__(src)
        assert isinstance(source, str), 'source code path should be str'
        assert isinstance(src, str), 'src path should be str'
        self.source = source

    def gen_src(self):
        cmd = ' '.join(['cp', self.source, self.src])
        os.system(cmd)


class HicondSrcGenerateStatus:
    timeout = 'program-gen-timeout'
    crash = 'program-gen-crash'
    success = 'program-gen-success'
    status = [timeout, crash, success]


class HicondSrcGenerator(SrcGenerator):
    def __init__(self, src, hicond=CsmithConfigurationConf.csmith, feature_file=None, conf=None, seed=None,
                 random=False,
                 err_file=None, time_limit=GeneralOperationConf.time_limit):
        """
        Generate test program via hicond
        :param src: target src file
        :param hicond: csmith modified by hicond
        :param feature_file: feature file output by hicond
        :param conf: configure file
        :param seed: seed
        :param random: random option of hicond
        :param time_limit: time limit for test program generation
        """
        super().__init__(src)
        assert os.path.exists(hicond)
        assert feature_file is None or isinstance(feature_file, str)
        assert conf is None or os.path.exists(conf)
        assert seed is None or isinstance(seed, str)
        assert isinstance(random, bool)
        assert err_file is None or isinstance(err_file, str)
        '''
        random=false
        or 
        random=true, and conf is None
        '''
        assert not random or conf is None
        assert isinstance(time_limit, int) or isinstance(time_limit, float)
        self.hicond = hicond
        self.conf = conf
        self.feature_file = feature_file
        self.seed = seed
        self.random = random
        self.time_limit = time_limit
        self.err_file = err_file

    def gen_src(self):
        cmd_conf = ' '.join(['--probability-configuration', self.conf]) if self.conf is not None else ''
        cmd_seed = ' '.join(['--seed', self.seed]) if self.seed is not None else ''
        cmd_rand = '--random-random' if self.random else ''
        cmd_feature = ' '.join(['--record-file', self.feature_file]) if self.feature_file is not None else ''
        cmd_warn = '2>' + self.err_file if self.err_file is not None else ''
        cmd_src = '1>' + self.src
        params = [self.hicond, cmd_feature, cmd_seed, cmd_conf, cmd_rand]
        params = [p for p in params if len(p) != 0]
        cmd_hicond = '"' + ' '.join(params) + '"'
        limit_memory_cmd = ' '.join([GeneralOperationConf.limit_memory_script, cmd_hicond, cmd_src, cmd_warn])
        common_base_funs.rm_file(self.src)
        common_base_funs.rm_file(self.err_file)

        t = time.time()
        time_in = common_base_funs.execmd_limit_time(limit_memory_cmd, self.time_limit)
        t = time.time() - t
        if not time_in:
            return HicondSrcGenerateStatus.timeout, t
        if not os.path.exists(self.feature_file) or os.path.getsize(self.feature_file) == 0:
            return HicondSrcGenerateStatus.crash, t
        return HicondSrcGenerateStatus.success, t

    def get_seed(self):
        seed = 'None'
        if os.path.exists(self.src):
            seed = common_base_funs.get_file_lines(self.src)[6].split(':')[-1].strip()
        return seed


# Abstract class for compiler testing
class TestProcess(object):
    def do_test(self):
        raise NotImplementedError

    def do_dol(self):
        raise NotImplementedError


class TestState:
    success = 'success'


# GCC test code
class GCCTestStatus(TestState):
    compile_timeout = 'compile_timeout'
    compile_crash = 'compile_crash'
    execute_timeout = 'execute_timeout'
    execute_crash = 'execute_crash'
    mis_compile = 'mis_compile'


class GCCTestProcess(TestProcess):
    def __init__(self, src, work_dir, libs, time_limit=GeneralOperationConf.time_limit,
                 opts=GCCTestConf.default_opts, gcc=GCCTestConf.gcc):
        assert isinstance(libs, list)
        assert len(libs) == 0 or isinstance(libs[0], str)
        assert os.path.exists(gcc)
        assert isinstance(opts, list)
        assert len(opts) == 0 or isinstance(opts[0], str)
        assert os.path.exists(src)
        assert os.path.exists(work_dir)
        assert isinstance(time_limit, str) and time_limit.isnumeric() or isinstance(time_limit, int) or isinstance(
            time_limit, float)
        self.gcc = gcc
        self.libs = ' '.join(['-I ' + lib for lib in libs])
        self.src = src
        self.work_dir = common_base_funs.add_sep(work_dir)
        self.opts = opts[:]
        self.time_limit = time_limit

    def test_compile(self, opt):
        compile_res = self.work_dir + GCCTestConf.compile_res
        compile_err = self.work_dir + GCCTestConf.compile_err
        compile_timeout_recorder = self.work_dir + GCCTestConf.compile_timeout_recorder
        compile_crash_recorder = self.work_dir + GCCTestConf.compile_crash_recorder
        common_base_funs.rm_file(compile_res)
        common_base_funs.rm_file(compile_err)
        cmd = ' '.join([self.gcc, self.libs, opt + ' -fno-strict-aliasing -fwrapv', self.src, '-o', compile_res, '2>'+compile_err])
        t = time.time()
        time_in = common_base_funs.execmd_limit_time(cmd, self.time_limit)
        t = time.time() - t
        if not time_in:
            err = common_base_funs.get_file_content(compile_err)
            common_base_funs.put_file_content(compile_timeout_recorder, opt + '\n' + err)
            return GCCTestStatus.compile_timeout, t
        if not os.path.exists(compile_res) or os.path.getsize(compile_res) == 0:
            err = common_base_funs.get_file_content(compile_err)
            common_base_funs.put_file_content(compile_crash_recorder, opt + '\n' + err)
            return GCCTestStatus.compile_crash, t
        return GCCTestStatus.success, t

    def test_execute(self, opt, execute_res):
        compile_res = self.work_dir + GCCTestConf.compile_res
        execute_err = self.work_dir + GCCTestConf.execute_err
        execute_timeout_recorder = self.work_dir + GCCTestConf.execute_timeout_recorder
        execute_crash_recorder = self.work_dir + GCCTestConf.execute_crash_recorder
        common_base_funs.rm_file(execute_err)
        common_base_funs.rm_file(execute_res)
        cmd = ' '.join([compile_res, '1>', execute_res, '2>', execute_err])

        t = time.time()
        time_in = common_base_funs.execmd_limit_time(cmd, self.time_limit)
        t = time.time() - t
        if not time_in:
            err = common_base_funs.get_file_content(execute_err)
            common_base_funs.put_file_content(execute_timeout_recorder, opt + '\n' + err)
            return GCCTestStatus.execute_timeout, t
        if not os.path.exists(execute_res) or os.path.getsize(execute_res) == 0:
            err = common_base_funs.get_file_content(execute_err)
            common_base_funs.put_file_content(execute_crash_recorder, opt + '\n' + err)
            return GCCTestStatus.execute_crash, t
        return GCCTestStatus.success, t

    def diff(self, file1, file2, opt):
        mis_compile_recorder = self.work_dir + GCCTestConf.mis_copmpile_recorder
        diff_recorder = self.work_dir + GCCTestConf.diff
        common_base_funs.rm_file(diff_recorder)
        diff_res = common_base_funs.diff(file1, file2)
        if len(diff_res) == 0:
            return GCCTestStatus.success
        else:
            err = diff_res
            common_base_funs.put_file_content(mis_compile_recorder, opt + '\n' + err)
            return GCCTestStatus.mis_compile

    def do_test(self):
        return self.do_dol()

    def do_dol(self):
        O0_execute_res = self.work_dir + GCCTestConf.O0_execute_res
        o_execute_res = self.work_dir + GCCTestConf.o_execute_res
        time_spt = []

        res, t = self.test_compile('-O0')
        time_spt.append(t)
        if res != GCCTestStatus.success:
            return res, sum(time_spt)

        res, t = self.test_execute('-O0', O0_execute_res)
        time_spt.append(t)
        if res != GCCTestStatus.success:
            return res, sum(time_spt)

        for o in self.opts:
            res, t = self.test_compile(o)
            time_spt.append(t)
            if res != GCCTestStatus.success:
                return res, sum(time_spt)

            res, t = self.test_execute(o, o_execute_res)
            time_spt.append(t)
            if res != GCCTestStatus.success:
                return res, sum(time_spt)

            res = self.diff(O0_execute_res, o_execute_res, o)
            if res != GCCTestStatus.success:
                return res, sum(time_spt)
        return GCCTestStatus.success, sum(time_spt)


class GCCInfo:
    def __init__(self, gcc=GCCTestConf.gcc):
        assert isinstance(gcc, str), 'gcc should be a str'
        self.gcc = gcc

    @staticmethod
    def filter_list(l, s):
        res = []
        for elem in l:
            if s in elem:
                res.append(elem)
        return res

    @staticmethod
    def list_strip_each(l):
        res = []
        for elem in l:
            res.append(elem.strip())
        return res

    @staticmethod
    def list_get_first_word_each(l):
        res = []
        for elem in l:
            res.append(elem[:elem.find(' ')])
        return res

    def get_ox_opts(self, level):
        opt3 = common_base_funs.execmd(self.gcc + ' ' + level + ' -Q --help=optimizers').split('\n')
        opt3 = GCCInfo.filter_list(opt3, '[enabled]')
        opt3 = GCCInfo.list_strip_each(opt3)
        opt3 = GCCInfo.list_get_first_word_each(opt3)
        return opt3


# LLVM test code
class LLVMTestState(TestState):
    compile_timeout = 'compile_timeout'
    compile_crash = 'compile_crash'
    execute_timeout = 'execute_timeout'
    execute_crash = 'execute_crash'
    mis_compile = 'mis_compile'
    ub = 'undefined-behavior'


class LLVMUBCheckerState:
    compile_timeout = 'compile-timeout'
    compile_crash = 'compile-crash'
    execute_timeout = 'execute-timeout'
    normal = 'normal'
    ub = 'undefined-behavior'


class LLVMUBChecker:
    def __init__(self, src, libs, llvm_ub_clang=LLVMTestConf.llvm_ub_clang):
        assert os.path.exists(llvm_ub_clang)
        assert os.path.exists(src)
        assert isinstance(libs, str)
        self.clang = llvm_ub_clang
        self.src = src
        self.libs = libs
        self.work_dir = common_base_funs.add_sep('/'.join(src.split('/')[:-1]))

    def has_ub(self):
        """
        Check whether a program have undefined-behavior
        :return: states in {LLVMUBChecker}
        """
        compile_out = self.work_dir + 'a.ubc.out'
        compile_err = self.work_dir + 'a.ubc.err'
        execute_out = self.work_dir + 'a.ubc.res'
        execute_err = self.work_dir + 'a.ubc.err'
        common_base_funs.rm_file(compile_out)
        common_base_funs.rm_file(compile_err)
        cmd = ' '.join(
            [self.clang, self.libs, '-fsanitize=undefined,address', self.src, '-o', compile_out, '2>' + compile_err])
        if not common_base_funs.execmd_limit_time(cmd, GeneralOperationConf.time_limit):
            return LLVMUBCheckerState.compile_timeout
        if not os.path.exists(compile_out) or os.path.getsize(compile_out) == 0:
            return LLVMUBCheckerState.compile_crash
        common_base_funs.rm_file(execute_out)
        common_base_funs.rm_file(execute_err)
        cmd = ' '.join([compile_out, '1>' + execute_out, '2>' + execute_err])
        time_in = common_base_funs.execmd_limit_time(cmd, GeneralOperationConf.time_limit)
        err = common_base_funs.get_file_content(execute_err)
        '''
        This if statement should be first,
        because a timeout execution can print ub message 
        '''
        if 'undefined-behavior' in err or 'ERROR' in err:
            return LLVMUBCheckerState.ub
        if not time_in:
            return LLVMUBCheckerState.execute_timeout
        return LLVMUBCheckerState.normal


class LLVMTestProcess(TestProcess):
    def __init__(self, src, libs, work_dir, time_limit=GeneralOperationConf.time_limit,
                 opts=LLVMTestConf.default_opts, llvm_bin_path=LLVMTestConf.llvm_bin_path):
        assert isinstance(libs, list)
        assert len(libs) == 0 or isinstance(libs[0], str)
        assert os.path.exists(llvm_bin_path)
        assert isinstance(opts, list)
        assert len(opts) == 0 or isinstance(opts[0], str)
        assert os.path.exists(src)
        assert os.path.exists(work_dir)
        assert isinstance(time_limit, str) and time_limit.isnumeric() or isinstance(time_limit, int) or isinstance(
            time_limit, float)
        llvm_bin_path = common_base_funs.add_sep(llvm_bin_path)
        # self.llvm_opt = llvm_bin_path + 'opt'
        self.llvm_clang = llvm_bin_path + 'clang'
        # self.llvm_llvm_as = llvm_bin_path + 'llvm-as'
        self.libs = ' '.join(['-I ' + lib for lib in libs])
        self.src = src
        self.work_dir = common_base_funs.add_sep(work_dir)
        self.opts = opts[:]
        self.time_limit = time_limit
        # assert os.path.exists(self.llvm_opt)
        assert os.path.exists(self.llvm_clang)
        # assert os.path.exists(self.llvm_llvm_as)

    def test_compile(self, opt):
        compile_res = self.work_dir + LLVMTestConf.compile_res
        compile_err = self.work_dir + LLVMTestConf.compile_err
        compile_timeout_recorder = self.work_dir + LLVMTestConf.compile_timeout_recorder
        compile_crash_recorder = self.work_dir + LLVMTestConf.compile_crash_recorder
        common_base_funs.rm_file(compile_res)
        common_base_funs.rm_file(compile_err)
        cmd = ' '.join([self.llvm_clang, self.libs, opt, self.src, '-o', compile_res, '2>', compile_err])

        t = time.time()
        time_in = common_base_funs.execmd_limit_time(cmd, self.time_limit)
        t = time.time() - t
        if not time_in:
            err = common_base_funs.get_file_content(compile_err)
            common_base_funs.put_file_content(compile_timeout_recorder, opt + '\n' + err)
            return LLVMTestState.compile_timeout, t
        if not os.path.exists(compile_res) or os.path.getsize(compile_res) == 0:
            err = common_base_funs.get_file_content(compile_err)
            common_base_funs.put_file_content(compile_crash_recorder, opt + '\n' + err)
            return LLVMTestState.compile_crash, t
        return LLVMTestState.success, t

    def test_execute(self, opt, execute_res):
        compile_res = self.work_dir + LLVMTestConf.compile_res
        execute_err = self.work_dir + LLVMTestConf.execute_err
        execute_timeout_recorder = self.work_dir + LLVMTestConf.execute_timeout_recorder
        execute_crash_recorder = self.work_dir + LLVMTestConf.execute_crash_recorder
        common_base_funs.rm_file(execute_err)
        common_base_funs.rm_file(execute_res)
        cmd = ' '.join([compile_res, '1>', execute_res, '2>', execute_err])

        t = time.time()
        time_in = common_base_funs.execmd_limit_time(cmd, self.time_limit)
        t = time.time() - t
        if not time_in:
            err = common_base_funs.get_file_content(execute_err)
            common_base_funs.put_file_content(execute_timeout_recorder, opt + '\n' + err)
            return LLVMTestState.execute_timeout, t
        if not os.path.exists(execute_res) or os.path.getsize(execute_res) == 0:
            err = common_base_funs.get_file_content(execute_err)
            common_base_funs.put_file_content(execute_crash_recorder, opt + '\n' + err)
            return LLVMTestState.execute_crash, t
        return LLVMTestState.success, t

    def diff(self, file1, file2, opt):
        mis_compile_recorder = self.work_dir + LLVMTestConf.mis_copmpile_recorder
        diff_res = common_base_funs.diff(file1, file2)
        if len(diff_res) == 0:
            return LLVMTestState.success
        else:
            err = diff_res
            common_base_funs.put_file_content(mis_compile_recorder, opt + '\n' + err)
            return LLVMTestState.mis_compile

    def do_test(self):
        return self.do_dol()

    def do_dol(self):
        spts = []

        O0_execute_res = self.work_dir + LLVMTestConf.O0_execute_res
        o_execute_res = self.work_dir + LLVMTestConf.o_execute_res

        res, t = self.test_compile('-O0')
        spts.append(t)
        if res != LLVMTestState.success:
            return res, sum(spts)

        res, t = self.test_execute('-O0', O0_execute_res)
        spts.append(t)
        if res != LLVMTestState.success:
            return res, sum(spts)

        for o in self.opts:
            res, t = self.test_compile(o)
            spts.append(t)
            if res != LLVMTestState.success:
                return res, sum(spts)

            res, t = self.test_execute(o, o_execute_res)
            spts.append(t)
            if res != LLVMTestState.success:
                lubc = LLVMUBChecker(self.src, self.libs)
                ub_res = lubc.has_ub()
                res = LLVMTestState.ub if ub_res == LLVMUBCheckerState.ub else res
                return res, sum(spts)

            res = self.diff(O0_execute_res, o_execute_res, o)
            if res != LLVMTestState.success:
                lubc = LLVMUBChecker(self.src, self.libs)
                ub_res = lubc.has_ub()
                res = LLVMTestState.ub if ub_res in [LLVMUBCheckerState.ub] else res
                return res, sum(spts)
        return LLVMTestState.success, sum(spts)


class TestResult:
    invalid_results = [HicondSrcGenerateStatus.timeout, HicondSrcGenerateStatus.crash,
                       GCCTestStatus.compile_timeout, GCCTestStatus.execute_timeout,
                       LLVMTestState.compile_timeout, LLVMTestState.execute_timeout,
                       LLVMTestState.ub]
    bug_results = [GCCTestStatus.compile_crash, GCCTestStatus.execute_crash, GCCTestStatus.mis_compile,
                   LLVMTestState.compile_crash, LLVMTestState.execute_crash, LLVMTestState.mis_compile]
    normal_results = [GCCTestStatus.success, LLVMTestState.success, HicondSrcGenerateStatus.success]


def test_process(conf_id, compiler_type, todo_queue, program_info_queue, timeout_queue):
    """
    Given Csmith's configuration, do compiler testing, collect following data:
        1. program_id: {configure_number}-{program_number}
        2. feature: program feature generated by Csmith
        3. result: test result, in {HicondSrcGenerateStatus.states} or {GCCTestStatus} | {LLVMTestConf}
        4. time_spt: time spent on each testing.
        5. seed
        Note that: all programs will be record, even generate failed one.
    And collect following files if timeout or bug occurs:
        1. Src file (if exists)
        2. Error Message File during test (even Csmith's crash message)
    Note that:
        1. All data will be recorded
        2. When number of timeout program over threshold, test process will be omit, at that time, program_num is waste,
            all processes will finished when todo_queue is empty
        3. This program ensures that all timeout_queue update first, program_queue update later.
            Because of the end condition.
        4. program_info_queue.qsize() == program_number or timeout is the end condition. When meet end condition,
            this program ensures that all queue is updated.
    :param conf_id: configuration number
    :param compiler_type: new it is either gcc nor llvm, in {MainProcessConf.compiler_types}
    :param todo_queue: contain {program_number}, each number represent a program
    :param program_info_queue: [[program_id, feature, result, time spent], ...]
    :param timeout_queue: [timeout_program_id1, ...]
    :return: None
    """
    while not todo_queue.empty():
        program_cnt = todo_queue.get()
        if timeout_queue.qsize() >= EnvironmentConf.timeout_percentages * EnvironmentConf.program_cnt:  # gen program
            program_info_queue.put([])
            continue
        program_id = str(conf_id) + '-' + str(program_cnt)
        test_dir = common_base_funs.add_sep(MainProcessConf.test_dir_prefix + program_id)
        common_base_funs.rm_file(test_dir)
        common_base_funs.mkdir_if_not_exists(test_dir)
        src = test_dir + 'a.c'
        feature_file = test_dir + 'a.csv'
        gen_err_file = test_dir + 'gen-err.txt'
        conf_num = (conf_id - 1) * EnvironmentConf.program_cnt + program_cnt
        if MainProcessConf.use_seed and conf_num < len(MainProcessConf.seeds):
            seed = MainProcessConf.seeds[conf_num]
        else:
            seed = None
        hsg = HicondSrcGenerator(src=src, feature_file=feature_file,
                                 conf=MainProcessConf.csmith_conf_file_prefix + str(conf_id), err_file=gen_err_file,
                                 seed=seed)
        res, gen_program_time = hsg.gen_src()
        if res != HicondSrcGenerateStatus.success:  # record gen program fail
            timeout_queue.put(program_id)
            program_info_queue.put([program_id, [], res, gen_program_time, str(seed)])
            print(str(program_id) + '-res:' + res + ',time:' + str(gen_program_time))
            continue
        if seed is None:
            seed = hsg.get_seed()
        tp = None
        if compiler_type == MainProcessConf.compiler_types[0]:
            tp = GCCTestProcess(libs=[GCCTestConf.csmith_lib], src=src, work_dir=test_dir)
        if compiler_type == MainProcessConf.compiler_types[1]:
            tp = LLVMTestProcess(libs=[LLVMTestConf.csmith_lib], src=src, work_dir=test_dir)
        assert tp is not None
        res, test_program_time = tp.do_test()
        print(str(program_id) + '-res:' + res + ',time:' + str(gen_program_time + test_program_time))
        if res in TestResult.invalid_results:
            timeout_queue.put(program_id)
        program_info_queue.put([program_id, ProgramFeature.load_feature(feature_file), res, gen_program_time + test_program_time, seed])
        if res == TestState.success:
            common_base_funs.rm_file(test_dir)


def multi_process_test_via_conf(conf_id):
    """
    Do compiler testing via multi-processing.
    Collect following data: See return
    Collect following file:
        1. Src file (if exists)
        2. Error Message File during test (even Csmith's crash message)
    :param conf_id: configuration number
    :return:
        1. {test_info}
            <1> program_id: {configure_number}-{program_number}
            <2> feature: program feature generated by Csmith
            <3> result: test result, in {HicondSrcGenerateStatus.states} or {GCCTestStatus} | {LLVMTestConf}
            <4> time_spt: time spent on each testing.
            <5> seed: seed
        2. {timeout_cnt}
            size of {timeout_queue}, e.g. the number of programs that over time limitation.
        Note that: all program test_info will be recorded, until timeout program over the threshold.
    """
    todo_queue = Queue()
    for i in range(EnvironmentConf.program_cnt):
        todo_queue.put(i)
    program_info_queue = Queue()
    timeout_queue = Queue()
    ps = []
    for core_num in range(EnvironmentConf.n_thread):
        p = Process(target=test_process,
                    args=(conf_id, MainProcessConf.test_compiler_type, todo_queue, program_info_queue, timeout_queue,))
        p.daemon = True
        p.start()
        ps.append(p)
    # wait
    all_exe = program_info_queue.qsize() == EnvironmentConf.program_cnt
    while not all_exe:
        time.sleep(10)
        print('wait-info:' + str([todo_queue.qsize(), program_info_queue.qsize(), timeout_queue.qsize()]))
        all_exe = program_info_queue.qsize() == EnvironmentConf.program_cnt
    # result
    timeout_cnt = timeout_queue.qsize()
    test_info = []
    info_size = program_info_queue.qsize()
    get_time = 0
    while get_time != info_size:
        info = program_info_queue.get()
        if len(info) != 0:
            test_info.append(info[:])
        get_time += 1
        print('extract-program-info:' + str([len(test_info), program_info_queue.qsize(), get_time]))
    test_info_sorted = sorted([[int(test_info[_][0].split('-')[1]), test_info[_]] for _ in range(len(test_info))],
                              reverse=False)
    test_info_sorted = [_[1] for _ in test_info_sorted]
    # check
    no_omit = len(test_info) == EnvironmentConf.program_cnt
    timeout = EnvironmentConf.timeout_percentages * EnvironmentConf.program_cnt <= timeout_queue.qsize()
    assert no_omit or timeout
    # timeout
    timeout = EnvironmentConf.timeout_percentages * EnvironmentConf.program_cnt <= timeout_queue.qsize()
    if timeout:
        # check
        info_timeout_cnt = sum([1 for _ in test_info if _[2] in TestResult.invalid_results])
        assert info_timeout_cnt == timeout_queue.qsize()
        # truncate
        timeout_size = timeout_queue.qsize()
        timeout_info = []
        while len(timeout_info) != timeout_size:
            timeout_info.append(timeout_queue.get())
            print('extract-timeout-info:' + str([len(timeout_info), timeout_queue.qsize()]))
        assert len(timeout_info) == timeout_size
        timeout_info_sorted = sorted([[int(_.split('-')[1]), _] for _ in timeout_info], reverse=False)
        timeout_info_sorted = [_[1] for _ in timeout_info_sorted]
        timeout_limit = int(EnvironmentConf.timeout_percentages * EnvironmentConf.program_cnt)
        last_timeout_id = timeout_info_sorted[timeout_limit - 1]
        truncate_inx = -1
        for _ in range(len(test_info_sorted)):
            if test_info_sorted[_][0] == last_timeout_id:
                truncate_inx = _
                break
        assert truncate_inx >= EnvironmentConf.timeout_percentages * EnvironmentConf.program_cnt - 1
        test_info_sorted = test_info_sorted[:truncate_inx+1]
        timeout_cnt = timeout_limit
    # release resource
    for _ in ps:
        _.terminate()
    todo_queue.close()
    program_info_queue.close()
    timeout_queue.close()
    return test_info_sorted, timeout_cnt


if __name__ == '__main__':
    '''test code'''
    # # test HicondSrcGenerator
    # src = 'tmp/a.c'
    # feature = 'tmp/a.csv'
    # common_base_funs.rm_file(src)
    # common_base_funs.rm_file(feature)
    # hicond_gener = HicondSrcGenerator(src, feature_file=feature, conf='/data/data/dev/conf/config1',
    #                                   err_file='tmp/src_gen.err')
    # hicond_gener.gen_src()
    #
    # # A basic test for compiler test method
    # gcc_work_dir = 'tmp/tmp-gcc'
    # llvm_work_dir = 'tmp/tmp-llvm'
    # common_base_funs.rm_file(gcc_work_dir)
    # common_base_funs.rm_file(llvm_work_dir)
    # common_base_funs.mkdir_if_not_exists(gcc_work_dir)
    # common_base_funs.mkdir_if_not_exists(llvm_work_dir)
    # gcc_test_process = GCCTestProcess(src, work_dir=gcc_work_dir, libs=[GCCTestConf.csmith_lib])
    # gcc_test_process.do_test()
    # llvm_test_process = LLVMTestProcess(src=src, libs=[LLVMTestConf.csmith_lib], work_dir=llvm_work_dir)
    # llvm_test_process.do_test()
    #
    # # test for multi_thread_test_via_conf
    # test_info, timeout_cnt = multi_thread_test_via_conf(1)
    # test_info = '***************************\n'.join(['\n'.join(['program_id='+str(p[0]), 'program_feature='+str(p[1]), 'test_res='+str(p[2]), 'time_spent='+str(p[3])]) for p in test_info])
    # print(test_info)
    # print(timeout_cnt)
    pass

