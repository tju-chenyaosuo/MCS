# -*- encoding=utf-8 -*-
import copy

import common_base_funs

'''
This file is to control configuration in each test.
'''


class GeneralOperationConf:
    time_limit = 60
    limit_memory_script = './limit-memory-run.sh'


class GCCTestConf:
    gcc = ''
    default_opts = ['-O1', '-O2', '-O3', '-Os']
    compile_res = 'a.out'
    compile_err = 'error'
    O0_execute_res = 'O0.res'
    o_execute_res = 'o.res'
    execute_err = 'error'
    diff = 'diff'
    compile_timeout_recorder = 'compile_timeout'
    compile_crash_recorder = 'compile_crash'
    execute_timeout_recorder = 'execute_timeout'
    execute_crash_recorder = 'execute_crash'
    mis_copmpile_recorder = 'mis_compile'
    csmith_lib = ''


class LLVMTestConf:
    llvm_bin_path = ''
    default_opts = ['-O1', '-O2', '-O3', '-Os']
    compile_res = 'a.out'
    compile_err = 'error'
    O0_execute_res = 'O0.res'
    o_execute_res = 'o.res'
    execute_err = 'error'
    diff = 'diff'
    compile_timeout_recorder = 'compile_timeout'
    compile_crash_recorder = 'compile_crash'
    execute_timeout_recorder = 'execute_timeout'
    execute_crash_recorder = 'execute_crash'
    mis_copmpile_recorder = 'mis_compile'
    csmith_lib = ''
    llvm_ub_clang = ''


class CsmithConfigurationConf:
    csmith = ''
    single_keys = ["more_struct_union_type_prob",
                   "bitfields_creation_prob",
                   "bitfields_signed_prob",
                   "bitfield_in_normal_struct_prob",
                   "scalar_field_in_full_bitfields_struct_prob",
                   "exhaustive_bitfield_prob",
                   "safe_ops_signed_prob",
                   "select_deref_pointer_prob",
                   "regular_volatile_prob",
                   "regular_const_prob",
                   "stricter_const_prob",
                   "looser_const_prob",
                   "field_volatile_prob",
                   "field_const_prob",
                   "std_unary_func_prob",
                   "shift_by_non_constant_prob",
                   "pointer_as_ltype_prob",
                   "struct_as_ltype_prob",
                   "union_as_ltype_prob",
                   "float_as_ltype_prob",
                   "new_array_var_prob",
                   "access_once_var_prob",
                   "inline_function_prob",
                   "builtin_function_prob"]
    group_keys = ["statement_prob", "assign_unary_ops_prob",
                  "assign_binary_ops_prob", "simple_types_prob",
                  "safe_ops_size_prob"]
    group_items_keys = {
        "statement_prob": ["statement_assign_prob", "statement_block_prob",
                           "statement_for_prob", "statement_ifelse_prob",
                           "statement_return_prob", "statement_continue_prob",
                           "statement_break_prob", "statement_goto_prob",
                           "statement_arrayop_prob"],
        "assign_unary_ops_prob": ["unary_plus_prob", "unary_minus_prob",
                                  "unary_not_prob", "unary_bit_not_prob"],
        "assign_binary_ops_prob": ["binary_add_prob", "binary_sub_prob",
                                   "binary_mul_prob", "binary_div_prob",
                                   "binary_mod_prob", "binary_gt_prob",
                                   "binary_lt_prob", "binary_ge_prob",
                                   "binary_le_prob", "binary_eq_prob",
                                   "binary_ne_prob", "binary_and_prob",
                                   "binary_or_prob", "binary_bit_xor_prob",
                                   "binary_bit_and_prob", "binary_bit_or_prob",
                                   "binary_bit_rshift_prob", "binary_bit_lshift_prob"],
        "simple_types_prob": ["void_prob", "char_prob", "int_prob",
                              "short_prob", "long_prob", "long_long_prob",
                              "uchar_prob", "uint_prob", "ushort_prob",
                              "ulong_prob", "ulong_long_prob", "float_prob"],
        "safe_ops_size_prob": ["safe_ops_size_int8", "safe_ops_size_int16",
                               "safe_ops_size_int32", "safe_ops_size_int64"]
    }

    untune_options = [
        ## fix 6 non-effective options, corresponding features do not change (into a non-zero value)
        ## no matter how option probs are.
        # 'exhaustive_bitfield_prob',
        # 'stricter_const_prob',
        # 'union_as_ltype_prob',
        # 'access_once_var_prob',
        # 'inline_function_prob',
        # 'builtin_function_prob',
        ## set 4 options to 0, because csmith disable those by default.
        'float_as_ltype_prob',
        'statement_prob:statement_block_prob',
        'simple_types_prob:void_prob',
        'simple_types_prob:float_prob',
    ]

    untune_single_cnt = len([u for u in untune_options if ':' not in u])
    untune_group_cnt = len([u for u in untune_options if ':' in u])

    tune_single_cnt = len(single_keys) - untune_single_cnt
    tune_group_cnt = 47 - untune_group_cnt

    hicond_confs = ['hicond-conf/config1', 'hicond-conf/config2', 'hicond-conf/config3', 'hicond-conf/config4',
                    'hicond-conf/config5', 'hicond-conf/config6', 'hicond-conf/config7', 'hicond-conf/config8',
                    'hicond-conf/config9', 'hicond-conf/config10']


class A2cConfigure:
    clusinfo_dir = {
        "clusinfo3": [0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2, 0, 2, 1, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2, 1, 2, 0, 1, 1, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                      2, 0, 1, 0, 1],
        "clusinfo4": [2, 2, 2, 2, 2, 3, 1, 1, 0, 0, 3, 0, 3, 1, 0, 0, 0, 1, 3, 3, 1, 3, 3, 3, 1, 3, 0, 1, 1, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                      3, 0, 1, 0, 1],
        "clusinfo5": [2, 2, 2, 2, 2, 4, 1, 1, 0, 0, 4, 0, 4, 1, 0, 0, 0, 1, 4, 4, 1, 4, 4, 4, 1, 4, 0, 1, 1, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 4, 3, 0, 3, 1, 0, 1, 0, 1, 1, 1,
                      4, 0, 1, 0, 1],
        "clusinfo6": [2, 2, 2, 2, 2, 5, 1, 1, 0, 0, 5, 0, 5, 1, 0, 0, 0, 1, 5, 5, 1, 5, 5, 5, 1, 5, 0, 1, 1, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 5, 3, 0, 3, 1, 0, 1, 0, 1, 4, 1,
                      5, 0, 1, 0, 1],
        "clusinfo7": [2, 2, 2, 2, 2, 6, 1, 1, 5, 0, 6, 5, 6, 1, 0, 0, 0, 1, 6, 6, 1, 6, 6, 6, 1, 6, 0, 1, 1, 5, 5, 0, 0,
                      0, 0, 1, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 6, 3, 0, 3, 1, 0, 1, 0, 1, 4, 1,
                      6, 0, 1, 0, 1],
        "clusinfo8": [2, 2, 2, 2, 2, 7, 1, 1, 5, 0, 7, 5, 7, 1, 0, 0, 0, 1, 7, 7, 1, 7, 7, 7, 1, 7, 0, 1, 1, 5, 5, 0, 0,
                      0, 6, 1, 6, 0, 4, 0, 4, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 7, 3, 0, 3, 1, 0, 1, 0, 1, 4, 1,
                      7, 0, 1, 0, 1],
        "clusinfo9": [2, 2, 2, 2, 2, 8, 1, 1, 5, 0, 8, 5, 8, 1, 0, 0, 0, 1, 8, 8, 1, 8, 8, 8, 1, 8, 0, 1, 1, 5, 5, 7, 0,
                      0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 3, 7, 3, 0, 0, 3, 0, 3, 0, 8, 3, 7, 3, 1, 0, 1, 0, 1, 4, 1,
                      8, 0, 1, 0, 1],
        "clusinfo10": [2, 2, 2, 2, 2, 9, 1, 1, 5, 0, 9, 5, 9, 8, 0, 0, 0, 8, 9, 9, 1, 9, 9, 9, 1, 9, 0, 1, 1, 5, 5, 7,
                       0, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 3, 7, 3, 0, 0, 3, 0, 3, 0, 9, 3, 7, 3, 1, 0, 1, 0, 1,
                       4, 1, 9, 0, 1, 0, 1],
        "clusinfo11": [2, 2, 2, 2, 2, 10, 1, 1, 5, 9, 10, 5, 10, 8, 0, 0, 9, 8, 10, 10, 1, 10, 10, 10, 1, 10, 9, 1, 1,
                       5, 5, 7, 0, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 3, 7, 3, 0, 0, 3, 0, 3, 9, 10, 3, 7, 3, 1, 0,
                       1, 0, 1, 4, 1, 10, 9, 1, 0, 1],
        "clusinfo12": [2, 2, 2, 2, 2, 11, 1, 1, 5, 9, 11, 5, 11, 8, 0, 0, 9, 8, 11, 11, 1, 11, 11, 11, 1, 11, 9, 1, 1,
                       5, 5, 7, 0, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 11, 10, 7, 3, 1,
                       0, 1, 0, 1, 4, 1, 11, 9, 1, 0, 1],
        "clusinfo13": [2, 2, 2, 2, 2, 12, 1, 1, 5, 9, 12, 5, 12, 8, 0, 0, 9, 8, 12, 12, 1, 12, 12, 12, 1, 12, 9, 1, 1,
                       5, 5, 7, 0, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 12, 10, 7, 3, 1,
                       0, 11, 0, 11, 4, 1, 12, 9, 1, 0, 1],
        "clusinfo14": [12, 2, 2, 2, 2, 13, 1, 1, 5, 9, 13, 5, 13, 8, 0, 0, 9, 8, 13, 13, 1, 13, 13, 13, 1, 13, 9, 1, 1,
                       5, 5, 7, 0, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 13, 10, 7, 3, 1,
                       0, 11, 0, 11, 4, 1, 13, 9, 1, 0, 1],
        "clusinfo15": [12, 2, 2, 2, 2, 14, 1, 1, 5, 9, 14, 5, 14, 8, 0, 0, 9, 8, 14, 14, 1, 14, 14, 14, 1, 14, 9, 1, 1,
                       5, 5, 7, 13, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 14, 10, 7, 3, 1,
                       0, 11, 0, 11, 4, 1, 14, 9, 1, 0, 1],
        "clusinfo16": [12, 2, 2, 2, 2, 15, 1, 1, 5, 9, 15, 5, 15, 8, 0, 0, 9, 8, 15, 15, 1, 15, 15, 15, 1, 15, 9, 1, 1,
                       5, 5, 7, 13, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 15, 10, 7, 3, 1,
                       0, 11, 14, 11, 4, 1, 15, 9, 1, 0, 1],
        "clusinfo17": [12, 2, 2, 2, 2, 16, 1, 1, 5, 9, 16, 5, 16, 8, 0, 0, 9, 8, 16, 16, 1, 16, 16, 16, 1, 16, 9, 1, 1,
                       5, 5, 7, 13, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 16, 10, 7, 3, 1,
                       0, 11, 14, 11, 4, 15, 16, 9, 1, 0, 1],
        "clusinfo18": [12, 2, 2, 2, 2, 17, 1, 1, 5, 9, 17, 5, 17, 8, 0, 0, 9, 8, 17, 17, 1, 17, 17, 17, 1, 17, 9, 1, 1,
                       5, 5, 7, 13, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 17, 10, 7, 3,
                       16, 0, 11, 14, 11, 4, 15, 17, 9, 1, 0, 1],
        "clusinfo19": [12, 2, 2, 2, 2, 18, 1, 1, 5, 17, 18, 5, 18, 8, 0, 0, 9, 8, 18, 18, 1, 18, 18, 18, 1, 18, 9, 1, 1,
                       5, 5, 7, 13, 0, 6, 1, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 18, 10, 7, 3,
                       16, 0, 11, 14, 11, 4, 15, 18, 9, 1, 0, 1],
        "clusinfo20": [12, 2, 2, 2, 2, 19, 1, 1, 5, 17, 19, 5, 19, 8, 0, 0, 9, 8, 19, 19, 18, 19, 19, 19, 1, 19, 9, 1,
                       1, 5, 5, 7, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 19, 10, 7,
                       3, 16, 0, 11, 14, 11, 4, 15, 19, 9, 1, 0, 1],
        "clusinfo21": [12, 2, 2, 2, 2, 20, 1, 1, 19, 17, 20, 5, 20, 8, 0, 0, 9, 8, 20, 20, 18, 20, 20, 20, 1, 20, 9, 1,
                       1, 5, 5, 7, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 20, 10, 7,
                       3, 16, 0, 11, 14, 11, 4, 15, 20, 9, 1, 0, 1],
        "clusinfo22": [12, 2, 2, 2, 2, 21, 1, 1, 19, 17, 21, 5, 21, 8, 0, 0, 9, 8, 21, 21, 18, 21, 21, 21, 1, 21, 9, 1,
                       1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 21, 10, 7,
                       3, 16, 0, 11, 14, 11, 4, 15, 21, 9, 1, 0, 1],
        "clusinfo23": [12, 21, 21, 21, 2, 22, 1, 1, 19, 17, 22, 5, 22, 8, 0, 0, 9, 8, 22, 22, 18, 22, 22, 22, 1, 22, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 22, 10,
                       7, 3, 16, 0, 11, 14, 11, 4, 15, 22, 9, 1, 0, 1],
        "clusinfo24": [12, 21, 21, 21, 2, 23, 1, 1, 19, 17, 23, 5, 23, 8, 0, 0, 9, 8, 23, 23, 18, 23, 23, 23, 1, 23, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 23, 10,
                       7, 3, 16, 0, 11, 14, 11, 4, 15, 23, 9, 1, 0, 22],
        "clusinfo25": [12, 21, 21, 21, 2, 24, 1, 1, 19, 17, 24, 5, 24, 8, 0, 0, 9, 8, 24, 24, 18, 24, 24, 24, 1, 24, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 24, 10,
                       7, 3, 16, 23, 11, 14, 11, 4, 15, 24, 9, 1, 0, 22],
        "clusinfo26": [12, 21, 21, 21, 2, 25, 1, 1, 19, 17, 25, 24, 25, 8, 0, 0, 9, 8, 25, 25, 18, 25, 25, 25, 1, 25, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 25, 10,
                       7, 3, 16, 23, 11, 14, 11, 4, 15, 25, 9, 1, 0, 22],
        "clusinfo27": [12, 21, 21, 21, 2, 26, 1, 1, 19, 17, 26, 24, 26, 8, 0, 0, 9, 8, 26, 26, 18, 26, 26, 26, 1, 26, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 26, 25,
                       7, 3, 16, 23, 11, 14, 11, 4, 15, 26, 9, 1, 0, 22],
        "clusinfo28": [12, 21, 21, 21, 2, 27, 1, 1, 19, 17, 27, 24, 27, 8, 0, 0, 9, 8, 27, 27, 18, 27, 27, 27, 1, 27, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 27, 25,
                       7, 3, 16, 23, 26, 14, 11, 4, 15, 27, 9, 1, 0, 22],
        "clusinfo29": [12, 21, 27, 21, 2, 28, 1, 1, 19, 17, 28, 24, 28, 8, 0, 0, 9, 8, 28, 28, 18, 28, 28, 28, 1, 28, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 28, 25,
                       7, 3, 16, 23, 26, 14, 11, 4, 15, 28, 9, 1, 0, 22],
        "clusinfo30": [12, 21, 27, 21, 2, 29, 1, 1, 19, 17, 29, 24, 29, 8, 0, 0, 9, 8, 29, 29, 18, 29, 29, 29, 1, 29, 9,
                       1, 1, 5, 5, 20, 13, 0, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 29, 25,
                       28, 3, 16, 23, 26, 14, 11, 4, 15, 29, 9, 1, 0, 22],
        "clusinfo31": [12, 21, 27, 21, 2, 30, 1, 1, 19, 17, 30, 24, 30, 8, 0, 0, 9, 8, 30, 30, 18, 30, 30, 30, 1, 30, 9,
                       1, 1, 5, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 30, 25,
                       28, 3, 16, 23, 26, 14, 11, 4, 15, 30, 9, 1, 0, 22],
        "clusinfo32": [12, 21, 27, 21, 2, 31, 1, 1, 19, 17, 31, 24, 31, 8, 0, 0, 9, 8, 31, 31, 18, 31, 31, 31, 1, 31, 9,
                       1, 1, 5, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 31, 25,
                       28, 3, 16, 23, 26, 14, 11, 30, 15, 31, 9, 1, 0, 22],
        "clusinfo33": [12, 21, 27, 21, 2, 32, 1, 1, 19, 17, 32, 24, 32, 8, 0, 0, 9, 8, 32, 32, 31, 32, 32, 32, 1, 32, 9,
                       1, 1, 5, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 32, 25,
                       28, 3, 16, 23, 26, 14, 11, 30, 15, 32, 9, 1, 0, 22],
        "clusinfo34": [12, 21, 27, 21, 2, 33, 1, 1, 19, 17, 33, 24, 33, 8, 0, 0, 9, 32, 33, 33, 31, 33, 33, 33, 1, 33,
                       9, 1, 1, 5, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 33,
                       25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 33, 9, 1, 0, 22],
        "clusinfo35": [12, 21, 27, 21, 2, 34, 1, 1, 19, 17, 34, 24, 34, 8, 0, 0, 9, 32, 34, 34, 31, 34, 34, 34, 1, 34,
                       9, 1, 1, 33, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9, 34,
                       25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 34, 9, 1, 0, 22],
        "clusinfo36": [12, 21, 27, 21, 2, 35, 1, 34, 19, 17, 35, 24, 35, 8, 0, 0, 9, 32, 35, 35, 31, 35, 35, 35, 34, 35,
                       9, 34, 34, 33, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 0, 0, 10, 7, 3, 0, 0, 3, 0, 10, 9,
                       35, 25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 35, 9, 1, 0, 22],
        "clusinfo37": [12, 21, 27, 21, 2, 36, 1, 34, 19, 17, 36, 24, 36, 8, 0, 0, 9, 32, 36, 36, 31, 36, 36, 36, 34, 36,
                       9, 34, 34, 33, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0, 10, 9,
                       36, 25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 36, 9, 1, 0, 22],
        "clusinfo38": [12, 21, 27, 21, 2, 37, 1, 34, 19, 17, 37, 24, 37, 8, 0, 0, 36, 32, 37, 37, 31, 37, 37, 37, 34,
                       37, 9, 34, 34, 33, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 0, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 9, 37, 25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 37, 9, 1, 0, 22],
        "clusinfo39": [12, 21, 27, 21, 2, 38, 1, 34, 19, 17, 38, 24, 38, 8, 0, 0, 36, 32, 38, 38, 31, 38, 38, 38, 34,
                       38, 9, 34, 34, 33, 5, 20, 13, 29, 6, 18, 6, 0, 4, 0, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 9, 38, 25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 38, 9, 1, 0, 22],
        "clusinfo40": [12, 21, 27, 21, 2, 39, 1, 34, 19, 17, 39, 24, 39, 8, 0, 0, 36, 32, 39, 39, 31, 39, 39, 39, 34,
                       39, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 0, 4, 0, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 9, 39, 25, 28, 3, 16, 23, 26, 14, 11, 30, 15, 39, 9, 1, 0, 22],
        "clusinfo41": [12, 21, 27, 21, 2, 40, 1, 34, 19, 17, 40, 24, 40, 8, 0, 0, 36, 32, 40, 40, 31, 40, 40, 40, 34,
                       40, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 0, 4, 0, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 9, 40, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 40, 9, 1, 0, 22],
        "clusinfo42": [12, 21, 27, 21, 2, 41, 1, 34, 19, 17, 41, 24, 41, 8, 0, 0, 36, 32, 41, 41, 31, 41, 41, 41, 34,
                       41, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 0, 4, 40, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 9, 41, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 41, 9, 1, 0, 22],
        "clusinfo43": [12, 21, 27, 21, 2, 42, 1, 34, 19, 17, 42, 24, 42, 8, 0, 0, 36, 32, 42, 42, 31, 42, 42, 42, 34,
                       42, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 0, 4, 40, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3, 0,
                       10, 41, 42, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 42, 9, 1, 0, 22],
        "clusinfo44": [12, 21, 27, 21, 2, 43, 1, 34, 19, 17, 43, 24, 43, 8, 0, 0, 36, 32, 43, 43, 31, 43, 43, 43, 34,
                       43, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 35, 0, 10, 7, 3, 0, 35, 3,
                       0, 10, 41, 43, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 43, 9, 1, 0, 22],
        "clusinfo45": [12, 21, 27, 21, 2, 44, 1, 34, 19, 17, 44, 24, 44, 8, 43, 43, 36, 32, 44, 44, 31, 44, 44, 44, 34,
                       44, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 35, 43, 10, 7, 3, 43, 35, 3,
                       43, 10, 41, 44, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 44, 9, 1, 43, 22],
        "clusinfo46": [12, 21, 27, 21, 2, 45, 1, 34, 19, 17, 45, 24, 45, 8, 43, 43, 36, 32, 45, 45, 31, 45, 45, 45, 34,
                       45, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 35, 43, 10, 7, 3, 43, 35, 3,
                       44, 10, 41, 45, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 45, 9, 1, 43, 22],
        "clusinfo47": [12, 21, 27, 21, 2, 46, 1, 34, 19, 17, 46, 24, 46, 8, 45, 45, 36, 32, 46, 46, 31, 46, 46, 46, 34,
                       46, 9, 34, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 35, 45, 10, 7, 3, 43, 35, 3,
                       44, 10, 41, 46, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 46, 9, 1, 45, 22],
        "clusinfo48": [12, 21, 27, 21, 2, 47, 1, 34, 19, 17, 47, 24, 47, 8, 45, 45, 36, 32, 47, 47, 31, 47, 47, 47, 34,
                       47, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 35, 45, 10, 7, 3, 43, 35, 3,
                       44, 10, 41, 47, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 47, 9, 1, 45, 22],
        "clusinfo49": [12, 21, 27, 21, 2, 48, 1, 34, 19, 17, 48, 24, 48, 8, 45, 45, 36, 32, 48, 48, 31, 48, 48, 48, 34,
                       48, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 45, 10, 7, 3, 43, 35, 3,
                       44, 10, 41, 48, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 48, 9, 1, 45, 22],
        "clusinfo50": [12, 21, 27, 21, 2, 49, 1, 34, 19, 17, 49, 24, 49, 8, 48, 45, 36, 32, 49, 49, 31, 49, 49, 49, 34,
                       49, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 45, 10, 7, 3, 43, 35, 3,
                       44, 10, 41, 49, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 49, 9, 1, 45, 22],
        "clusinfo51": [12, 21, 27, 21, 2, 50, 1, 34, 19, 17, 50, 24, 50, 8, 48, 45, 36, 32, 50, 50, 31, 50, 50, 50, 34,
                       50, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 45, 10, 7, 49, 43, 35,
                       3, 44, 10, 41, 50, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 50, 9, 1, 45, 22],
        "clusinfo52": [12, 21, 27, 21, 2, 51, 1, 34, 19, 17, 51, 24, 51, 8, 48, 45, 36, 32, 51, 51, 31, 51, 51, 51, 34,
                       51, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 10, 41, 51, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 51, 9, 1, 45, 22],
        "clusinfo53": [12, 21, 27, 51, 2, 52, 1, 34, 19, 17, 52, 24, 52, 8, 48, 45, 36, 32, 52, 52, 31, 52, 52, 52, 34,
                       52, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 10, 41, 52, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 52, 9, 1, 45, 22],
        "clusinfo54": [12, 21, 27, 51, 2, 53, 1, 34, 19, 17, 53, 24, 53, 8, 48, 45, 36, 32, 53, 53, 31, 53, 53, 53, 34,
                       53, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 53, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 53, 9, 1, 45, 22],
        "clusinfo55": [12, 21, 27, 51, 2, 54, 1, 34, 19, 17, 54, 24, 54, 8, 48, 45, 36, 32, 54, 54, 31, 54, 54, 54, 34,
                       54, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 54, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 54, 9, 1, 53, 22],
        "clusinfo56": [12, 21, 27, 51, 2, 55, 1, 34, 19, 17, 55, 24, 55, 8, 48, 45, 36, 32, 55, 55, 31, 55, 55, 55, 34,
                       55, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 4, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 55, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 55, 54, 1, 53, 22],
        "clusinfo57": [12, 21, 27, 51, 2, 56, 1, 34, 19, 17, 56, 24, 56, 8, 48, 45, 36, 32, 56, 56, 31, 56, 56, 56, 34,
                       56, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 55, 37, 7, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 56, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 56, 54, 1, 53, 22],
        "clusinfo58": [12, 21, 27, 51, 2, 57, 1, 34, 19, 17, 57, 24, 57, 8, 48, 45, 36, 32, 57, 57, 31, 57, 57, 57, 34,
                       57, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 55, 37, 56, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 57, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 57, 54, 1, 53, 22],
        "clusinfo59": [12, 21, 27, 51, 2, 58, 57, 34, 19, 17, 58, 24, 58, 8, 48, 45, 36, 32, 58, 58, 31, 58, 58, 58, 34,
                       58, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 55, 37, 56, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 58, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 58, 54, 1, 53, 22],
        "clusinfo60": [12, 21, 27, 51, 2, 59, 57, 34, 19, 17, 59, 24, 59, 8, 48, 45, 36, 32, 59, 59, 31, 59, 59, 59, 58,
                       59, 9, 46, 34, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 55, 37, 56, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 59, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 59, 54, 1, 53, 22],
        "clusinfo61": [12, 21, 27, 51, 2, 60, 57, 34, 19, 17, 60, 24, 60, 8, 48, 45, 36, 32, 60, 60, 31, 60, 60, 60, 58,
                       60, 9, 46, 59, 33, 5, 20, 13, 29, 38, 18, 6, 42, 4, 40, 55, 37, 56, 0, 47, 50, 10, 7, 49, 43, 35,
                       3, 44, 52, 41, 60, 25, 28, 39, 16, 23, 26, 14, 11, 30, 15, 60, 54, 1, 53, 22],
        "clusinfo71": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    }
    n_single_agent = CsmithConfigurationConf.tune_single_cnt
    n_group_agent = CsmithConfigurationConf.tune_group_cnt
    single_action = [-5, 0, 5]
    group_action = [-3, 0, 3]
    lr = 0.01
    s_dim = 71
    update_iterations = 10

    model_dir = common_base_funs.add_sep('a2c_model')
    single_net_prefix = 'single_model'
    single_net_suffix = '.pkl'
    group_net_prefix = 'group_model'
    group_net_suffix = '.pkl'


class EnvironmentConf:
    program_cnt = [50, 100, 250][1]
    timeout_percentages = 0.2
    n_thread = 20
    # reward
    # reward_types = ['dv', 'dvn', 'orig', 'dv_avg_n']    # V2
    # [diversity_reward, avg_predict_prob, buggy, boundary_values], zero means off.    # V2
    # reward_weight = [1, 0, 1, 1]    # V2
    # [diversity_reward], zero means off.    # V2
    quality_weight = [1]
    buggy_reward = 4
    boundary_punish = -buggy_reward * 0.5
    reset_reward = -buggy_reward * 0.5
    boundary_threshold = 5
    boundary_num_threshold = 20
    # dv_avg_n
    dv_avg_n = 10
    # reward_type = reward_types[3]    # V2
    # fix_unique_ = True    # when its value is True, reward_weight[2] == 1    # V2
    # fix_unique_value = 0.1    # V2
    # assert (fix_unique_ and reward_weight[2] == 1) or not fix_unique_    # V2
    diversity_k = 10
    delta_rate = 1
    diversity_compensation_factor = None
    enable_all_program_feature = True
    feature_standardization = True
    standardization_data = []
    standardization_file = 'default_hicond_random.csv'
    standardization_mu = []
    standardization_std = []

    # ga conf
    ga_max_step = 5
    ga_inheritance_prob = 0.2
    ga_parent_threshold = 0  # 0.2, 0.02
    ga_dist_ratio = 0.4

    """
    global reset types
    default-ase-sample:
        Reset environment through default setting and ase settings, more frequently a setting is selected before, 
        less probability it will be select later.
    default-ase-ga:
        Reuse old-ga implementation and initialize the {config_history} with default setting and ase settings.
    """
    global_reset_types = ['default-ase-sample', 'default-ase-ga']
    global_reset_type = global_reset_types[0]
    reset_reward_history_types = ['zero', 'original', 'clean']
    reset_reward_history_type = reset_reward_history_types[0]

    n_single_agent = CsmithConfigurationConf.tune_single_cnt
    n_group_agent = CsmithConfigurationConf.tune_group_cnt
    action_single = A2cConfigure.single_action
    action_group = A2cConfigure.group_action

    # predict model
    predict_model_path = 'predict.pkl'
    new_fail_predict = False
    # timeout configure model
    timeout_model_path = 'timeoutModel/timeoutPrediction.pkl'
    use_timeout_model = True
    timeout_meta_colums = copy.deepcopy(CsmithConfigurationConf.single_keys)
    for k in CsmithConfigurationConf.group_keys:
        timeout_meta_colums.extend(CsmithConfigurationConf.group_items_keys[k])
    timeout_threshold = 0.8

    # dynamic configurations
    group_item_keys = []
    for item in CsmithConfigurationConf.group_items_keys:
        for key in CsmithConfigurationConf.group_items_keys[item]:
            group_item_keys.append(item + ':' + key)
    all_csmith_items = CsmithConfigurationConf.single_keys + group_item_keys
    assert len(all_csmith_items) == 71

    # mapping {n_single_agent + n_group_agent}-size action to 71-size actions
    action_mapping = []
    reset_type = ['random', 'ase', 'ga', 'default', 'fine', 'default-ase-sample']
    un_tune_indexes = []


EnvironmentConf.action_mapping = [i for i in range(len(EnvironmentConf.all_csmith_items))
                                  if EnvironmentConf.all_csmith_items[i] not in CsmithConfigurationConf.untune_options]
EnvironmentConf.un_tune_indexes = [i for i in range(len(EnvironmentConf.all_csmith_items))
                                   if EnvironmentConf.all_csmith_items[i] in CsmithConfigurationConf.untune_options]


class ProgramFeatureConf:
    @staticmethod
    def option_2_feature_inx(option_names):
        feature_idxs = []
        for op in option_names:
            if ':' not in op:
                # single options
                idx = CsmithConfigurationConf.single_keys.index(op)
                feature_idxs.append(idx)
            else:
                g_name = op.split(':')[0]
                ele_name = op.split(':')[1]
                group_id = CsmithConfigurationConf.group_keys.index(g_name)
                bias = 0
                idx_base = 0
                if group_id == 0:
                    # statement_prob [24:33]
                    idx_base = 24
                    pos = CsmithConfigurationConf.group_items_keys[g_name].index(ele_name)
                    if pos >= 4:  # "pInvokeCnt" [24+4]
                        bias += 1
                elif group_id == 1:
                    # assign_unary_ops_prob [49:52]
                    idx_base = 49
                elif group_id == 2:
                    # assign_binary_ops_prob [53:70]
                    idx_base = 53
                elif group_id == 3:
                    # simple_types_prob [71:82]
                    idx_base = 71
                elif group_id == 4:
                    # safe_ops_size_prob [83:86]
                    idx_base = 83
                assert idx_base != 0
                bias += CsmithConfigurationConf.group_items_keys[g_name].index(ele_name)
                idx = idx_base + bias
                feature_idxs.append(idx)
        return feature_idxs

    feature_vector_keys = [
        "pMoreStructUnionCnt",
        "pBitFieldsCreationCnt",
        "pBitFieldsSignedCnt",
        "pBitFieldInNormalStructCnt",
        "pScalarFieldInFullBitFieldsCnt",
        "pExhaustiveBitFieldsCnt",
        "pSafeOpsSignedCnt",
        "pSelectDerefPointerCnt",
        "pRegularVolatileCnt",
        "pRegularConstCnt",
        "pStricterConstCnt",
        "pLooserConstCnt",
        "pFieldVolatileCnt",
        "pFieldConstCnt",
        "pStdUnaryFuncCnt",
        "pShiftByNonConstantCnt",
        "pPointerAsLTypeCnt",
        "pStructAsLTypeCnt",
        "pUnionAsLTypeCnt",
        "pFloatAsLTypeCnt",
        "pNewArrayVariableCnt",
        "pAccessOnceVariableCnt",
        "pInlineFunctionCnt",
        "pBuiltinFunctionCnt",  # SINGLE [0:23]
        "pAssignCnt",
        "pBlockCnt",
        "pForCnt",
        "pIfElseCnt",
        "pInvokeCnt",  ## [24+4]
        "pReturnCnt",
        "pContinueCnt",
        "pBreakCnt",
        "pGotoCnt",
        "pArrayOpCnt",  # statement_prob [24:33]
        "pSimpleAssignCnt",
        "pMulAssignCnt",
        "pDivAssignCnt",
        "pRemAssignCnt",
        "pAddAssignCnt",
        "pSubAssignCnt",
        "pLShiftAssignCnt",
        "pRShiftAssignCnt",
        "pBitAndAssignCnt",
        "pBitXorAssignCnt",
        "pBitOrAssignCnt",
        "pPreIncrCnt",
        "pPreDecrCnt",
        "pPostIncrCnt",
        "pPostDecrCnt",  # [34:48]
        "pPlusCnt",
        "pMinusCnt",
        "pNotCnt",
        "pBitNotCnt",  # assign_unary_ops_prob [49:52]
        "pAddCnt",
        "pSubCnt",
        "pMulCnt",
        "pDivCnt",
        "pModCnt",
        "pCmpGtCnt",
        "pCmpLtCnt",
        "pCmpGeCnt",
        "pCmpLeCnt",
        "pCmpEqCnt",
        "pCmpNeCnt",
        "pAndCnt",
        "pOrCnt",
        "pBitXorCnt",
        "pBitAndCnt",
        "pBitOrCnt",
        "pRShiftCnt",
        "pLShiftCnt",  # assign_binary_ops_prob [53:70]
        "pVoidCnt",
        "pCharCnt",
        "pIntCnt",
        "pShortCnt",
        "pLongCnt",
        "pLongLongCnt",
        "pUCharCnt",
        "pUIntCnt",
        "pUShortCnt",
        "pULongCnt",
        "pULongLongCnt",
        "pFloatCnt",  # simple_types_prob [71:82]
        "pInt8Cnt",
        "pInt16Cnt",
        "pInt32Cnt",
        "pInt64Cnt",  # safe_ops_size_prob [83:86]
        "pMoreStructUnionTotalCnt",
        "pBitFieldsCreationTotalCnt",
        "pBitFieldsSignedTotalCnt",
        "pBitFieldInNormalStructTotalCnt",
        "pScalarFieldInFullBitFieldsTotalCnt",
        "pExhaustiveBitFieldsTotalCnt",
        "pSafeOpsSignedTotalCnt",
        "pSelectDerefPointerTotalCnt",
        "pRegularVolatileTotalCnt",
        "pRegularConstTotalCnt",
        "pStricterConstTotalCnt",
        "pLooserConstTotalCnt",
        "pFieldVolatileTotalCnt",
        "pFieldConstTotalCnt",
        "pStdUnaryFuncTotalCnt",
        "pShiftByNonConstantTotalCnt",
        "pPointerAsLTypeTotalCnt",
        "pStructAsLTypeTotalCnt",
        "pUnionAsLTypeTotalCnt",
        "pFloatAsLTypeTotalCnt",
        "pNewArrayVariableTotalCnt",
        "pAccessOnceVariableTotalCnt",
        "pInlineFunctionTotalCnt",
        "pBuiltinFunctionTotalCnt"  # [87:111]
    ]
    """
    options that can not be controlled with Csmith configuration file
    """
    default_discount_features = [
        # anonymous group
        "pSimpleAssignCnt",
        "pMulAssignCnt",
        "pDivAssignCnt",
        "pRemAssignCnt",
        "pAddAssignCnt",
        "pSubAssignCnt",
        "pLShiftAssignCnt",
        "pRShiftAssignCnt",
        "pBitAndAssignCnt",
        "pBitXorAssignCnt",
        "pBitOrAssignCnt",
        "pPreIncrCnt",
        "pPreDecrCnt",
        "pPostIncrCnt",
        "pPostDecrCnt",  # [34:48]
        # total count
        "pMoreStructUnionTotalCnt",
        "pBitFieldsCreationTotalCnt",
        "pBitFieldsSignedTotalCnt",
        "pBitFieldInNormalStructTotalCnt",
        "pScalarFieldInFullBitFieldsTotalCnt",
        "pExhaustiveBitFieldsTotalCnt",
        "pSafeOpsSignedTotalCnt",
        "pSelectDerefPointerTotalCnt",
        "pRegularVolatileTotalCnt",
        "pRegularConstTotalCnt",
        "pStricterConstTotalCnt",
        "pLooserConstTotalCnt",
        "pFieldVolatileTotalCnt",
        "pFieldConstTotalCnt",
        "pStdUnaryFuncTotalCnt",
        "pShiftByNonConstantTotalCnt",
        "pPointerAsLTypeTotalCnt",
        "pStructAsLTypeTotalCnt",
        "pUnionAsLTypeTotalCnt",
        "pFloatAsLTypeTotalCnt",
        "pNewArrayVariableTotalCnt",
        "pAccessOnceVariableTotalCnt",
        "pInlineFunctionTotalCnt",
        "pBuiltinFunctionTotalCnt"  # [87:111]
    ]

    default_discount_feature_inx = []
    for f in default_discount_features:
        default_discount_feature_inx.append(feature_vector_keys.index(f))
    untune_discount_feature_inx = []


ProgramFeatureConf.untune_discount_feature_inx = ProgramFeatureConf.option_2_feature_inx(CsmithConfigurationConf.untune_options)


class MainProcessConf:
    retrain = False

    run_time_limit = 10 * 24 * 60 * 60

    use_seed = True

    seed_file = 'seed.txt'
    seeds = []

    csmith_conf_dir = common_base_funs.add_sep('csmith-config')
    csmith_conf_file_prefix = csmith_conf_dir + 'config'
    csmith_tmp_conf = csmith_conf_dir + 'tmp-c'

    compiler_types = ['gcc', 'llvm']
    test_compiler_type = compiler_types[0]

    test_dir_prefix = common_base_funs.add_sep('test_dir')

    # os.environ['PYTHONHASHSEED'] = str(rl_seed)
    # torch.cuda.manual_seed(rl_seed)s
    # torch.backends.cudnn.deterministic = True

    log_dir = common_base_funs.add_sep('log')
    log_reward = log_dir + 'reward.txt'
    log_reward_detail = log_dir + 'reward_detail.txt'
    log_test_info = log_dir + 'test_info.txt'
    log_env = log_dir + 'env.txt'
    log_action_ids = log_dir + 'action_ids.txt'
    log_total = log_dir + 'total.txt'
    log_detailed_test_info = log_dir + 'detailed_test_info.txt'
    log_time = log_dir + 'time_log.txt'
    log_model = log_dir + 'model_log.txt'
    log_fail_features = log_dir + 'fail_features.txt'
    log_children = log_dir + 'ga_children.txt'
    log_delta_quality = log_dir + 'delta_quality.txt'
    log_reset_cfg_history = log_dir + 'reset_conf.txt'
    log_dir_counter_action = common_base_funs.add_sep(log_dir + 'action_cnt')
    log_dir_config_cnt = common_base_funs.add_sep(log_dir + 'config_cnt')
    log_policy = log_dir + 'policy.txt'
    log_timeout_model = log_dir + 'timeout_model.txt'


if __name__ == '__main__':
    # test clusinfo_dir
    # for key in A2cConfigure.clusinfo_dir:
    #     print(key + ':' + str(len(list(set(A2cConfigure.clusinfo_dir[key])))) + ':' + str(max(A2cConfigure.clusinfo_dir[key])))
    #     assert len(list(set(A2cConfigure.clusinfo_dir[key]))) == max(A2cConfigure.clusinfo_dir[key]) + 1

    # test function option_2_feature_inx
    all_csmith_group_item_keys = []
    for item in CsmithConfigurationConf.group_keys:
        for key in CsmithConfigurationConf.group_items_keys[item]:
            all_csmith_group_item_keys.append(item + ':' + key)
    all_csmith_options = CsmithConfigurationConf.single_keys + all_csmith_group_item_keys
    related_program_features = ProgramFeatureConf.option_2_feature_inx(all_csmith_options)
    print('\n'.join([all_csmith_options[inx] + ' => ' + ProgramFeatureConf.feature_vector_keys[related_program_features[inx]]
                     for inx in range(len(related_program_features))]))

