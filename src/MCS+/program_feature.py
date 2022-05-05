# -*- encoding=utf-8 -*-
import time

import common_base_funs
import numpy as np
from scipy.spatial.distance import pdist


class ProgramFeature:
    @staticmethod
    def load_feature(feature_file):
        feature = common_base_funs.get_file_lines(feature_file)[0].split(',')
        return [float(f) for f in feature]

    @staticmethod
    def load_features(feature_file):
        features = common_base_funs.get_file_lines(feature_file)
        features = [_.split(',') for _ in features]
        return [[float(_) for _ in f] for f in features]

    @staticmethod
    def k_cosine_distance(x, y, default_val, untune_feature_inx, k, compensation_factor=None):
        """
        Get the average of k smallest cosine distance between x and y
        :param x: 1*111 program feature
        :param y: m*111 program features, typically history program features
        :param default_val: return {default_val} if x or y is empty
        :param untune_feature_inx: untune program feautre related to untune csmith optimizations
        :param k: the number of cosine distance used to calculate the result
        :param compensation_factor: prevent the result of this function become smaller and smaller, with len(y) increases.
        :return: The average of k smallest cosine distance between x and y
        """
        if len(y) == 0:
            return default_val
        m = np.vstack([x, y])
        m = np.delete(m, tuple(untune_feature_inx), axis=1)
        dist2 = pdist(m, 'cosine')
        # rely on the order of pdist' return value.
        cosine_ds = dist2[:len(y)]
        rt = np.mean(sorted(cosine_ds)[:k])
        w = pow(compensation_factor, len(m)) if compensation_factor is not None else 1
        return rt * w

    @staticmethod
    def k_manhattan_distance(x, y, default_val, untune_feature_inx, k, compensation_factor=None):
        if len(y) == 0:
            return default_val
        m = np.vstack([x, y])
        m = np.delete(m, tuple(untune_feature_inx), axis=1)
        dist2 = pdist(m, 'cityblock')
        # rely on the order of pdist' return value.
        cosine_ds = dist2[:len(y)]
        rt = np.mean(sorted(cosine_ds)[:k])
        if compensation_factor is not None:
            w = pow(compensation_factor, len(m))
        else:
            w = 1
        return rt * w

    @staticmethod
    def k_euclidean_distance(x, y, default_val, untune_feature_inx, k, compensation_factor=None):
        if len(y) == 0:
            return default_val
        m = np.vstack([x, y])
        m = np.delete(m, tuple(untune_feature_inx), axis=1)
        dist2 = pdist(m, 'euclidean')
        # rely on the order of pdist' return value.
        cosine_ds = dist2[:len(y)]
        rt = np.mean(sorted(cosine_ds)[:k])
        if compensation_factor is not None:
            w = pow(compensation_factor, len(m))
        else:
            w = 1
        return rt * w


if __name__ == '__main__':
    # def cosine_distance(x, y, rt_val_empty, uncount_idxs, k=1):
    #     """
    #     :param x: 1 * 111-len(uncount_idxs)
    #     :param y: m * 111-len(uncount_idxs)
    #     :param rt_val_empty: return 0 for inter-cosine-reward, and 1 for unique-reward
    #     :param uncount_idxs: unused feature while calculating distance
    #     :param k: mean distance of k smallest distance
    #     :return: float, mean distance of k smallest cosine distances between x and each item in y
    #     """
    #     if len(y) == 0:
    #         return rt_val_empty
    #     else:
    #         from scipy.spatial.distance import pdist
    #         m = np.vstack([x, y])
    #         m = np.delete(m, tuple(uncount_idxs), axis=1)
    #         dist2 = pdist(m, 'cosine')
    #         cosine_ds = dist2[:len(y)]
    #         rt = np.mean(sorted(cosine_ds)[:k])
    #     return rt * (pow(1.001, len(m)))
    #
    # import random
    # from main_configure_approach import ProgramFeatureConf
    # for _ in range(100):
    #     start_time = time.time()
    #     feature1 = [random.randint(0, 1000000) for i in range(111)]
    #     features = [[random.randint(0, 1000000) for i in range(111)] for j in range(random.randint(0, 10000))]
    #     a = ProgramFeature.k_cosine_distance(feature1, features, 0, ProgramFeatureConf.untune_discount_feature_inx, k=1,
    #                                          compensation_factor=1.001)
    #     b = cosine_distance(feature1, features, 0, ProgramFeatureConf.untune_discount_feature_inx, k=1)
    #     assert a == b, str(a) + '!=' + str(b)
    #     end_time = time.time()
    #     print(str(len(features)) + ':' + str(end_time - start_time))
    pass