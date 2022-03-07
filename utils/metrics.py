import numpy as np
import sys

import time, logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(handler)

import sklearn

class TAREvaluator(object):
    @staticmethod
    def report_TP_at_FP_thres(same_distances, diff_distances):
        # report true positive rate at a false positive rate
        n_same = same_distances.size
        n_diff = diff_distances.size
        scores = 1-np.concatenate((same_distances, diff_distances))
        labels = np.concatenate((np.ones(n_same), -np.ones(n_diff)))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)

        return tpr, fpr, thresholds

    @staticmethod
    def compute_roc(same_pair_dist, diff_pair_dist, eval_name):
        same_pair_dist = np.asarray(same_pair_dist)
        diff_pair_dist = np.asarray(diff_pair_dist)
        tpr, fpr, thresholds = TAREvaluator().report_TP_at_FP_thres(same_pair_dist, diff_pair_dist)
        def cal_tp_thres(fpr, tpr, thresholds, fp_th=0.01):
            idx = np.argmin(np.abs(fpr - fp_th))
            th = 1-thresholds[idx]
            return tpr[idx], fpr[idx], th

        tpr2, fpr2, th2 = cal_tp_thres(fpr, tpr, thresholds, fp_th=0.01)
        tpr3, fpr3, th3 = cal_tp_thres(fpr, tpr, thresholds, fp_th=0.001)
        tpr4, fpr4, th4 = cal_tp_thres(fpr, tpr, thresholds, fp_th=0.0001)
        tpr5, fpr5, th5 = cal_tp_thres(fpr, tpr, thresholds, fp_th=0.00001)
        this_stat = [
                [tpr2, tpr3, tpr4],
                [th2, th3, th4],
                ]

        logger.info('evaluation dataset is ' + eval_name)
        logger.info('\tsame_pairs are {0}, diff_pairs are {1}'.format(str(same_pair_dist.size), str(diff_pair_dist.size)))
        logger.info('\ttpr={0}, dist_th={1}, fpr={2}'.format('%.5f'%tpr2, '%.6f'%th2, '%.5f'%fpr2))
        logger.info('\ttpr={0}, dist_th={1}, fpr={2}'.format('%.5f'%tpr3, '%.6f'%th3, '%.5f'%fpr3))
        logger.info('\ttpr={0}, dist_th={1}, fpr={2}'.format('%.5f' % tpr4, '%.6f'%th4, '%.5f' % fpr4))
        return this_stat


class AvgMetric:
    def __init__(self):
        self.total_sum = 0.
        self.count = 0
        self.max_ = 0.
    
    def update(self, item):
        self.total_sum += item
        self.count += 1
        self.max_ = max(self.max_, item)
    
    @property
    def avg(self):
        return self.total_sum / (max(1, self.count))
    
    @property
    def total(self):
        return self.total_sum
    
    @property
    def max(self):
        return self.max_