'''
Numpy/Scipy based roc operations
'''

import numpy as np
import math
from scipy import optimize

def bce(y, y_pred_proba, k=1, log=True):
    """
        :param y: 0 or 1
        :param y_pred_proba: probabilistic output 0 to 1
        :param k: topk loss
        :return:
        """
    if log:
        loss = (y * (-np.log(y_pred_proba))) + ((1 - y) * (-np.log(1 - y_pred_proba)))
    else:
        loss = (y != (y_pred_proba > 0.5)).astype(float)
    if k:
        return np.partition(loss, -k)[-k:].mean()
    else:
        return loss.mean()


def threspauc(y, y_pred, thres, k=1):
    """
    :param y:
    :param y_pred:
    :param thres: numpy ndarray
    :param k:
    :return:
    """
    y_pos = np.reshape(y_pred[y > 0.5], (-1, 1))
    y_neg = np.reshape(y_pred[y < 0.5], (-1, 1))
    yy = y_pos - np.transpose(y_neg, (1, 0))
    bce = np.log(1. + np.exp(-yy))
    thres = np.reshape(thres, (-1, 1))
    loss = (np.clip(bce - thres, 0., None).sum() / k + thres.sum()) / len(y_pos)
    return loss

def auc(score_neg, score_pos, k=None, log=False):
    """
    :param score_neg:
    :param score_pos:
    :param groupindices:
    :param k: pAUC [0, beta] range
    :param log: logistic loss
    :return:
    """
    y_pos = np.reshape(score_pos, (-1, 1))
    y_neg = np.reshape(score_neg, (-1, 1))
    yy = y_pos - np.transpose(y_neg, (1, 0))
    if log:
        matrix = np.log(1. + np.exp(-yy))
        if k:
            return np.partition(matrix, -k, axis=1)[:, -k:].mean() # log is non-increasing
    else:
        matrix = np.greater(yy, 0).astype(float)
        if k:
            return np.partition(matrix, k, axis=1)[:, :k].mean() # 0-1 is increasing
    return matrix.mean()

def roc(score_neg, score_pos, n_thres=100):
    '''
    Numerical ROC by discrete thresholds
    Not accurate
    :param y: np.ndarray
    :param score:
    :param n_thres: int
    :return:
    '''
    # thresholds = np.linspace(max(max(score), max(y)), min(min(score), min(y)), n_thres)
    score = np.concatenate((score_pos, score_neg))
    thresholds = np.linspace(score.max(), score.min(), n_thres)
    fprs = np.zeros(n_thres)
    tprs = np.zeros(n_thres)
    for i in range(n_thres):
        t = thresholds[i]
        TP_t = np.sum(np.greater(score_pos, t).astype(float)) + 0.5 * np.sum(np.equal(score_pos, t).astype(float))
        TN_t = np.sum(np.less(score_neg, t).astype(float)) + 0.5 * np.sum(np.equal(score_neg, t).astype(float))
        FP_t = np.sum(np.greater(score_neg, t).astype(float)) + 0.5 * np.sum(np.equal(score_neg, t).astype(float))
        FN_t = np.sum(np.less(score_pos, t).astype(float)) + 0.5 * np.sum(np.equal(score_pos, t).astype(float))
        # Compute false positive rate for current threshold.
        FPR_t = FP_t / (FP_t + TN_t)
        fprs[i] = FPR_t
        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / (TP_t + FN_t)
        tprs[i] = TPR_t
    return fprs, tprs

def get_accuracy(score, y):
    '''
    binary accuracy by logit function
    '''
    pred = (score >= 0.5) * (max(y) - min(y)) + min(y)
    return sum(pred == y) / len(y)

def get_roc(score, y, n_thres=500):
    '''
    Numerical ROC by discrete thresholds
    Not accurate
    :param score: np.ndarray
    :param y: np.ndarray
    :param n_thres: int
    :return:
    '''
    # thresholds = np.linspace(max(max(score), max(y)), min(min(score), min(y)), n_thres)
    thresholds = np.linspace(max(score), min(score), n_thres)
    fprs = np.zeros(n_thres)
    tprs = np.zeros(n_thres)
    score_pos, score_neg = score[y == 1], score[y == -1]
    for i in range(n_thres):
        t = thresholds[i]
        TP_t = np.sum(np.greater(score_pos, t).astype(float)) + 0.5 * np.sum(np.equal(score_pos, t).astype(float))
        TN_t = np.sum(np.less(score_neg, t).astype(float)) + 0.5 * np.sum(np.equal(score_neg, t).astype(float))
        FP_t = np.sum(np.greater(score_neg, t).astype(float)) + 0.5 * np.sum(np.equal(score_neg, t).astype(float))
        FN_t = np.sum(np.less(score_pos, t).astype(float)) + 0.5 * np.sum(np.equal(score_pos, t).astype(float))
        # Classifier / label agree and disagreements for current threshold.
        # TP_t = np.logical_and(score > t, y == 1).sum()
        # TN_t = np.logical_and(score <= t, y == 0).sum()
        # FP_t = np.logical_and(score > t, y == 0).sum()
        # FN_t = np.logical_and(score <= t, y == 1).sum()
        # Compute false positive rate for current threshold.
        FPR_t = FP_t / (FP_t + TN_t)
        fprs[i] = FPR_t
        # Compute true  positive rate for current threshold.
        TPR_t = TP_t / (TP_t + FN_t)
        tprs[i] = TPR_t
    return fprs, tprs


def get_auc(score, y):
    '''
    Sub-optimal avoiding it
    :param score:
    :param y:
    :return:
    '''
    yy = np.reshape(y, (-1, 1))
    delta_y = yy - np.transpose(yy)
    n_diff = np.sum(np.not_equal(delta_y, 0).astype(float))
    if n_diff == 0:
        return 0.

    sc = np.reshape(score, (-1, 1))
    delta_sc = sc - np.transpose(sc)
    mask = np.greater(delta_y, 0)
    # We only select relevant instances to multiply
    prod = delta_sc[mask] * delta_y[mask]
    auc = np.mean(np.greater(prod, 0).astype(float)) + 0.5 * np.mean(np.equal(prod, 0).astype(float))

    return auc

def get_pauc(score, y, alpha=0., beta=0.5):
    '''
    Calculate the normalized pAUC
    :param score:
    :param y:
    :param alpha:
    :param beta:
    :return:
    '''
    assert beta >= 0. and beta <= 1. and alpha >= 0. and alpha <= beta, "Wrong range!"
    if alpha == beta:
        return 0.

    score_pos, y_pos = score[y == 1], y[y == 1]
    score_neg, y_neg = score[y == -1], y[y == -1]  # same index

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        return 0.
    # calculate the negative interval
    lower = math.floor(num_neg * alpha)
    upper = math.ceil(num_neg * beta)
    valid_idx = np.argsort(score_neg, axis=0).reshape(-1)[::-1][lower:upper]  # score shape [-1, 1]
    score_valid, y_valid = score_neg[valid_idx], y_neg[valid_idx]  # same index

    # actually it is redundant
    # yy_valid = np.reshape(y_valid, (-1, 1))
    # yy_pos = np.reshape(y_pos, (-1, 1))
    # delta_y = yy_pos - np.transpose(yy_valid)

    sc_valid = np.reshape(score_valid, (-1, 1))
    sc_pos = np.reshape(score_pos, (-1, 1))
    delta_sc = sc_pos - np.transpose(sc_valid)
    prod = delta_sc * 2. # 2 is coming from delta_y
    # This is the normalized pauc since we stretch the fpr
    # hence it is only meaningful when pauc > 0.5
    pauc = np.mean(np.greater(prod, 0).astype(float)) + 0.5 * np.mean(np.equal(prod, 0).astype(float))

    return pauc

def get_pauc_from_threshold(score, y, alpha=0., beta=0.5):
    '''
    Calculate the normalized pAUC from minimizing threshold
    Only works for 0 started pAUC
    Suboptimal
    :param score:
    :param y:
    :param alpha:
    :param beta:
    :return: float
    '''
    assert beta >= 0. and beta <= 1. and alpha >= 0. and alpha <= beta, "Wrong range!"
    if alpha == beta:
        return 0.

    score_pos, y_pos = score[y == 1], y[y == 1]
    score_neg, y_neg = score[y == -1], y[y == -1]  # same index

    num_pos = len(y_pos)
    num_neg = len(y_neg)
    if num_pos == 0 or num_neg == 0:
        return 0.

    sc_pos = np.reshape(score_pos, (-1, 1))
    sc_neg = np.reshape(score_neg, (-1, 1))
    delta_sc = sc_neg - np.transpose(sc_pos)
    sc_pos_avg_by_neg = np.mean(np.greater(delta_sc, 0.).astype(float), axis=1)

    lower = math.floor(num_neg * alpha)
    upper = math.ceil(num_neg * beta)

    def get_pauc_from_start(upper):
        '''
        Scipy solver
        :param upper:
        :return:
        '''
        def f(x):
            return np.maximum(sc_pos_avg_by_neg - x, 0.).sum() / upper + x
        result = optimize.minimize_scalar(f, method="golden")
        if result.success:
            pauc = 1. - f(result.x)
        else:
            return 0.
        pauc += 0.5 * np.mean(np.equal(delta_sc, 0.).astype(float))
        return pauc

    if alpha > 0.:
        pauc_upper = get_pauc_from_start(upper) * upper / num_neg
        pauc_lower = get_pauc_from_start(lower) * lower / num_neg
        return (pauc_upper - pauc_lower) * num_neg / (upper - lower)
    else:
        return get_pauc_from_start(upper)

# def get_rel_pauc_from_threshold(score, y, alpha=0., beta=0.5):
#     '''
#     Calculate the normalized pAUC from minimizing threshold
#     Only works for 0 started pAUC
#     Suboptimal
#     :param score:
#     :param y:
#     :param alpha:
#     :param beta:
#     :return: float
#     '''
#     assert beta >= 0. and beta <= 1. and alpha >= 0. and alpha <= beta, "Wrong range!"
#     if alpha == beta:
#         return 0.
#
#     score_pos, y_pos = score[y == 1], y[y == 1]
#     score_neg, y_neg = score[y == -1], y[y == -1]  # same index
#
#     num_pos = len(y_pos)
#     num_neg = len(y_neg)
#     if num_pos == 0 or num_neg == 0:
#         return 0.
#
#     sc_pos = np.reshape(score_pos, (-1, 1))
#     sc_neg = np.reshape(score_neg, (-1, 1))
#     delta_sc = sc_neg - np.transpose(sc_pos)
#     sc_pos_avg_by_neg = np.mean(np.greater(delta_sc, 0.).astype(float), axis=1)
#
#     lower = math.floor(num_neg * alpha)
#     upper = math.ceil(num_neg * beta)
#
#     def get_pauc_from_start(upper):
#         '''
#         Scipy solver
#         :param upper:
#         :return:
#         '''
#         def f(x):
#             return np.maximum(sc_pos_avg_by_neg - x, 0.).sum() / upper + x
#         result = optimize.minimize_scalar(f, method="golden")
#         if result.success:
#             pauc = 1. - f(result.x)
#         else:
#             return 0.
#         pauc += 0.5 * np.mean(np.equal(delta_sc, 0.).astype(float))
#         return pauc
#
#     if alpha > 0.:
#         pauc_upper = get_pauc_from_start(upper) * upper / num_neg
#         pauc_lower = get_pauc_from_start(lower) * lower / num_neg
#         return (pauc_upper - pauc_lower) * num_neg / (upper - lower)
#     else:
#         return get_pauc_from_start(upper)
#
# def get_incomplete_rel_pauc_from_threshold(score, y, alpha=0., beta=0.5, B=1024):
#     '''
#     Calculate the normalized pAUC from minimizing threshold
#     Only works for 0 started pAUC
#     Suboptimal
#     :param score:
#     :param y:
#     :param alpha:
#     :param beta:
#     :return: float
#     '''
#     assert beta >= 0. and beta <= 1. and alpha >= 0. and alpha <= beta, "Wrong range!"
#     if alpha == beta:
#         return 0.
#
#     score_pos, y_pos = score[y == 1], y[y == 1]
#     score_neg, y_neg = score[y == -1], y[y == -1]  # same index
#
#     num_pos = len(y_pos)
#     num_neg = len(y_neg)
#     if num_pos == 0 or num_neg == 0:
#         return 0.
#
#     sc_pos = np.reshape(score_pos, (-1, 1))
#     sc_neg = np.reshape(score_neg, (-1, 1))
#     delta_sc = sc_neg - np.transpose(sc_pos)
#     sc_pos_avg_by_neg = np.mean(np.greater(delta_sc, 0.).astype(float), axis=1)
#
#     lower = math.floor(num_neg * alpha)
#     upper = math.ceil(num_neg * beta)
#
#     def get_pauc_from_start(upper):
#         '''
#         Scipy solver
#         :param upper:
#         :return:
#         '''
#         def f(x):
#             return np.maximum(sc_pos_avg_by_neg - x, 0.).sum() / upper + x
#         result = optimize.minimize_scalar(f, method="golden")
#         if result.success:
#             pauc = 1. - f(result.x)
#         else:
#             return 0.
#         pauc += 0.5 * np.mean(np.equal(delta_sc, 0.).astype(float))
#         return pauc
#
#     if alpha > 0.:
#         pauc_upper = get_pauc_from_start(upper) * upper / num_neg
#         pauc_lower = get_pauc_from_start(lower) * lower / num_neg
#         return (pauc_upper - pauc_lower) * num_neg / (upper - lower)
#     else:
#         return get_pauc_from_start(upper)

def get_auc_from_roc(fpr, tpr):
    # FPR should be increasing
    auc = 0.
    n = len(fpr)
    for i in range(0, n - 1):
        if fpr[i + 1] - fpr[i] > 0:
            auc += ((tpr[i + 1] + tpr[i]) / 2) * (fpr[i + 1] - fpr[i])
    return auc


def get_pauc_from_roc(fpr, tpr, alpha=0., beta=0.5):
    # FPR should be increasing
    assert beta > alpha, "Invalid range"
    pauc = 0.
    n = len(fpr)
    for i in range(0, n - 1):
        if fpr[i + 1] - fpr[i] > 0 and fpr[i] >= alpha and fpr[i + 1] <= beta:
            pauc += ((tpr[i + 1] + tpr[i]) / 2) * (fpr[i + 1] - fpr[i])
    # Normalize the pauc
    pauc = pauc / (beta - alpha)
    return pauc
