# Copyright (c) 2023, Ahmed Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def cross_fold_computation(models, X, logistic_reg):
    y_list  = []
    if logistic_reg:
        for i in range(len(models)):
            y = models[i].predict_proba(X)[:, 1]
            y_list.append(y)
    else:
        for i in range(len(models)):
            y = models[i].predict(X)
            y_list.append(y)

    return np.mean(np.array(y_list), axis=0).reshape((-1,))


def weighted_conformal(alpha, weights_calib, weights_test, scores):
    """Weighted conformal prediction

    Args:
        alpha (float): 1-alpha is the desired coverage
        weights_calib (np.array (N_calib,) ): weights for the calibration set
        weights_test (np.array (N_test,) ): weights for the test set
        scores (np.array (N_calib, ) ): nonconformity scores for the calibration set

    Returns:
        offset (np.array (N_test, ) ): offset values for the test set
    """
    weights_calib_sum = np.sum(weights_calib)
    weights_calib = weights_calib / weights_calib_sum
    q = (1 + weights_test / weights_calib_sum) * (1 - alpha)
    order = np.argsort(scores)
    scores = scores[order]
    weights_calib = weights_calib[order]
    cw = np.cumsum(weights_calib)
    cw_all = np.repeat(cw[:, None], len(weights_test), axis=1)
    quantile_value = np.quantile(cw_all, q)
    index_quantile = np.argmax(cw_all >= quantile_value[None,:], axis=0)
    offset = scores[index_quantile]
    return offset


def weights_and_scores(weight_fn, X_test, X_calib, Y_calib, Y_calib_hat_l, Y_calib_hat_u, pscores_models):
    weights_test = weight_fn(pscores_models, X_test)
    weights_calib = weight_fn(pscores_models, X_calib)
    scores  = np.maximum(Y_calib_hat_l - Y_calib, Y_calib - Y_calib_hat_u)
    return weights_calib, weights_test, scores