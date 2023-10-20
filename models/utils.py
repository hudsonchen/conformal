# Copyright (c) 2023, Ahmed Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def cross_fold_computation(models, X):
    y_list  = []

    for i in range(len(models)):
        y = models[i].predict(X)
        y_list.append(y)

    return np.mean(np.array(y_list), axis=0).reshape((-1,))


def weighted_conformal(alpha, models_u, models_l, pscores_models, X_calib, Y_calib):
    y_hat_calibs_u = cross_fold_computation(models_u, X_calib)
    y_hat_calibs_l = cross_fold_computation(models_l, X_calib)
    pscores_calibs = cross_fold_computation(pscores_models, X_calib)

    nonconformity_scores  = np.maximum(y_hat_calibs_u - Y_calib, 
                                        Y_calib - y_hat_calibs_l)
    weight = 1. / pscores_calibs
    weight_normalized = weight / np.sum(weight)
    cw = np.cumsum(weight_normalized)
    quantile_value = np.quantile(cw, 1 - alpha)
    index_quantile = np.argmax(cw >= quantile_value)
    offset = nonconformity_scores[index_quantile]
    return offset

