# Copyright (c) 2023, Ahmed Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import absolute_import, division, print_function

import sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import models.utils as utils

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Global options for baselearners (see class attributes below)

base_learners_dict = dict({"GBM": GradientBoostingRegressor, "RF": RandomForestRegressor})


class WCP:

    """

    """

    def __init__(self, n_folds=5, alpha=0.1, base_learner="GBM", quantile_regression=True, metalearner="DR"):

        """
            :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
            :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
            :param base_learner: the underlying regression model
                                - current options: ["GBM": gradient boosting machines, "RF": random forest]
            :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                        or a point estimate of the CATE function. 

        """

        # set base learner
        self.base_learner        = base_learner
        self.quantile_regression = quantile_regression
        n_estimators_nuisance    = 100
        n_estimators_target      = 100
        alpha_ = alpha 
        
        # set meta learner type
        self.metalearner  = metalearner

        # set conformal correction term to 0
        self.offset_1 = self.offset_2 = 0

        # set cross-fitting parameters and plug-in models for \mu_0 and \mu_1
        self.n_folds      = n_folds
        
        base_args_u    = dict({"loss": "quantile", "alpha":1 - (alpha_/2), "n_estimators": n_estimators_target}) 
        base_args_l    = dict({"loss": "quantile", "alpha":alpha_/2, "n_estimators": n_estimators_target}) 

        self.models_u_0  = [base_learners_dict[self.base_learner](**base_args_u) for _ in range(self.n_folds)]
        self.models_l_0  = [base_learners_dict[self.base_learner](**base_args_l) for _ in range(self.n_folds)] 
        self.models_u_1  = [base_learners_dict[self.base_learner](**base_args_u) for _ in range(self.n_folds)]
        self.models_l_1  = [base_learners_dict[self.base_learner](**base_args_l) for _ in range(self.n_folds)] 

        self.pscores_models = [LogisticRegression() for _ in range(self.n_folds)]

        self.tilde_C_ITE_model = [base_learners_dict[self.base_learner]] * 2
        # set the meta-learner and cross-fitting parameters
        self.skf          = StratifiedKFold(n_splits=self.n_folds)  


    def fit(self, X, W, Y, pscores):

        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores

        :param W: treatment assignment indicator
        :param pscores: true propensity scores
        :param Y: observed factual outcomes
        :param X: covariates
        :param oracle_pscore: whether use the true propensity scores or the estimated ones

        """

        # loop over the cross-fitting folds

        for i, (train_index, test_index) in enumerate(self.skf.split(W, W)):
        
            X_1, W_1, Y_1, pscores_1 = X[train_index, :], W[train_index], Y[train_index], pscores[train_index]
            X_2, W_2, Y_2, pscores_2 = X[test_index, :], W[test_index], Y[test_index], pscores[test_index]

            self.pscores_models[i].fit(X_1, W_1)

            self.models_u_0[i].fit(X_1[W_1==0, :], Y_1[W_1==0])
            self.models_l_0[i].fit(X_1[W_1==0, :], Y_1[W_1==0])

            self.models_u_1[i].fit(X_1[W_1==1, :], Y_1[W_1==1])
            self.models_l_1[i].fit(X_1[W_1==1, :], Y_1[W_1==1])

    def predict_counterfactuals(self, X):
        y1_u = utils.cross_fold_computation(self.models_u_1, X)
        y1_l = utils.cross_fold_computation(self.models_l_1, X)
        y0_u = utils.cross_fold_computation(self.models_u_0, X)
        y0_l = utils.cross_fold_computation(self.models_l_0, X)

        y1_u += self.offset_1
        y1_l -= self.offset_1
        y0_u += self.offset_0
        y0_l -= self.offset_0
        return [y1_l, y1_u], [y0_l, y0_u]
    
    def predict_ITE(self, X, method='naive'):
        """
        Interval-valued prediction of ITEs

        :param X: covariates of the test point

        outputs >> point estimate, lower bound and upper bound

        """
        if method == 'naive':
            y1_l = utils.cross_fold_computation(self.models_l_1, X)
            y1_u = utils.cross_fold_computation(self.models_u_1, X)
            y0_l = utils.cross_fold_computation(self.models_l_0, X)
            y0_u = utils.cross_fold_computation(self.models_u_0, X)
            
            y1_u += self.offset_1
            y1_l -= self.offset_1
            y0_u += self.offset_0
            y0_l -= self.offset_0

            return y1_l - y0_u, y1_u - y0_l
        elif method == 'nested_inexact':
            CI_ITE_l = self.tilde_C_ITE_model[0].predict(X)
            CI_ITE_u = self.tilde_C_ITE_model[1].predict(X)
            
            return CI_ITE_l, CI_ITE_u


    def conformalize(self, alpha, X_calib_0, X_calib_1, W_calib, Y_calib_0, Y_calib_1, method='naive'):

        """
        Calibrate the predictions of the meta-learner using standard conformal prediction

        """
        self.offset_1 = utils.weighted_conformal(alpha, self.models_u_1, self.models_l_1, self.pscores_models,
                                                     X_calib_1, Y_calib_1)
        self.offset_0 = utils.weighted_conformal(alpha, self.models_u_0, self.models_l_0, self.pscores_models,
                                                     X_calib_0, Y_calib_0)
        if method == 'naive':   
            # Nothing needs to be done.
            pass 
        elif method == 'nested_inexact':      
            X_calib_fold_one_0, X_calib_fold_two_0 = train_test_split(X_calib_0, test_size=0.5, random_state=42)
            Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(Y_calib_0, test_size=0.5, random_state=42)
            X_calib_fold_one_1, X_calib_fold_two_1 = train_test_split(X_calib_1, test_size=0.5, random_state=42)
            Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(Y_calib_1, test_size=0.5, random_state=42)

            offset_1 = utils.weighted_conformal(alpha, self.models_u_1, self.models_l_1, self.pscores_models,
                                                X_calib_fold_one_1, Y_calib_fold_one_1)
            offset_0 = utils.weighted_conformal(alpha, self.models_u_0, self.models_l_0, self.pscores_models,
                                                X_calib_fold_one_0, Y_calib_fold_one_0)
            
            # This is first line in Table 3 of Lei and Candes
            C0_u = Y_calib_fold_two_1 - (utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1) - offset_0) 
            C0_l = Y_calib_fold_two_1 - (utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1) + offset_0)
            # This is second line in Table 3 of Lei and Candes
            C1_u = utils.cross_fold_computation(self.models_u_1, X_calib_fold_two_0) + offset_1 - Y_calib_fold_two_0
            C1_l = utils.cross_fold_computation(self.models_l_1, X_calib_fold_two_0) - offset_1 - Y_calib_fold_two_0

            self.tilde_C_ITE_model[0].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1)),
                                          np.concatenate((C1_l, C0_l)))
            self.tilde_C_ITE_model[1].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1)), 
                                          np.concatenate((C1_u, C0_u)))

        elif method == 'nested_exact':
            pass
        else:
            raise ValueError('method must be one of naive, nested_inexact, nested_exact')
