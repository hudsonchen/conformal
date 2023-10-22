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

        # set cross-fitting parameters and plug-in models for \mu_0 and \mu_1
        self.n_folds      = n_folds
        
        base_args_u    = dict({"loss": "quantile", "alpha":1 - (alpha_/2), "n_estimators": n_estimators_target}) 
        base_args_l    = dict({"loss": "quantile", "alpha":alpha_/2, "n_estimators": n_estimators_target}) 

        self.models_u_0  = [base_learners_dict[self.base_learner](**base_args_u) for _ in range(self.n_folds)]
        self.models_l_0  = [base_learners_dict[self.base_learner](**base_args_l) for _ in range(self.n_folds)] 
        self.models_u_1  = [base_learners_dict[self.base_learner](**base_args_u) for _ in range(self.n_folds)]
        self.models_l_1  = [base_learners_dict[self.base_learner](**base_args_l) for _ in range(self.n_folds)] 

        self.pscores_models = [LogisticRegression() for _ in range(self.n_folds)]

        self.tilde_C_ITE_model = [base_learners_dict[self.base_learner]() for _ in range(2)] 
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
        
            X_train, W_train, Y_train, pscores_train = X[train_index, :], W[train_index], Y[train_index], pscores[train_index]
            X_test, W_test, Y_test, pscores_test = X[test_index, :], W[test_index], Y[test_index], pscores[test_index]

            self.pscores_models[i].fit(X_train, W_train)
            self.models_u_0[i].fit(X_train[W_train==0, :], Y_train[W_train==0])
            self.models_l_0[i].fit(X_train[W_train==0, :], Y_train[W_train==0])
            self.models_u_1[i].fit(X_train[W_train==1, :], Y_train[W_train==1])
            self.models_l_1[i].fit(X_train[W_train==1, :], Y_train[W_train==1])

    def predict_counterfactuals(self, alpha, X_test, X_calib_0, Y_calib_0, X_calib_1, Y_calib_1):
        Y1_calib_hat_u = utils.cross_fold_computation(self.models_u_1, X_calib_1, logistic_reg=False)
        Y1_calib_hat_l = utils.cross_fold_computation(self.models_l_1, X_calib_1, logistic_reg=False)
        weights_calib_1, weights_test_1, scores = utils.weights_and_scores(self.weight_1, X_test, X_calib_1, Y_calib_1, 
                                            Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models)
        offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores)
        
        Y0_calib_hat_u = utils.cross_fold_computation(self.models_u_0, X_calib_0, logistic_reg=False)
        Y0_calib_hat_l = utils.cross_fold_computation(self.models_l_0, X_calib_0, logistic_reg=False)
        weights_calib_0, weights_test_0, scores = utils.weights_and_scores(self.weight_0, X_test, X_calib_0, Y_calib_0, 
                                            Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models)
        offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores)
    
        y1_l = utils.cross_fold_computation(self.models_l_1, X_test, logistic_reg=False)
        y1_u = utils.cross_fold_computation(self.models_u_1, X_test, logistic_reg=False)
        y0_l = utils.cross_fold_computation(self.models_l_0, X_test, logistic_reg=False)
        y0_u = utils.cross_fold_computation(self.models_u_0, X_test, logistic_reg=False)
        
        y1_u += offset_1
        y1_l -= offset_1
        y0_u += offset_0
        y0_l -= offset_0
        pause = True
        return [y1_l, y1_u], [y0_l, y0_u]
    
    def predict_ITE(self, alpha, X_test, X_calib_0, Y_calib_0, X_calib_1, Y_calib_1, method='naive'):
        """
        Interval-valued prediction of ITEs

        :param X: covariates of the test point

        outputs >> point estimate, lower bound and upper bound

        """
        if method == 'naive':
            Y1_calib_hat_u = utils.cross_fold_computation(self.models_u_1, X_calib_1, logistic_reg=False)
            Y1_calib_hat_l = utils.cross_fold_computation(self.models_l_1, X_calib_1, logistic_reg=False)
            weights_calib_1, weights_test_1, scores = utils.weights_and_scores(self.weight_1, X_test, X_calib_1, Y_calib_1, 
                                                Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models)
            offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores)
            
            Y0_calib_hat_u = utils.cross_fold_computation(self.models_u_0, X_calib_0, logistic_reg=False)
            Y0_calib_hat_l = utils.cross_fold_computation(self.models_l_0, X_calib_0, logistic_reg=False)
            weights_calib_0, weights_test_0, scores = utils.weights_and_scores(self.weight_0, X_test, X_calib_0, Y_calib_0, 
                                                Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models)
            offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores)
        
            y1_l = utils.cross_fold_computation(self.models_l_1, X_test, logistic_reg=False)
            y1_u = utils.cross_fold_computation(self.models_u_1, X_test, logistic_reg=False)
            y0_l = utils.cross_fold_computation(self.models_l_0, X_test, logistic_reg=False)
            y0_u = utils.cross_fold_computation(self.models_u_0, X_test, logistic_reg=False)
            
            y1_u += offset_1
            y1_l -= offset_1
            y0_u += offset_0
            y0_l -= offset_0
            return y1_l - y0_u, y1_u - y0_l
        
        elif method == 'nested_inexact':
            CI_ITE_l = self.tilde_C_ITE_model[0].predict(X_test)
            CI_ITE_u = self.tilde_C_ITE_model[1].predict(X_test)
            
            return CI_ITE_l, CI_ITE_u
        else:
            raise ValueError('method must be one of naive, nested_inexact, nested_exact')


    def conformalize(self, alpha, X_calib_0, X_calib_1, W_calib, Y_calib_0, Y_calib_1, method='naive'):
        """
        Calibrate the predictions of the meta-learner using standard conformal prediction

        """
        def weight_1(pscores_models, x):
            pscores_calibs = utils.cross_fold_computation(pscores_models, x, logistic_reg=True)
            return 1. / pscores_calibs
        
        def weight_0(pscores_models, x):
            pscores_calibs = utils.cross_fold_computation(pscores_models, x, logistic_reg=True)
            return 1. / (1.0 - pscores_calibs)
        
        
        if method == 'naive':   
            self.weight_0 = weight_0
            self.weight_1 = weight_1
            # Nothing needs to be done.
            pass 
        elif method == 'nested_inexact':  

            X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, test_size=0.5, random_state=42)
            X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, test_size=0.5, random_state=42)

            Y1_calib_hat_u = utils.cross_fold_computation(self.models_u_1, X_calib_fold_one_1, logistic_reg=False)
            Y1_calib_hat_l = utils.cross_fold_computation(self.models_l_1, X_calib_fold_one_1, logistic_reg=False)
            
            def weight_fn(pscores_models, x):
                pscores = utils.cross_fold_computation(pscores_models, x, logistic_reg=True)
                return (1.0 - pscores) / pscores
            
            weights_calib_1, weights_test_1, scores = utils.weights_and_scores(weight_fn, X_calib_fold_two_0, X_calib_fold_one_1, Y_calib_fold_one_1, 
                                                                               Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models)
            
            offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores)
            
            Y0_calib_hat_u = utils.cross_fold_computation(self.models_u_0, X_calib_fold_one_0, logistic_reg=False)
            Y0_calib_hat_l = utils.cross_fold_computation(self.models_l_0, X_calib_fold_one_0, logistic_reg=False)
            
            def weight_fn(pscores_models, x):
                pscores = utils.cross_fold_computation(pscores_models, x, logistic_reg=True)
                return pscores / (1.0 - pscores)
            
            weights_calib_0, weights_test_0, scores = utils.weights_and_scores(weight_fn, X_calib_fold_two_1, X_calib_fold_one_0, Y_calib_fold_one_0, 
                                                                               Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models)
            offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores)
            
            # This is second line in Table 3 of Lei and Candes
            # Note that C1 is for the control group
            C1_u = (utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_0, logistic_reg=False) + offset_1) - Y_calib_fold_two_0
            C1_l = (utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_0, logistic_reg=False) - offset_1) - Y_calib_fold_two_0
            
            # This is first line in Table 3 of Lei and Candes
            # Note that C0 is for the control group
            C0_u = Y_calib_fold_two_1 - (utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1, logistic_reg=False) - offset_0)
            C0_l = Y_calib_fold_two_1 - (utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1, logistic_reg=False) + offset_0)

            self.tilde_C_ITE_model[0].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1)),
                                          np.concatenate((C1_l, C0_l)))
            self.tilde_C_ITE_model[1].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1)), 
                                          np.concatenate((C1_u, C0_u)))
            pause = True
        elif method == 'nested_exact':
            pass
        else:
            raise ValueError('method must be one of naive, nested_inexact, nested_exact')
