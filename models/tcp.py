from __future__ import absolute_import, division, print_function

import sys, os, time
import jax.numpy as jnp
import jax
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from quantile_forest import RandomForestQuantileRegressor
from densratio import densratio
from tqdm import tqdm

import numpy as np
import models.utils as utils

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Global options for baselearners (see class attributes below)

base_learners_dict = dict({"GBM": GradientBoostingRegressor, 
                           "QRF": RandomForestQuantileRegressor})


class TCP:

    """

    """

    def __init__(self, data_obs, data_inter, n_folds=5, alpha=0.1, base_learner="GBM", quantile_regression=True):

        """
        Transductive conformal prediction

            :param n_folds: the number of folds for the DR learner cross-fitting (See [1])
            :param alpha: the target miscoverage level. alpha=.1 means that target coverage is 90%
            :param base_learner: the underlying regression model
                                - current options: ["GBM": gradient boosting machines, "RF": random forest]
            :param quantile_regression: Boolean for indicating whether the base learner is a quantile regression model
                                        or a point estimate of the CATE function. 

        """

        # set base learner
        self.base_learner = base_learner
        self.quantile_regression = quantile_regression
        self.n_folds = n_folds
        self.alpha = alpha
        n_estimators_nuisance = 100
        n_estimators_target = 100
        
        if self.base_learner == "GBM":
            first_CQR_args_u = dict({"loss": "quantile", "alpha":1 - (self.alpha / 2), "n_estimators": n_estimators_target}) 
            first_CQR_args_l = dict({"loss": "quantile", "alpha":self.alpha/2, "n_estimators": n_estimators_target}) 
        elif self.base_learner == "QRF":
            first_CQR_args_u = dict({"default_quantiles":1 - (self.alpha/2), "n_estimators": n_estimators_target})
            first_CQR_args_l = dict({"default_quantiles":self.alpha/2, "n_estimators": n_estimators_target})
        else:
            raise ValueError('base_learner must be one of GBM, QRF')
        self.models_u_0 = base_learners_dict[self.base_learner](**first_CQR_args_u) 
        self.models_l_0 = base_learners_dict[self.base_learner](**first_CQR_args_l) 
        self.models_u_1 = base_learners_dict[self.base_learner](**first_CQR_args_u) 
        self.models_l_1 = base_learners_dict[self.base_learner](**first_CQR_args_l) 

        self.density_models_0 = None
        self.density_models_1 = None

        if self.base_learner == "GBM":
            second_CQR_args_u = dict({"loss": "quantile", "alpha":0.6, "n_estimators": n_estimators_target})
            second_CQR_args_l = dict({"loss": "quantile", "alpha":0.4, "n_estimators": n_estimators_target})
        elif self.base_learner == "QRF":
            second_CQR_args_u = dict({"default_quantiles":0.6, "n_estimators": n_estimators_target}) 
            second_CQR_args_l = dict({"default_quantiles":0.4, "n_estimators": n_estimators_target})
        
        self.tilde_C_ITE_model_u = base_learners_dict[self.base_learner](**second_CQR_args_u) 
        self.tilde_C_ITE_model_l = base_learners_dict[self.base_learner](**second_CQR_args_l) 

        self.data_obs = data_obs
        self.X_train_obs = data_obs.filter(like = 'X').values
        self.T_train_obs = data_obs['T'].values
        self.Y_train_obs = data_obs['Y'].values

        self.data_inter = data_inter
        self.X_train_inter = data_inter.filter(like = 'X').values
        self.T_train_inter = data_inter['T'].values
        self.Y_train_inter = data_inter['Y'].values

        self.Y_min, self.Y_max = [np.min(self.Y_train_obs), np.max(self.Y_train_obs)]
        self.Y_interval = np.linspace(self.Y_min, self.Y_max, 100)
        return

    def fit(self, X, y, T):
        """
        Fits the plug-in models and meta-learners using the sample (X, W, Y) and true propensity scores pscores
        """
        if T == 0:
            self.models_u_0.fit(X, y)
            self.models_l_0.fit(X, y)
        else:
            self.models_u_1.fit(X, y)
            self.models_l_1.fit(X, y)

    def predict_counterfactuals(self, alpha, X_test, Y_test):
        # Predict Y(1)
        X_train_obs_1 = self.X_train_obs[self.T_train_obs==1, :]
        Y_train_obs_1 = self.Y_train_obs[self.T_train_obs==1]
        X_train_inter_1 = self.X_train_inter[self.T_train_inter==1, :]
        Y_train_inter_1 = self.Y_train_inter[self.T_train_inter==1]

        D_train_1 = np.concatenate((X_train_obs_1, Y_train_obs_1[:, None]), axis=1)
        D_inter_1 = np.concatenate((X_train_inter_1, Y_train_inter_1[:, None]), axis=1)
        self.density_models_1 = densratio(D_inter_1, D_train_1, alpha=alpha)
        weights_train_1 = self.density_models_1.compute_density_ratio(D_train_1)
        
        Y_interval_min = Y_train_inter_1.mean() - 3 * Y_train_inter_1.std()
        Y_interval_max = Y_train_inter_1.mean() + 3 * Y_train_inter_1.std()
        Y_interval = np.linspace(Y_interval_min, Y_interval_max, 10)
        y_test_min, y_test_max = np.zeros(len(X_test)), np.zeros(len(X_test))

        for x_test in X_test:
            X_train_test_mixed = np.concatenate((X_train_obs_1, x_test[None, :]), axis=0)
            y_interval = []
            for y in Y_interval:
                weight_test = self.density_models_1.compute_density_ratio(np.concatenate((x_test, np.array([y])), axis=0)[None, :])
                Y_train_test_mixed = np.concatenate((Y_train_obs_1, np.array([y])), axis=0)
                self.fit(X_train_test_mixed, Y_train_test_mixed, T=1)
                Y1_hat_u = self.models_u_1.predict(X_train_test_mixed)
                Y1_hat_l = self.models_l_1.predict(X_train_test_mixed)
                scores_1 = np.maximum(Y1_hat_l - Y_train_test_mixed, Y_train_test_mixed - Y1_hat_u)
                offset_1 = utils.weighted_transductive_conformal(alpha, weights_train_1, weight_test, scores_1)
                print(scores_1[-1], offset_1)
                if scores_1[-1] <= offset_1:
                    y_interval.append(y)
            y_test_min.append(min(y_interval))
            y_test_max.append(max(y_interval))
        return [y_test_min, y_test_max]
    
    def predict_ITE(self, alpha, X_test, method='naive'):
        """
        Interval-valued prediction of ITEs

        :param X: covariates of the test point

        outputs >> point estimate, lower bound and upper bound

        """
        if method == 'naive':
            C0, C1 = self.predict_counterfactuals(alpha, X_test)
            return C1[0] - C0[1], C1[1] - C0[0]
        
        elif method == 'nested_inexact':
            CI_ITE_l_list, CI_ITE_u_list = [], []
            for i in range(self.n_folds):
                CI_ITE_l = self.tilde_C_ITE_model_l[i].predict(X_test)
                CI_ITE_u = self.tilde_C_ITE_model_u[i].predict(X_test)
                CI_ITE_l_list.append(CI_ITE_l)
                CI_ITE_u_list.append(CI_ITE_u)
            CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
            CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
            return CI_ITE_l, CI_ITE_u
        elif method == 'nested_exact':
            CI_ITE_l_list, CI_ITE_u_list = [], []
            for i in range(self.n_folds):
                CI_ITE_l = self.tilde_C_ITE_model_l[i].predict(X_test) - self.offset_list[i]
                CI_ITE_u = self.tilde_C_ITE_model_u[i].predict(X_test) + self.offset_list[i]
                CI_ITE_l_list.append(CI_ITE_l)
                CI_ITE_u_list.append(CI_ITE_u)
            CI_ITE_l = np.median(np.array(CI_ITE_l_list), axis=0)
            CI_ITE_u = np.median(np.array(CI_ITE_u_list), axis=0)
            return CI_ITE_l, CI_ITE_u
        else:
            raise ValueError('method must be one of naive, nested_inexact, nested_exact')


    def conformalize(self, alpha, method='naive'):
        """
        Calibrate the predictions of the meta-learner using standard conformal prediction

        """
        
        if method == 'naive':   
            # Nothing needs to be done.
            pass 
        elif method == 'nested_inexact': 
            for i in range(self.n_folds): 
                X_calib_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
                X_calib_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
                Y_calib_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
                Y_calib_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

                calib_data = self.data_obs.loc[self.calib_index_list[i]]
                Y1_calib_0 = calib_data[calib_data['T'] == 0]['Y1'].values
                Y0_calib_1 = calib_data[calib_data['T'] == 1]['Y0'].values

                # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, index_0, test_size=0.5, random_state=42)
                # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, index_1, test_size=0.5, random_state=42)
                
                X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0, Y1_calib_fold_one_0, Y1_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, Y1_calib_0, test_size=0.5, random_state=42)
                X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1, Y0_calib_fold_one_1, Y0_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, Y0_calib_1, test_size=0.5, random_state=42)

                Y1_calib_hat_u = self.models_u_1[i].predict(X_calib_fold_one_1)
                Y1_calib_hat_l = self.models_l_1[i].predict(X_calib_fold_one_1)
                
                def weight_fn_1(pscores_models, x):
                    pscores = pscores_models.predict_proba(x)[:, 1]
                    return (1.0 - pscores) / pscores
            
                weights_calib_1, weights_test_1, scores_1 = utils.weights_and_scores(weight_fn_1, X_calib_fold_two_0, X_calib_fold_one_1, 
                                                                                     Y_calib_fold_one_1, 
                                                                                     Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models[i])
            
                offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores_1)
            
                Y0_calib_hat_u = self.models_u_0[i].predict(X_calib_fold_one_0)
                Y0_calib_hat_l = self.models_l_0[i].predict(X_calib_fold_one_0)
            
                def weight_fn_0(pscores_models, x):
                    pscores = pscores_models.predict_proba(x)[:, 1]
                    return pscores / (1.0 - pscores)
            
                weights_calib_0, weights_test_0, scores_0 = utils.weights_and_scores(weight_fn_0, X_calib_fold_two_1, X_calib_fold_one_0, Y_calib_fold_one_0, 
                                                                               Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models[i])
                offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores_0)

                # ==================== Debug code ====================
                # u1 = utils.cross_fold_computation(self.models_u_1, X_calib_fold_two_0, proba=False) + offset_1
                # l1 = utils.cross_fold_computation(self.models_l_1, X_calib_fold_two_0, proba=False) - offset_1
                # coverage_1 = np.mean((Y1_calib_fold_two_0 >= l1) & (Y1_calib_fold_two_0 <= u1))
                # print('Debug: Coverage of Y(1) on second fold of calibration Y|T=1', coverage_1)
                # u0 = utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1, proba=False) + offset_0
                # l0 = utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1, proba=False) - offset_0
                # coverage_0 = np.mean((Y0_calib_fold_two_1 >= l0) & (Y0_calib_fold_two_1 <= u0))
                # print('Debug: Coverage of Y(0) on second fold of calibration Y|T=0', coverage_0)
                # pause = True

                # This is second line in Table 3 of Lei and Candes
                # Note that C1 is for the control group
                C1_u = (self.models_u_1[i].predict(X_calib_fold_two_0) + offset_1) - Y_calib_fold_two_0
                C1_l = (self.models_l_1[i].predict(X_calib_fold_two_0) - offset_1) - Y_calib_fold_two_0
                
                # This is first line in Table 3 of Lei and Candes
                # Note that C0 is for the control group
                C0_u = Y_calib_fold_two_1 - (self.models_l_0[i].predict(X_calib_fold_two_1) - offset_0)
                C0_l = Y_calib_fold_two_1 - (self.models_u_0[i].predict(X_calib_fold_two_1) + offset_0)

                dummy_index = np.random.permutation(len(X_calib_fold_two_0) + len(X_calib_fold_two_1))
                self.tilde_C_ITE_model_l[i].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1))[dummy_index, :],
                                            np.concatenate((C1_l, C0_l))[dummy_index])
                                            
                self.tilde_C_ITE_model_u[i].fit(np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1))[dummy_index, :], 
                                            np.concatenate((C1_u, C0_u))[dummy_index])
                pause = True
        elif method == 'nested_exact':
            self.offset_list = []
            for i in range(self.n_folds):
                X_calib_0 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==0, :]
                X_calib_1 = self.X_calib_obs_list[i][self.T_calib_obs_list[i]==1, :]
                Y_calib_0 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==0]
                Y_calib_1 = self.Y_calib_obs_list[i][self.T_calib_obs_list[i]==1]

                calib_data = self.data_obs.loc[self.calib_index_list[i]]
                Y1_calib_0 = calib_data[calib_data['T'] == 0]['Y1'].values
                Y0_calib_1 = calib_data[calib_data['T'] == 1]['Y0'].values

                # X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, index_0, test_size=0.5, random_state=42)
                # X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, index_1, test_size=0.5, random_state=42)
                
                X_calib_fold_one_0, X_calib_fold_two_0, Y_calib_fold_one_0, Y_calib_fold_two_0, Y1_calib_fold_one_0, Y1_calib_fold_two_0 = train_test_split(X_calib_0, Y_calib_0, Y1_calib_0, test_size=0.5, random_state=42)
                X_calib_fold_one_1, X_calib_fold_two_1, Y_calib_fold_one_1, Y_calib_fold_two_1, Y0_calib_fold_one_1, Y0_calib_fold_two_1 = train_test_split(X_calib_1, Y_calib_1, Y0_calib_1, test_size=0.5, random_state=42)

                Y1_calib_hat_u = self.models_u_1[i].predict(X_calib_fold_one_1)
                Y1_calib_hat_l = self.models_l_1[i].predict(X_calib_fold_one_1)
                
                def weight_fn_1(pscores_models, x):
                    pscores = pscores_models.predict_proba(x)[:, 1]
                    return (1.0 - pscores) / pscores
            
                weights_calib_1, weights_test_1, scores_1 = utils.weights_and_scores(weight_fn_1, X_calib_fold_two_0, X_calib_fold_one_1, 
                                                                                     Y_calib_fold_one_1, 
                                                                                     Y1_calib_hat_l, Y1_calib_hat_u, self.pscores_models[i])
            
                offset_1 = utils.weighted_conformal(alpha, weights_calib_1, weights_test_1, scores_1)
            
                Y0_calib_hat_u = self.models_u_0[i].predict(X_calib_fold_one_0)
                Y0_calib_hat_l = self.models_l_0[i].predict(X_calib_fold_one_0)
            
                def weight_fn_0(pscores_models, x):
                    pscores = pscores_models.predict_proba(x)[:, 1]
                    return pscores / (1.0 - pscores)
            
                weights_calib_0, weights_test_0, scores_0 = utils.weights_and_scores(weight_fn_0, X_calib_fold_two_1, X_calib_fold_one_0, Y_calib_fold_one_0, 
                                                                               Y0_calib_hat_l, Y0_calib_hat_u, self.pscores_models[i])
                offset_0 = utils.weighted_conformal(alpha, weights_calib_0, weights_test_0, scores_0)

                # ==================== Debug code ====================
                # u1 = utils.cross_fold_computation(self.models_u_1, X_calib_fold_two_0, proba=False) + offset_1
                # l1 = utils.cross_fold_computation(self.models_l_1, X_calib_fold_two_0, proba=False) - offset_1
                # coverage_1 = np.mean((Y1_calib_fold_two_0 >= l1) & (Y1_calib_fold_two_0 <= u1))
                # print('Debug: Coverage of Y(1) on second fold of calibration Y|T=1', coverage_1)
                # u0 = utils.cross_fold_computation(self.models_u_0, X_calib_fold_two_1, proba=False) + offset_0
                # l0 = utils.cross_fold_computation(self.models_l_0, X_calib_fold_two_1, proba=False) - offset_0
                # coverage_0 = np.mean((Y0_calib_fold_two_1 >= l0) & (Y0_calib_fold_two_1 <= u0))
                # print('Debug: Coverage of Y(0) on second fold of calibration Y|T=0', coverage_0)
                # pause = True

                # This is second line in Table 3 of Lei and Candes
                # Note that C1 is for the control group
                C1_u = (self.models_u_1[i].predict(X_calib_fold_two_0) + offset_1) - Y_calib_fold_two_0
                C1_l = (self.models_l_1[i].predict(X_calib_fold_two_0) - offset_1) - Y_calib_fold_two_0
                
                # This is first line in Table 3 of Lei and Candes
                # Note that C0 is for the control group
                C0_u = Y_calib_fold_two_1 - (self.models_l_0[i].predict(X_calib_fold_two_1) - offset_0)
                C0_l = Y_calib_fold_two_1 - (self.models_u_0[i].predict(X_calib_fold_two_1) + offset_0)

                dummy_index = np.random.permutation(len(X_calib_fold_two_0) + len(X_calib_fold_two_1))

                C_l = np.concatenate((C1_l, C0_l))[dummy_index]
                C_u = np.concatenate((C1_u, C0_u))[dummy_index]
                X = np.concatenate((X_calib_fold_two_0, X_calib_fold_two_1))[dummy_index, :]
                
                X_train, X_calib, C_l_train, C_l_calib, C_u_train, C_u_calib = train_test_split(X, C_l, C_u, test_size=0.25, random_state=42)

                self.tilde_C_ITE_model_l[i].fit(X_train, C_l_train)                                                
                self.tilde_C_ITE_model_u[i].fit(X_train, C_u_train)

                scores = np.maximum(C_u_calib - self.tilde_C_ITE_model_u[i].predict(X_calib), 
                                    self.tilde_C_ITE_model_l[i].predict(X_calib) - C_l_calib)
                offset = utils.standard_conformal(alpha, scores)
                self.offset_list.append(offset)
            
        else:
            raise ValueError('method must be one of naive, nested_inexact, nested_exact')
