# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Implementation of historically weighted regression method based on relevance.
"""
# pylint: disable=invalid-name

import warnings
from typing import Tuple
import numpy as np

from mlfinlab.util import devadarsh


class HistoryWeightRegression:
    """
    The class that houses all related methods for the historically weighted regression tool.
    """

    def __init__(self, Y_train: np.array, X_train: np.array, check_condi_num: bool = False):
        """
        Instantiate the class with data.

        :param Y_train: (np.array) The 1D (n, ) dependent data vector.
        :param X_train:  (np.array) The 2D (n-by-k) independent data vector, n: num of instances, k: num of variables
            or features.
        :param check_condi_num: (bool) Optional. Whether to check the condition number of the covariance matrix and
            fisher info matrix from the training data X (Their values are the same). If this number is too large then it
            may lead to numerical issues. Defaults to False. Toggle this off to save some computing time.
        """

        self.X = X_train.copy()
        self.Y = Y_train.copy()
        self.X_avg = np.average(self.X, axis=0)  # Columnwise average for the training data, a vector.
        self.Y_avg = np.average(self.Y)  # Average for the dependent data, a float.

        # Covariance matrix from X and inverse of the covariance matrix, (effectively the Fisher info matrix)
        self.cov_mtx, self.fisher_info_mtx = self._calc_cov_and_fisher(X=self.X, check_condi_num=check_condi_num)

        devadarsh.track('HistoryWeightRegression')

    def get_fit_result(self) -> dict:
        """
        Fit result and statistics using the training data.

        :return: (dict) The fit result and associated statistics.
        """

        results = {'Covariance matrix': self.cov_mtx,
                   'Inv of cov matrix': self.fisher_info_mtx,
                   'Condition number of cov': np.linalg.cond(self.cov_mtx),
                   'Cov mtx shape': self.cov_mtx.shape}

        return results

    def predict(self, X_t: np.array, relev_ratio_threshold: float = 1) -> np.array:
        """
        Predict the result using fitted model from a subsample chosen by the ratio of relevance.

        For example, if relev_ratio_threshold = 0.4, then it chooses the top 40 percentile data ranked by relevance to
        x_t. This method returns the prediction in column 0, also returns the associated prediction standard
        deviations in the column 1.

        For each row element x_t in X_t we have the following:
        y_t := y_avg + 1/(n-1) * sum{relevance(x_i, x_t) * (y_i - y_avg), subsample}
        where y_i, x_i are from subsamples. The matrix form is:
        y_t := y_avg + 1/(n-1) * (x_t - x_avg).T @ fisher_info_mtx @ (X_sub - x_avg).T @ (y_sub - y_avg)

        :param X_t: (np.array) The 2D (n_t-by-k) test data, n_t is the number of instances, k is the number of
            variables or features.
        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by
            relevance, must be a number between [0, 1]. For example, 0.6 corresponds to the top 60 percentile data
            ranked by relevance to x_t. Defaults to 1.
        :return: (np.array) The predicted results in col 0, and standard deviations in col 1.
        """

        # Apply for each row for X_t.
        Y_predicts = np.apply_along_axis(self.predict_one_val, axis=1, arr=X_t,
                                         relev_ratio_threshold=relev_ratio_threshold)

        return Y_predicts

    def predict_one_val(self, x_t: np.array, relev_ratio_threshold: float = 1) -> Tuple[float, float]:
        """
        Predict one value using fitted model from a subsample chosen by the ratio of relevance.

        For example, if relev_ratio_threshold = 0.4, then it chooses the top 40 percentile data ranked by relevance to
        x_t. This method also returns the associated prediction standard deviations.

        y_t := y_avg_sub + 1/(n-1) * sum{relevance(x_i, x_t) * (y_i - y_avg_sub), subsample}
        where y_i, x_i are from subsamples. The equivalent matrix form is:
        y_t := y_avg_sub + 1/(n-1) * (x_t - x_avg).T @ fisher_info_mtx @ (X_sub - x_avg).T @ (y_sub - y_avg_sub)

        :param x_t: (np.array) A single row element test data, 1D (k, 1). k is the number of features.
        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by
            relevance, must be a number between [0, 1]. For example, 0.6 corresponds to the top 60 percentile data
            ranked by relevance to x_t. Defaults to 1.
        :return: (Tuple[float, float]) The predicted result and associated standard deviation.
        """

        # 1. Find the subsample above a relevance threshold for prediction
        # This is different for each x_t.
        X_sub, Y_sub, _, pred_std = self.find_subsample(x_t, relev_ratio_threshold, above=True)

        # 2. Predict
        subsample_size = len(Y_sub)
        Y_avg = np.average(Y_sub)
        y_t = Y_avg + 1 / (subsample_size - 1) * (
                (x_t - self.X_avg).reshape(1, -1) @ self.fisher_info_mtx
                @ ((X_sub - self.X_avg).T @ (Y_sub - Y_avg)).reshape(-1, 1))

        return y_t[0][0], pred_std

    def find_subsample(self, x_t: np.array, relev_ratio_threshold: float = 1, above: bool = True) \
            -> Tuple[np.array, np.array, np.array, float]:
        """
        Find the subsamples of X and Y in the training set by relevance above or below a given threshold with x_t.

        For example, if relev_ratio_threshold=0.3, above=True, then it finds the top 30 percentile.
        If relev_ratio_threshold=0.3, above=False, then it finds the bottom 70 percentile.

        The standard deviation is calculated as the sqrt of the variance of y_t hat, the prediction w.r.t. x_t:
        var_yt_hat = [(n-1)/n^2 * var_y] + [1/n * y_mean^2] + [var_y/n + y_mean^2/(n-1)]*var_r, where
        var_y is the subsample variance of Y, y_mean is the subsample average of Y, var_r is the subsample variance of
        relevance.

        :param x_t: (np.array) A single row element test data, 1D (k, 1). k is the number of features.
        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by
            relevance, must be a number between [0, 1].
        :param above: (bool) Optional. Whether to find the subsample above the threshold or below the threshold.
        :return: (Tuple[np.array, np.array, np.array, float]) The subsample for X, for Y, the corresponding
            indices to select the subsample and the std.
        """

        # 1. For all occurances, find their relevance value.
        relevance_vals = np.apply_along_axis(func1d=self.calc_relevance, axis=1, arr=self.X, x_j=x_t)

        # 2. Get the index that are above/below the threshold.
        sorted_idx = np.argsort(relevance_vals)
        cutoff_idx = int(len(sorted_idx) * (1 - relev_ratio_threshold))
        if above:
            result_idx = sorted_idx[cutoff_idx:]
        else:
            result_idx = sorted_idx[:cutoff_idx]

        # 3. Calculate subsamples and indices
        X_sub = self.X[result_idx]  # Fancy indexing
        Y_sub = self.Y[result_idx]  # Fancy indexing
        relev_sub = relevance_vals[result_idx]  # Fancy indexing

        # 4. Calculate standard deviation
        n = len(result_idx)
        Y_sub_var = np.var(Y_sub, ddof=1)
        relev_sub_var = np.var(relev_sub, ddof=1)
        Y_sub_mean = np.average(Y_sub)

        group1 = (n-1) / (n*n) * Y_sub_var
        group2 = (1 / n * Y_sub_mean * Y_sub_mean)
        group3 = relev_sub_var * (Y_sub_var / n + Y_sub_mean * Y_sub_mean / (n-1))
        var_yt_hat = group1 + group2 + group3
        std_yt_hat = np.sqrt(var_yt_hat)

        return X_sub, Y_sub, result_idx, std_yt_hat

    @staticmethod
    def _calc_cov_and_fisher(X: np.array, check_condi_num: bool = False) -> Tuple[np.array, np.array]:
        """
        Find the (non-biased) covariance matrix and its inverse (fisher info matrix).

        i.e., cov = X.T @ X / (n-1), fisher_info_mtx = (n-1) inv(X.T @ X)

        :param X: (np.array) The 2D (n-by-k) independent data vector, n: num of instances, k: num of variables
            or features.
        :param check_condi_num: (bool) Optional. Whether to check the condition number of the covariance matrix and
            fisher info matrix from the training data X (Their values are the same). If this number is too large then it
            may lead to numerical issues. Defaults to False.
        :return: (Tuple[np.array, np.array]) The covariance matrix and its inverse.
        """

        cov_mtx = np.cov(X.T)
        if check_condi_num:
            condi_num_cov = np.linalg.cond(cov_mtx)
            if condi_num_cov > 1e6:
                warnings.warn(("The condition number for covariance matrix > 10^6. This may lead to numerical" +
                               " issues, consider refactoring the original data."), RuntimeWarning)

        fisher_info_mtx = np.linalg.inv(cov_mtx)

        return cov_mtx, fisher_info_mtx

    def calc_relevance(self, x_i: np.array, x_j: np.array, fisher_info_mtx: np.array = None) -> float:
        """
        Calculate relevance of x_i and x_j: r(x_i, x_j).

        r(x_i, x_j) := sim(x_i, x_j) + info(x_i) + info(x_j)

        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.
        :param x_j: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.
        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the
            fisher info matrix stored in the class calculated using training data.
        :return: (float) The relevance value.
        """

        if fisher_info_mtx is None:
            fisher_info_mtx = self.fisher_info_mtx

        sim_ij = self.calc_sim(x_i, x_j, fisher_info_mtx)
        info_i = self.calc_info(x_i, fisher_info_mtx)
        info_j = self.calc_info(x_j, fisher_info_mtx)

        relevance_value = sim_ij + info_i + info_j

        return relevance_value

    def calc_sim(self, x_i: np.array, x_j: np.array, fisher_info_mtx: np.array = None) -> float:
        """
        Calculate the similarity of x_i and x_j: sim(x_i, x_j)

        sim(x_i, x_j) := -1/2 * (x_i - x_j).T @ fisher_info @ (x_i - x_j)

        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.
        :param x_j: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.
        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the
            fisher info matrix stored in the class calculated using training data.
        :return: (float) The value of similarity.
        """

        if fisher_info_mtx is None:
            fisher_info_mtx = self.fisher_info_mtx

        xi_m_xj_horiz = (x_i - x_j).reshape(1, -1)  # Horizontal vector
        xi_m_xj_verti = (x_i - x_j).reshape(-1, 1)  # Vertical vector

        sim_value = - 1 / 2 * (xi_m_xj_horiz @ fisher_info_mtx @ xi_m_xj_verti)

        return sim_value[0, 0]

    def calc_info(self, x_i: np.array, fisher_info_mtx: np.array = None) -> float:
        """
        Calculate the informativeness of x_i: info(x_i)

        info(x_i) := 1/2 * (x_i - x_avg).T @ fisher_info @ (x_i - x_avg)
        Here x_avg is the training data average for each column.

        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.
        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the
            fisher info matrix stored in the class calculated using training data.
        :return: (float) The informativeness value.
        """

        if fisher_info_mtx is None:
            fisher_info_mtx = self.fisher_info_mtx

        xi_m_xavg_horiz = (x_i - self.X_avg).reshape(1, -1)  # Horizontal vector
        xi_m_xavg_verti = (x_i - self.X_avg).reshape(-1, 1)  # Vertical vector

        infomativeness_value = 1 / 2 * (xi_m_xavg_horiz @ fisher_info_mtx @ xi_m_xavg_verti)

        return infomativeness_value[0, 0]
