# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/portfoliolab/blob/master/LICENSE.txt

# pylint: disable=missing-module-docstring

import warnings
from abc import ABC
from portfoliolab.estimators import ReturnsEstimators
from portfoliolab.utils import RiskMetrics
from portfoliolab.estimators import RiskEstimators


class BaseClusteringOptimizer(ABC):
    """
    Abstract class for clustering-based portfolio optimization algorithms.
    """

    def __init__(self):
        """
        Abstract init method.
        """

        self.constraints = None
        self.weights = list()

        self.returns_estimator = ReturnsEstimators()
        self.risk_estimator = RiskEstimators()
        self.risk_metrics = RiskMetrics()

    def apply_weight_constraints(self, constraints: dict = None, n_iter: int = 100, precision: int = 3):
        # pylint: disable=too-many-branches
        """
        Apply weights constraining based on self.constraints.

        :param constraints: (dict) Dictionary user-specified weights-constraints: asset: (min_w, max_w).
        :param n_iter: (int) Maximum number of iterations to use when optimizing weights.
        :param precision: (int) Precision error when adjusting weights given as a number of decimals.
        """

        # If weights list is empty, this method will not work
        if len(self.weights) == 0:
            raise ValueError("Weights are empty. Please run the allocate method first.")

        if constraints is not None:
            self.constraints = constraints

        if self.constraints is None:
            raise ValueError("Please provide a constraints dictionary.")

        lower_constraint = {}  # Lower bound constraint
        upper_constraint = {}  # Upper bound constraint

        for ticker, constraint in self.constraints.items():
            if constraint[0] is not None:
                lower_constraint[ticker] = constraint[0]
            if constraint[1] is not None:
                upper_constraint[ticker] = constraint[1]

        weight_to_distribute = 0

        # Upper bound weights constraining
        upper_constraint_tickers = []  # Tickers which were constrained by upper bound
        for ticker, max_w in upper_constraint.items():
            original_w = self.weights[ticker][0]
            if original_w.round(precision) > max_w:
                self.weights[ticker] = max_w
                upper_constraint_tickers.append(ticker)
                weight_to_distribute += original_w - max_w

        # Lower bound weights constraining
        lower_constraint_tickers = []  # Tickers which were constrained by upper bound
        for ticker, min_w in lower_constraint.items():
            original_w = self.weights[ticker][0]
            if original_w.round(precision) < min_w:
                self.weights[ticker] = min_w
                lower_constraint_tickers.append(ticker)
                weight_to_distribute -= min_w - original_w

        # Redistribute excess/deficit weights
        if weight_to_distribute > 0:
            tickers_to_add = [x for x in self.weights.columns if x not in upper_constraint_tickers]
            weight_to_add = self.weights[tickers_to_add] / self.weights[tickers_to_add].sum(axis=1).iloc[0]
            self.weights[tickers_to_add] += weight_to_add * weight_to_distribute
        else:
            tickers_to_subtract = [x for x in self.weights.columns if x not in lower_constraint_tickers]
            weight_to_subtract = self.weights[tickers_to_subtract] / self.weights[tickers_to_subtract].sum(axis=1).iloc[
                0]
            self.weights[
                tickers_to_subtract] += weight_to_subtract * weight_to_distribute  # Weight to distribute is already negative

        # Checking for unmatched constraints
        unmatched_constraint = upper_constraint_tickers + lower_constraint_tickers

        # Decreasing the iteration counter
        n_iter = n_iter - 1

        # If the constraints are not reached, we continue iterating
        if (len(unmatched_constraint) > 0) and n_iter > 0:
            self.apply_weight_constraints(constraints, n_iter, precision)

        # If the weights are constrained, or the max iteration number is reached
        else:
            # If constraints are still not reached
            if len(unmatched_constraint) > 0:
                warnings.warn("Weight constraints weren't reached after given number of iterations. "
                              "The best-fit weights after a set number of iterations were output. "
                              "Please check your weight constraints or change the max number of iterations/precision.")
