# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Implements the following classes from Chapter 12 of AFML:

- Combinatorial Purged Cross-Validation class.
- Stacked Combinatorial Purged Cross-Validation class.
"""
# pylint: disable=too-many-locals, arguments-differ, invalid-name, unused-argument

from itertools import combinations
from typing import List

import pandas as pd
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold

from mlfinlab.cross_validation.cross_validation import ml_get_train_times
from mlfinlab.util import devadarsh


def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> int:
    """
    Number of combinatorial paths for CPCV(N,K).

    :param n_train_splits: (int) Number of train splits.
    :param n_test_splits: (int) Number of test splits.
    :return: (int) Number of backtest paths for CPCV(N,k).
    """

    return int(comb(n_train_splits, n_train_splits - n_test_splits) * n_test_splits / n_train_splits)


class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super().__init__(n_splits, shuffle=False, random_state=None)

        devadarsh.track('CombinatorialPurgedKFold')
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_test_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(self.n_splits, self.n_test_splits)
        self.backtest_paths = []  # Array of backtest paths

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        # Possible test splits for each fold
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)

        return combinatorial_test_ranges

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        # Fill backtest paths using train/test splits from CPCV
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths:
                for path_el in path:
                    if path_el['train'] is None and split == path_el['test'] and found is False:
                        path_el['train'] = np.array(train_indices)
                        path_el['test'] = list(range(split[0], split[-1]))
                        found = True

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        self.backtest_paths = []  # Reset backtest paths

        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(splits_indices)
        # Prepare backtest paths
        for _ in range(self.num_backtest_paths):
            path = []
            for split_idx in splits_indices.values():
                path.append({'train': None, 'test': split_idx})
            self.backtest_paths.append(path)

        embargo: int = int(X.shape[0] * self.pct_embargo)
        for test_splits in combinatorial_test_ranges:

            # Embargo
            test_times = pd.Series(index=[self.samples_info_sets[ix[0]] for ix in test_splits], data=[
                max(self.samples_info_sets[ix[0]: ix[1]]) if ix[1] - 1 + embargo >= X.shape[0] else
                max(self.samples_info_sets[ix[0]: ix[1] + embargo]) for ix in test_splits])

            test_indices = []
            for [start_ix, end_ix] in test_splits:
                test_indices.extend(list(range(start_ix, end_ix)))

            # Purge
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            # Get indices
            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))

            self._fill_backtest_paths(train_indices, test_splits)

            yield np.array(train_indices), np.array(test_indices)


class StackedCombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Stacked Combinatorial Purged Cross Validation (CPCV). It implements CPCV for multiasset dataset.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets_dict: dict = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets_dict: (dict) Dictionary of samples info sets.
                                        ASSET_1: SAMPLE_INFO_SETS, ASSET_2:...

            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        super().__init__(n_splits, shuffle=False, random_state=None)

        devadarsh.track('StackedCombinatorialPurgedKFold')
        self.samples_info_sets_dict = samples_info_sets_dict
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_test_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(self.n_splits, self.n_test_splits)
        self.backtest_paths = {}  # Dict of arrays of backtest paths

    def _fill_backtest_paths(self, asset, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param asset: (str) Asset for which backtest paths are filled.
        :param train_indices: (list) List of lists with first element corresponding to train start index, second - test end.
        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        # Fill backtest paths using train/test splits from CPCV
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths[asset]:
                for path_el in path:
                    if path_el['train'] is None and split == path_el['test'] and found is False:
                        path_el['train'] = np.array(train_indices)
                        path_el['test'] = list(range(split[0], split[-1]))
                        found = True

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        # Possible test splits for each fold
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)

        return combinatorial_test_ranges

    def split(self,
              X_dict: dict,
              y_dict: dict = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X_dict: (dict) Dictionary of asset : X_{asset}.
        :param y_dict: (dict) Dictionary of asset : y_{asset}.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        first_asset = list(X_dict.keys())[0]
        for asset in X_dict:
            if X_dict[asset].shape[0] != self.samples_info_sets_dict[asset].shape[0]:
                raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges_assets = {}
        for asset in X_dict:
            test_ranges_assets[asset] = [(ix[0], ix[-1] + 1) for ix in
                                         np.array_split(np.arange(X_dict[asset].shape[0]), self.n_splits)]
            self.backtest_paths[asset] = []

        split_indices_assets = {}
        combinatorial_test_ranges_assets = {}
        for asset in X_dict:
            splits_indices = {}
            for index, [start_ix, end_ix] in enumerate(test_ranges_assets[asset]):
                splits_indices[index] = [start_ix, end_ix]
            split_indices_assets[asset] = splits_indices
            combinatorial_test_ranges_assets[asset] = self._generate_combinatorial_test_ranges(
                splits_indices)

        # Prepare backtest paths
        for asset in X_dict:
            for _ in range(self.num_backtest_paths):
                path = []
                for split_idx in split_indices_assets[asset].values():
                    path.append({'train': None, 'test': split_idx})
                self.backtest_paths[asset].append(path)

        for i in range(len(combinatorial_test_ranges_assets[first_asset])):
            train_indices_dict = {}  # Dictionary of asset: [train indices]
            test_indices_dict = {}
            for asset, X_asset in X_dict.items():
                embargo: int = int(X_asset.shape[0] * self.pct_embargo)
                test_splits = combinatorial_test_ranges_assets[asset][i]
                # Embargo
                test_times = pd.Series(index=[self.samples_info_sets_dict[asset][ix[0]] for ix in test_splits], data=[
                    max(self.samples_info_sets_dict[asset][ix[0]:ix[1]]) if ix[1] - 1 + embargo >= X_asset.shape[0] else
                    max(self.samples_info_sets_dict[asset][ix[0]:ix[1] + embargo])
                    for ix in test_splits])

                test_indices = []
                for [start_ix, end_ix] in test_splits:
                    test_indices.extend(list(range(start_ix, end_ix)))

                # Purge
                train_times = ml_get_train_times(self.samples_info_sets_dict[asset], test_times)

                # Get indices
                train_indices = []
                for train_ix in train_times.index:
                    train_indices.append(self.samples_info_sets_dict[asset].index.get_loc(train_ix))

                train_indices_dict[asset] = train_indices
                test_indices_dict[asset] = test_indices

                self._fill_backtest_paths(asset, train_indices, test_splits)

            yield train_indices_dict, test_indices_dict
