import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    """
    Verify that the dataset contains the expected columns in the correct order.

    Args:
        data: Input dataset to validate.
    """

    expected_colums = [
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y',
    ]

    these_columns = data.columns.to_numpy()  

    # This also enforces the same order using numpy comparison for better performance
    assert np.array_equal(expected_colums, these_columns)  


def test_row_count(data):
    """
    Verify that the number of rows in the dataset is within the expected range.

    Args:
        data: Input dataset to validate.
    """
    n_rows = data.shape[0]

    assert 30000 < n_rows < 50000


def test_similar_target_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset

    Args:
        data: Input dataset containing the new target distribution.
        ref_data: Reference dataset used for distribution comparison.
        kl_threshold: Maximum allowed KL divergence between the distributions.
    """
    dist1 = data['y'].value_counts().sort_index()
    dist2 = ref_data['y'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold