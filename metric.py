import numpy as np


def wmsfe(y_true: np.array, y_pred: np.array, features: np.array) -> float:
    """
    :param y_true: matrix (num_of_series, horizon)
    :param y_pred: predicted matrix (num_of_series, horizon)
    :param features: matrix (num_of_series, len_of_series)

    :return wmsfe metrics
    """
    h = y_true.shape[0]
    K = h * y_true.shape[1]

    D = np.array([np.var(features[:, i]) for i in range(features.shape[1])])
    wmsfe = (np.sum((y_true - y_pred) ** 2 / (h * D))) / K
    return wmsfe


def score(wmse_arr: np.array) -> float:
    """
    Final Score metric

    :param wmse_arr: array of wmse values of all datasets
    """
    alpha = 12
    return np.sum([(1.8 - 1.6 / (1 + np.exp(-alpha * el))) for el in wmse_arr]) / len(wmse_arr)


def test_wmsfe_metric():

    features = np.array([[1, 1, 1],
                        [2, 2, 2],
                        [3, 4, 5],
                        [8, 9, 6],
                        [4, 5, 7],
                        [5, 2, 1],
                        [5, 5, 8]])

    y_true = np.array([[2, 3, 4],
                      [3, 2, 3],
                      [4, 1, 2],
                      [5, 0, 1]])

    y_pred = np.array([[2.1, 3.3, 4.2],
                      [3.1, 2.2, 3.1],
                      [4.2, 1.4, 2.5],
                      [5.2, 0.1, 1.2]])

    assert np.equal(wmsfe(y_true, y_pred, features), 0.0024141677188552198)