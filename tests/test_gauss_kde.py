import pytest
import numpy as np
from itertools import product

from kde_numpy.errors import DataShapeError, ModelNotFittedError
from kde_numpy.kde import get_KDE
from kde_numpy.kde.base import KDE


@pytest.mark.parametrize('batch_size, kernel, eps, kernel_args, expected',
                         [   # invalid kernel choice
                             (128, 0, 1e-8, {'bandwidth': 0.1}, ValueError),
                             (128, '', 1e-8, {'bandwidth': 0.1}, ValueError),
                             (128, 'invalid_kernel', 1e-8, {'bandwidth': 0.1}, ValueError),
                             (128, 1, 1e-8, {'bandwidth': 0.1}, ValueError),
                             (128, 0.1, 1e-8, {'bandwidth': 0.1}, ValueError),
                             
                             # invalid batch size
                             ('128', 'gaussian', 1e-8, {'bandwidth': 0.1}, TypeError),
                             (0.1, 'gaussian', 1e-8, {'bandwidth': 0.1}, TypeError),
                             (0, 'gaussian', 1e-8, {'bandwidth': 0.1}, ValueError),
                             (-1, 'gaussian', 1e-8, {'bandwidth': 0.1}, ValueError),
                             
                             # invalid eps
                             (128, 'gaussian', '1e-8', {'bandwidth': 0.1}, TypeError),

                             # invalid kernel params
                             (128, 'gaussian', 1e-8, {'bandwidth': -1}, ValueError),
                             (128, 'gaussian', 1e-8, {'bandwidth': 0}, ValueError),
                             (128, 'gaussian', 1e-8, {'bandwidth': '0'}, TypeError),
                             (128, 'gaussian', 1e-8, {'fake_param': 0}, TypeError),
                             (128, 'gaussian', 1e-8, {'bandwidth': 0.1, 'fake_param': 0}, TypeError),
                             (128, 'gaussian', 1e-8, {}, TypeError),
                         ]
                         )
def test_kde_init_error(batch_size: int, kernel: str, eps: float, kernel_args: dict, expected):
    with pytest.raises(expected):
        k = get_KDE(batch_size=batch_size, kernel_type=kernel, eps=eps, **kernel_args)


@pytest.mark.parametrize('batch_size, kernel, eps, bandwidth, expected',
                         [
                             (128, 'gaussian', 1e-7, 0.1, KDE),
                         ]
                         )
def test_kde_init_ok(batch_size: int, kernel: str, eps: float, bandwidth: float, expected):
    k = get_KDE(batch_size=batch_size, kernel_type=kernel, eps=eps, bandwidth=bandwidth)
    assert isinstance(k, expected)


@pytest.mark.parametrize('train_X, test_X, expected',
                        # test a few predictable values
                         [
                             (
                                 np.array([[0] * d] * k),
                                 np.array([[0] * d]),
                                 np.log(np.pi ** (-d/2),),
                             ) for (d, k) in product(range(1, 10), range(1, 10))
                         ]
                         )
def test_kde_fit_ok(train_X: np.array, test_X: np.array, expected, setup_fixture,):
    # fit model with train_X and get the output prob of testpoint test_X
    k: kde.KDE = setup_fixture['kde_half_bw']
    k.fit(train_X,)
    mlp = float(k.mean_log_prob(test_X))
    # account for FP error
    assert mlp == pytest.approx(expected)


@pytest.mark.parametrize('train_X, test_X',
                         [
                             (np.array([[0, 0, 0, 0]]),
                              np.array([[0, 0, 0, 0]]))
                         ]
                         )
def test_kde_batch_larger_than_test_ok(train_X: np.array, test_X: np.array, setup_fixture,):
    k: kde.KDE = setup_fixture['kde_half_bw']
    k.fit(train_X)
    mlp = k.mean_log_prob(test_X)
    slp = np.mean(k.sample_log_prob(test_X[:k.batch_size]))
    assert mlp == slp, f'mlp: {mlp}, slp: {slp}'


@pytest.mark.parametrize('test_X, expected', [
    (np.array([[0, 0, 0, 0]]), ModelNotFittedError),
    (np.array([0, 0, 0, 0]), ModelNotFittedError)
]
)
def test_kde_not_fit(test_X: np.array, expected, setup_fixture):
    with pytest.raises(expected):
        k: kde.KDE = setup_fixture['kde_unfitted']
        mlp = float(k.sample_log_prob(test_X))


@pytest.mark.parametrize('train_X, test_X, expected', [
    # test type errors: training/testing data should be a np.ndarray
    ([0, 0, 0, 0], np.array([[0, 0, 0, 0]]), TypeError),
    (np.array([[0, 0, 0, 0]]), [0, 0, 0, 0], TypeError),
    ('[0, 0, 0, 0]', np.array([[0, 0, 0, 0]]), TypeError),
    (np.array([[0, 0, 0, 0]]), '[0, 0, 0, 0]', TypeError),

    # test data shape: training/testing data should be of rank 2
    (np.array([0, 0, 0, 0]), np.array([[0, 0, 0, 0]]), DataShapeError),
    (np.array([[[0, 0, 0, 0]]]), np.array([[0, 0, 0, 0]]), DataShapeError),
    (np.array([[0, 0, 0, 0]]), np.array([0, 0, 0, 0]), DataShapeError),
    (np.array([[0, 0, 0, 0]]), np.array([[[0, 0, 0, 0]]]), DataShapeError),
    (np.array([[]]), np.array([[[0]]]), DataShapeError),
    
    # dimensionality of datapoints must match
    (np.array([[0, 0, 0, 0, 0]]), np.array([[[0, 0, 0, 0]]]), DataShapeError),
    (np.array([[0, 0, 0, 0]]), np.array([[[0, 0, 0, 0, 0]]]), DataShapeError),
]
)
def test_kde_test_train_shape_error(train_X: np.ndarray, test_X: np.array, expected, setup_fixture):
    with pytest.raises(expected):
        k: KDE = setup_fixture['kde_half_bw']
        k.fit(train_X)
        mlp = float(k.sample_log_prob(test_X))



# def test_generate_sample_