import copy

import numpy as np

from kde_numpy.kde import register_KDE
from kde_numpy.kde.base import KDE
from kde_numpy.interfaces import sampling, fittable
from kde_numpy.function_decorators import check_init_args_gaussian_KDE,  check_kernel_z, check_kernel_test_data, check_np_array


@register_KDE('gaussian')
class GaussianKDE(KDE):
    @check_init_args_gaussian_KDE
    def __init__(self, bandwidth: float, *args, **kwargs):
        self.bandwidth = bandwidth
        super(GaussianKDE, self).__init__(*args, **kwargs)

    @check_np_array
    def fit(self, X: np.ndarray, *args, **kwargs):
        """
        For our Gaussian KDE our parameter will be 
        p(z_i) = 1 / k therefore we also set it here

        Args:
            X (np.ndarray): training points
        """
        self.mu = copy.deepcopy(X)
        self.k = np.shape(X)[0]
        self.log_p_z = -np.log(self.k)
        self.log_normalization = - (1 / 2) * np.log(2 * np.pi * self.bandwidth)

        super(GaussianKDE, self).fit(*args, **kwargs)

    @check_kernel_test_data
    @check_kernel_z
    def _kernel_log_prob(self, test_X: np.ndarray, z_i: int, *args, **kwargs):
        sq_diff = (test_X - self.mu[z_i, :]) ** 2
        gauss_term = sq_diff / (2 * self.bandwidth)
        log_normalized_term = self.log_normalization - gauss_term
        return np.sum(log_normalized_term, axis=1)

    def _component_log_prob(self, *args, **kwargs):
        return self.log_p_z

    def _sample_component():
        return np.random.choice(self.k)

    def _sample_conditional(z_i: int):
        return np.random.multivariate_normal(mean=self.mu[z_i, :], cov=np.eye(self.d) * self.bandwidth)
