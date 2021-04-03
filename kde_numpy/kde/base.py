import numpy as np
from tqdm import tqdm
from functools import reduce

from kde_numpy.interfaces import Sampler, sampling
from kde_numpy.function_decorators import check_init_args_KDE, check_np_array, check_kernel_test_data, check_output_type


@sampling
class KDE(Sampler):
    @check_init_args_KDE
    def __init__(self, batch_size: int, eps=1e-8):
        """
        Args:
            batch_size (int): batch size when computing mean log probability
            eps (float, optional): Eps value to prevent division by zero. Defaults to 1e-8.
        """
        self.batch_size = batch_size
        self.eps = eps

    def fit(self, *args, **kwargs):
        pass

    @check_np_array
    @check_output_type(float)
    def mean_log_prob(self, datapoints: np.ndarray, *args, **kwargs) -> float:
        """Compute mean log probability of test set datapoints

        Args:
            datapoints (np.ndarray): test data

        Returns:
            float: mean log probability of data
        """
        # reducer for batch version of running mean
        def reducer(curr_state: tuple, next_batch: np.array):
            curr_mean, N_curr_samples = curr_state
            N_new = np.shape(next_batch)[0]
            N = N_curr_samples + N_new
            addend = (np.sum(next_batch) - N_new * curr_mean) / N
            return (curr_mean + addend, N)

        # the following generator gets the log probabilities of the current batch
        gen = (self.sample_log_prob(datapoints[i: i + self.batch_size, :], *args, **kwargs)
               for i in tqdm(range(0, len(datapoints), self.batch_size)))

        mean = reduce(reducer, gen, (0, 0))[0]
        return float(mean)

    @check_np_array
    @check_output_type(np.ndarray)
    def sample_log_prob(self, test_X: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Compute log p(x) = ∑_i log p(z_i) + log p(x|z_i) for a batch of test points

        Args:
            test_X (np.ndarray): Test points

        Returns:
            np.ndarray: Log probabilities, shape (batch_size,)
        """
        def reducer(val1: np.ndarray, val2: np.ndarray):
            return val1 + val2

        gen = (self._joint_prob_i(test_X, i, *args) for i in range(self.k))
        sum_across_means = reduce(reducer, gen)

        # add eps to prevent division by zero
        return np.log(sum_across_means + self.eps)

    @check_output_type(np.ndarray)
    def generate_sample(self, *args, **kwargs) -> np.ndarray:
        """Generate sample of x. First, z_i is sampled and used to sample x from the kernel.
        The z_i sample is subsequently discarded and the x sample is returned.

        Returns:
            np.ndarray: sample of x
        """
        z_i = self._sample_component(*args, **kwargs)
        x = self._sample_conditional(z_i, *args, **kwargs)
        return x

    def _joint_prob_i(self, test_X: np.ndarray, i, *args, **kwargs) -> np.ndarray:
        """Compute log probability p(x, z_i) = log p(z_i) + log p(x|z_i)
        """
        log_p_x_z = self._kernel_log_prob(test_X, i, *args, **kwargs)
        log_p_z = self._component_log_prob(test_X, i, *args, **kwargs)
        return np.exp(log_p_z + log_p_x_z, dtype=np.float128)

    def _kernel_log_prob(self, test_X: np.ndarray, z_i: int, *args, **kwargs):
        """This method defines the kernel function. As per our formulation,
        this corresponds to log p(x|z_i) and thus depends on these arguments

        Args:
            test_X (np.ndarray): batch of items to be scored
            z_i (int): component
        """
        raise NotImplementedError

    def _component_log_prob(self, z_i: int, *args, **kwargs):
        """This returns the log probability of the component z_i

        Args:
            z_i (int): component
        """
        raise NotImplementedError

    def _sample_component():
        """Here we sample our component z_i when generating a sample of x
        This method should agree with the parameters set during fitting,

        Returns:
            int: component z_i
        """
        raise NotImplementedError

    def _sample_conditional(z_i: int):
        """Sample x conditional on z_i as per the generation process we defined. Again, 
        as this function corresponds to sampling from the distribution represented by the
        kernel, it should agree witht the kernel function and its parameters set in .fit()

        Args:
            z_i (int): component to select Gaussian

        Returns:
            np.ndarray: sample of x
        """
        raise NotImplementedError
