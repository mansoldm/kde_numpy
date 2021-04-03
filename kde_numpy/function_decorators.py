
import numpy as np

from kde_numpy.errors import DataShapeError


def check_init_args_gaussian_KDE(init_func):
    def wrapper(self, bandwidth, *args, **kwargs):
        if not isinstance(bandwidth, float) and not isinstance(bandwidth, int):
            raise TypeError(f'bandwidth must be an integer or float, got {type(bandwidth)}')
        if bandwidth <= 0:
            raise ValueError(f'bandwidth must be greater than zero, got {bandwidth}')
        return init_func(self, bandwidth, *args, **kwargs)
    return wrapper


def check_init_args_KDE(init_func):
    def wrapper(self, batch_size, *args, **kwargs):
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an integer, got {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be greater than zero, got {batch_size}")
        if 'eps' in kwargs:
            eps = kwargs.pop('eps')
        elif len(args):
            eps = args.pop()
        else:
            eps = 1e-8

        if not isinstance(eps, float) and not isinstance(eps, int):
            raise TypeError(f'eps must be an integer or float, got {type(eps)}')
        return init_func(self,  batch_size=batch_size, eps=eps, *args, **kwargs)
    return wrapper


def check_np_array(func):
    def wrapper(self, datapoints: np.ndarray, *args, **kwargs):
        # handle type
        if not isinstance(datapoints, np.ndarray):
            raise TypeError(
                f'Input data to {func} must be a np.ndarray, got {type(datapoints)}')
        # handle shape
        shape = np.shape(datapoints)
        if not len(shape) == 2:
            raise DataShapeError(
                f'Data must be of rank 2, but got rank {len(shape)} with shape {shape}')
        return func(self, datapoints, *args, **kwargs)
    return wrapper


def check_np_array_out(func):
    def wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        # handle type
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f'Function output must be a np.ndarray, got {type(out)}')
        return out
    return wrapper


def check_output_type(out_type):
    def check_output_type_c(func):
        def wrapper(self, *args, **kwargs):
            out = func(self, *args, **kwargs)
            if not isinstance(out, out_type):
                raise TypeError(
                    f'Function output must be a {output_type}, got {type(out)}')
            return out
        return wrapper
    return check_output_type_c


def check_kernel_z(func):
    def wrapper(self, z_i, *args, **kwargs):
        if not isinstance(z_i, int):
            raise TypeError(
                f'z_i must be an integer, got value of type {type(z_i)}')

        k = np.shape(self.mu)[0]
        if not 0 <= z_i < k:
            raise ValueError(
                f'z_i must be between 0 and {k}, got {z_i}. No training datapoint provided for z_i={z_i}')

        return func(self, z_i=z_i, *args, **kwargs)
    return wrapper


def check_kernel_test_data(func):
    def wrapper(self, test_X, *args, **kwargs):
        # z_i should be checked by decorator above
        # check testdata np array dimensionality
        td, d = np.shape(test_X)[-1], np.shape(self.mu)[1]
        if td != d:
            raise DataShapeError(
                f'Test datapoints must have the same dimensionality as training datapoints, but got {td} for test datapoints and {d} for training datapoints')

        return func(self, test_X=test_X, *args, **kwargs)
    return wrapper
