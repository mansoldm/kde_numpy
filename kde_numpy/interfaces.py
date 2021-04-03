from types import FunctionType

from kde_numpy.errors import ModelNotFittedError


class Fittable:
    def fit(self, *args, **kwargs):
        raise NotImplementedError


class Sampler(Fittable):
    # a SamplingComponent is a component we must fit and from which we can draw samples
    def mean_log_prob(self, *args, **kwargs):
        raise NotImplementedError

    def sample_log_prob(self, *args, **kwargs):
        raise NotImplementedError

    def generate_sample(*args, **kwargs):
        raise NotImplementedError


def __init_decorator(func):
    def wrapper(self, *args, **kwargs):
        val = func(self, *args, **kwargs)
        self.fitted = False
        return val
    return wrapper


def __fit_decorator(func):
    def wrapper(self, *args, **kwargs):
        val = func(self, *args, **kwargs)
        self.fitted = True
        return val
    return wrapper


def __check_fit(func):
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise ModelNotFittedError(
                f'Attempted to call {func} without fitting the model')
        return func(self, *args, **kwargs)
    return wrapper


def __dec_applyer(cls, decorator_dict):
    for fname, decs in decorator_dict.items():
        func = getattr(cls, fname)
        wrapped = func
        for dec in decs:
            wrapped = dec(func)
        setattr(cls, fname, wrapped)
    return cls


def list_methods(cls):
    return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]


# fittable decorator
fit_decorator_dict = {'__init__': [__init_decorator], 'fit': [__fit_decorator]}

# sampling decorator
sampling_decorator_dict = {method_name: [
    __check_fit] for method_name in list_methods(Sampler)}
fittable_sampling_decorator_dict = {**fit_decorator_dict, **sampling_decorator_dict}


def fittable(cls,):
    return __dec_applyer(cls, decorator_dict=fit_decorator_dict)

# we define a class decorator 'sampling' to apply to subclasses of SamplingComponent
# this takes func_dec and applies a decorator to each function
def sampling(cls,):
    return __dec_applyer(cls, decorator_dict=fittable_sampling_decorator_dict)
