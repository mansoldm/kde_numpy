import os
import importlib

import numpy as np

from kde_numpy.kde.base import KDE

SUPPORTED_KDE = {}

def get_KDE(kernel_type, *args, **kwargs) -> KDE:
    if kernel_type not in SUPPORTED_KDE:
        raise ValueError(
            f'Invalid kernel function choice. \'{kernel_type}\' not supported.')

    return SUPPORTED_KDE[kernel_type](*args, **kwargs)

# decorator for populating dictionary
def register_KDE(kernel_type):
    def register(cls):
        SUPPORTED_KDE[kernel_type] = cls
        return cls
    return register

# import supported KDEs
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('kde_numpy.kde.' + module)
