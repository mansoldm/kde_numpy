import pytest

from kde_numpy.kde import get_KDE

@pytest.fixture(scope='session')
def setup_fixture(request):
    kde_half_bw = get_KDE(batch_size=128, kernel_type='gaussian', eps=0, bandwidth=1/2)
    kde_unfitted = get_KDE(batch_size=128, kernel_type='gaussian', eps=0, bandwidth=1/2)
    return locals()
