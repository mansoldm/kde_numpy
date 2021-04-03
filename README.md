# KDE in NumPy

A simple example of Kernel Density Estimation in NumPy.

## Setup and Testing
For setup and testing, make sure to install `pytest` and `setuptools`.
Install via pip: 
```bash
pip install kde_numpy/
```
Run tests:
```bash
python -m pytest
```
## Sample usage
Assuming the availability of training and test sets, we can use the KDE as follows: 
```python
# instantiate KDE
kde = get_KDE(batch_size=128, kernel_type='gaussian', eps=1e-8, bandwidth=0.2)

# set parameters
kde.fit(train_X)

# score test set
mlp = kde.mean_log_prob(test_X)
```