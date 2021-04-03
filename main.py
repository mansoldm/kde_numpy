import argparse
from datetime import datetime

from kde_numpy.kde import get_KDE

import numpy as np


from mnist_data_processing import load_processed_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sigma', type=float, default=0.2)
parser.add_argument('--kernel-type', type=str, default='gaussian')
args = parser.parse_args()


if __name__ == '__main__':
    train_X, test_X = load_processed_datasets()
    print(f'KDE on {np.shape(test_X)} test data')

    start = datetime.now()
    kde = get_KDE(kernel_type=args.kernel_type, batch_size=args.batch_size, bandwidth=args.sigma ** 2)
    kde.fit(train_X)
    mlp = kde.mean_log_prob(test_X)
    print(f'Mean log prob: {mlp}')
    print(f'Custom KDE time elapsed: {datetime.now() - start}')
