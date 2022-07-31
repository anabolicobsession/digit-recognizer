import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('train.csv').transpose()
    X = df.drop(index='label').to_numpy()
    y = df.loc['label'].to_numpy().reshape(1, -1)

    train_set_share = 0.8
    n_samples = X.shape[1]
    train_set_idx = int(n_samples * train_set_share)
    dev_set_idx = train_set_idx + int((n_samples - train_set_idx) / 2)

    train_X, train_y = X[:, :train_set_idx], y[:, train_set_idx]
    dev_X, dev_y = X[:, train_set_idx:dev_set_idx], y[:, train_set_idx:dev_set_idx]
    test_X, test_y = X[:, dev_set_idx:], y[:, dev_set_idx:]

    df = pd.read_csv('test.csv').transpose()
    subm_X = df.to_numpy()

    rng = np.random.default_rng(0)
    subm_y = rng.integers(0, 9, (1, subm_X.shape[1]))

    df = pd.DataFrame(subm_y.T)
    df.index.name = 'ImageId'
    df.columns = ['Label']
    df.index += 1
    df.to_csv('submission.csv')
