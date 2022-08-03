import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression


def normalize(X):
    return X / 255


def accuracy(y, pred):
    return np.sum(y == pred) / y.shape[1]


if __name__ == '__main__':
    do_analysis = False
    print_test_accuracy = False
    learning_rate = 0.01
    epochs = 1200

    train_set_share = 0.8
    data_path = 'data/'
    np.set_printoptions(precision=3, suppress=True)

    df = pd.read_csv(data_path + 'train.csv').transpose()
    X = df.drop(index='label').to_numpy()
    X = normalize(X)
    y = df.loc['label'].to_numpy().reshape(1, -1)

    model = LogisticRegression(n_classes=10)

    if do_analysis:
        n_samples = X.shape[1]
        train_set_idx = int(n_samples * train_set_share)
        dev_set_idx = train_set_idx + int((n_samples - train_set_idx) / 2)

        train_X, train_y = X[:, :train_set_idx], y[:, :train_set_idx]
        dev_X, dev_y = X[:, train_set_idx:dev_set_idx], y[:, train_set_idx:dev_set_idx]
        test_X, test_y = X[:, dev_set_idx:], y[:, dev_set_idx:]
        print(f'the train set has been split as {train_set_idx}:{dev_set_idx - train_set_idx}:{n_samples - dev_set_idx}')

        model.fit(train_X, train_y, learning_rate=learning_rate, epochs=epochs)

        print(f'train acc: {accuracy(train_y, model.predict(train_X)):.2f}')
        print(f'dev   acc: {accuracy(dev_y, model.predict(dev_X)):.2f}')
        if print_test_accuracy: print(f'test  acc: {accuracy(test_y, model.predict(test_X)):.2f}')
    else:
        model.fit(X, y, learning_rate=learning_rate, epochs=epochs)
        subm_X = pd.read_csv(data_path + 'test.csv').transpose().to_numpy()
        subm_X = normalize(subm_X)
        subm_y = model.predict(subm_X)

        df = pd.DataFrame(subm_y.T)
        df.index.name = 'ImageId'
        df.columns = ['Label']
        df.index += 1
        df.to_csv(data_path + 'submission.csv')
        print('the submission file has been created')
