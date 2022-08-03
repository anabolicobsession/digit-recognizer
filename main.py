import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression


def preprocess(X):
    return X / 255


if __name__ == '__main__':
    do_analysis = True
    print_test_accuracy = False
    learning_rate = 0.01
    epochs = 30

    train_set_share = 0.8
    data_path = 'data/'
    np.set_printoptions(precision=3, suppress=True)

    df = pd.read_csv(data_path + 'train.csv').transpose()
    X = df.drop(index='label').to_numpy()
    X = preprocess(X)
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

        model.fit(train_X, train_y, learning_rate=learning_rate, n_epochs=epochs)

        print(f'train acc: {model.accuracy(train_X, train_y):.2f}')
        print(f'dev   acc: {model.accuracy(dev_X, dev_y):.2f}')
        if print_test_accuracy: print(f'test  acc: {model.accuracy(test_X, test_y):.2f}')
    else:
        model.fit(X, y, learning_rate=learning_rate, n_epochs=epochs)
        subm_X = pd.read_csv(data_path + 'test.csv').transpose().to_numpy()
        subm_X = preprocess(subm_X)
        subm_y = model.predict(subm_X)

        df = pd.DataFrame(subm_y.T)
        df.index.name = 'ImageId'
        df.columns = ['Label']
        df.index += 1
        df.to_csv(data_path + 'submission.csv')
        print('the submission file has been created')
