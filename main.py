import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression
from kmeans import KMeans
from metrics import accuracy


if __name__ == '__main__':
    # model = LogisticRegression(n_classes=10, metric=accuracy)
    model = KMeans(K=10, metric=accuracy)

    do_analysis = True
    print_test_accuracy = False
    learning_rate = 0.001
    n_epochs = 10

    train_set_share = 0.8
    data_path = 'data/'
    np.set_printoptions(precision=3, suppress=True)

    df = pd.read_csv(data_path + 'train.csv').transpose()
    X = df.drop(index='label').to_numpy() / 255
    y = df.loc['label'].to_numpy().reshape(1, -1)

    subm_X = pd.read_csv(data_path + 'test.csv').transpose().to_numpy()
    subm_X = subm_X / 255

    if do_analysis:
        n_samples = X.shape[1]
        train_set_idx = int(n_samples * train_set_share)
        dev_set_idx = train_set_idx + int((n_samples - train_set_idx) / 2)

        train_X, train_y = X[:, :train_set_idx], y[:, :train_set_idx]
        dev_X, dev_y = X[:, train_set_idx:dev_set_idx], y[:, train_set_idx:dev_set_idx]
        test_X, test_y = X[:, dev_set_idx:], y[:, dev_set_idx:]
        print(f'the train set has been split as {train_set_idx}:{dev_set_idx - train_set_idx}:{n_samples - dev_set_idx}')

        if type(model) is LogisticRegression:
            model.fit(train_X, train_y, learning_rate=learning_rate, n_epochs=n_epochs)
        elif type(model) is KMeans:
            model.fit(train_X, n_epochs, y=train_y, supervised_learning=True)

        print(f'train acc: {accuracy(model.predict(train_X), train_y):.2f}')
        print(f'dev   acc: {accuracy(model.predict(dev_X), dev_y):.2f}')
        if print_test_accuracy: print(f'test  acc: {accuracy(model.predict(test_X), test_y):.2f}')
    else:
        if type(model) is LogisticRegression:
            model.fit(X, y, learning_rate=learning_rate, n_epochs=n_epochs)
        elif type(model) is KMeans:
            model.fit(X, n_epochs, y=y, supervised_learning=True)

        subm_y = model.predict(subm_X)

        df = pd.DataFrame(subm_y.T)
        df.index.name = 'ImageId'
        df.columns = ['Label']
        df.index += 1
        df.to_csv(data_path + 'submission.csv')
        print('the submission file has been created')
