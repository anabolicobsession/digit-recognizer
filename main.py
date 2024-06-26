import numpy as np
import pandas as pd

from functions.activation_functions import softmax, relu, d_relu, d_softmax
from functions.loss_functions import cross_entropy, d_cross_entropy
from functions.metrics import accuracy
from models.kmeans import KMeans
from models.knn import KNN
from models.logistic_regression import LogisticRegression
from models.neural_network import NeuralNetwork
from models.svm import SVM

if __name__ == '__main__':
    logistic_regression = LogisticRegression(learning_rate=0.001, n_epochs=10, verbose=True, metric=accuracy)
    kmeans = KMeans(K=10, unsupervised_learning=False, verbose=True, metric=accuracy)
    knn = KNN(K=3, verbose=True)
    svm = SVM(C=1, learning_rate=0.001, n_epochs=15, verbose=True)
    neural_network = NeuralNetwork(
        hidden_layers=[100],
        activations=[relu, softmax],
        derivatives=[d_relu, d_softmax],
        loss=cross_entropy,
        d_loss=d_cross_entropy,
        learning_rate=0.001,
        n_epochs=5,
        verbose=True,
        metric=accuracy
    )

    model = neural_network
    do_analysis = True
    print_test_accuracy = False
    train_set_share = 0.8
    data_path = 'data/'
    np.set_printoptions(precision=3, suppress=True)

    df = pd.read_csv(data_path + 'train.csv').transpose()
    X = df.drop(index='label').to_numpy() / 255
    y = df.loc['label'].to_numpy()

    subm_X = pd.read_csv(data_path + 'test.csv').transpose().to_numpy()
    subm_X = subm_X / 255

    if do_analysis:
        n_samples = X.shape[1]
        train_set_idx = int(n_samples * train_set_share)
        dev_set_idx = train_set_idx + int((n_samples - train_set_idx) / 2)

        train_X, train_y = X[:, :train_set_idx], y[:train_set_idx]
        dev_X, dev_y = X[:, train_set_idx:dev_set_idx], y[train_set_idx:dev_set_idx]
        test_X, test_y = X[:, dev_set_idx:], y[dev_set_idx:]
        print(f'the training set has been split as {train_set_idx}:{dev_set_idx - train_set_idx}:{n_samples - dev_set_idx}')

        model.fit(train_X, train_y)
        print(f'train acc: {accuracy(train_y, model.predict(train_X)):.2f}')
        print(f'dev   acc: {accuracy(dev_y, model.predict(dev_X)):.2f}')
        if print_test_accuracy: print(f'test  acc: {accuracy(test_y, model.predict(test_X)):.2f}')
    else:
        model.fit(X, y)
        subm_y = model.predict(subm_X)

        df = pd.DataFrame(subm_y.T)
        df.index.name = 'ImageId'
        df.columns = ['Label']
        df.index += 1
        df.to_csv(data_path + 'submission.csv')
        print('the submission file has been created')
