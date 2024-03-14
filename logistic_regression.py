import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def processdata(file):
    pd.set_option('display.max_columns', None)
    dataframe = pd.read_csv(file)

    dataframe = dataframe.drop('Name', axis=1)
    dataframe = pd.get_dummies(dataframe, columns=['Sex'], prefix='Sex').astype(int)

    target_column = dataframe['Survived']
    means = dataframe.iloc[:, :-1].mean()
    standard_deviations = dataframe.iloc[:, :-1].std()
    scaled_dataframe = (dataframe.iloc[:, :-1] - means) / standard_deviations
    features = scaled_dataframe.to_numpy()
    scaled_dataframe['Survived'] = target_column
    targets = target_column.to_numpy()

    return features, targets


def gradient_descent(features, targets, learning_rate=0.01, epochs=100):
    weights = np.zeros(features.shape[1])
    bias = 0

    accuracy = np.empty(epochs)

    for iternum in range(epochs):
        polynomial = np.dot(features, weights) + bias
        prediction = 1 / (1 + np.exp(-polynomial))

        accuracy[iternum] = accuracyratio(prediction, targets)

        weight_gradient = np.dot(features.T, (prediction - targets)) / len(features)
        bias_gradient = np.sum(prediction - targets) / len(features)

        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

    print(accuracy)
    return weights, bias, accuracy


def graphing(accuracy):
    epoch = range(len(accuracy))

    plt.plot(epoch, accuracy, label='Accuracy vs Epoch')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def accuracyratio(prediction, targets):
    prediction = (prediction > 0.5).astype(int)
    accuracy = (np.sum(prediction == targets) / len(targets)) * 100

    return accuracy


if __name__ == '__main__':
    x, y = processdata('train_logistic_regression.csv')
    w, b, a = gradient_descent(x, y)
    print(w)
    print(b)
    x, y = processdata('test_logistic_regression.csv')
    pol = np.dot(x, w) + b
    pred = 1 / (1 + np.exp(-pol))
    print(f"{accuracyratio(pred, y)}%")
    graphing(a)
