import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def datasplit(file):
    with open(file, 'r') as dataset:
        lines = dataset.readlines()
    header = lines[0]
    lines = lines[1:]

    random.shuffle(lines)
    split_index = int(0.7 * len(lines))
    training_set = lines[:split_index]
    testing_set = lines[split_index:]

    trainfile = 'train_house_prices.csv'
    testfile = 'test_house_prices.csv'
    with open(trainfile, 'w') as file1:
        file1.write(header)
        file1.writelines(training_set)
    with open(testfile, 'w') as file2:
        file2.write(header)
        file2.writelines(testing_set)

    return trainfile, testfile

def processData(file):
    dataframe = pd.read_csv(file)
    dataframe.fillna(dataframe.mean(), inplace=True)

    target_column = dataframe['House_Price']
    means = dataframe.iloc[:, :-1].mean()
    standard_deviations = dataframe.iloc[:, :-1].std()
    scaled_dataframe = (dataframe.iloc[:, :-1] - means) / standard_deviations
    features = scaled_dataframe.to_numpy()
    scaled_dataframe['House_Price'] = target_column

    targets = target_column.to_numpy()

    return features, targets

def accuracy(features, targets, weights, bias):
    predictions = np.dot(features, weights) + bias
    residual_square_sum = np.sum((targets - predictions) ** 2)
    total_square_sum = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (residual_square_sum / total_square_sum)

    return r_squared

def gradient_descent(features, targets, learning_rate=0.01, epochs=500):
    bias = 0
    weights = np.zeros(features.shape[1])

    mean_square_errors = np.empty(epochs)
    r_squareds = np.empty(epochs)

    for iternum in range(epochs):
        errors = np.dot(features, weights) + bias - targets

        mean_square_errors[iternum] = (0.5 / len(features)) * np.sum(errors ** 2)
        r_squareds[iternum] = accuracy(features, targets, weights, bias)

        weight_gradient = (1 / len(features)) * np.dot(features.T, errors)
        bias_gradient = (1 / len(features)) * np.sum(errors)

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    return weights, bias, mean_square_errors, r_squareds

def graphing(mean_square_errors, r_squareds):
    epoch = range(len(mean_square_errors))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epoch, mean_square_errors)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE v Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epoch, r_squareds)
    plt.xlabel('Epoch')
    plt.ylabel('R^2')
    plt.title('R^2 v Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    datafile = 'house_price_regression_dataset.csv'
    training_file, testing_file = datasplit(datafile)
    x, y = processData(training_file)
    w, b, mse, r2 = gradient_descent(x, y)
    x, y = processData(testing_file)
    print(f"{accuracy(x, y, w, b) * 100}% accuracy")
    print("observed with the testing set")
    graphing(mse, r2)