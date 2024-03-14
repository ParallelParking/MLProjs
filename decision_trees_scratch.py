import pandas as pd
import numpy as np


def processdata(file):
    df = pd.read_csv(file)
    targets = df['Threat']
    attributes = df.drop('Threat', axis=1)

    return attributes, targets


class KNearestNeighbours:
    def __init__(self, k=5):
        self.k = k

    def fit(self, attributes, targets):
        self.x = attributes
        self.y = targets

    def predict(self, ):

def smote(attributes, targets, minority=0, oversample_ratio=1, nearest_neighbours_count=5):
    minority_indices = np.where(attributes == minority)[0]
    majority_indices = np.where(attributes != minority)[0]

    # imbalance = len(minority_indices) / len(majority_indices)
    # imbalance == 0.02235871795426764 == 2.2%
    # ideal imbalance == 1.0, high but imo better to have false positives on cybersec than the alternative.

    synthetic_samples_count = int((len(minority_indices) * oversample_ratio) - len(minority_indices))



if __name__ == "__main__":
    x, y = processdata('threats.csv')
    print(smote(x, y))