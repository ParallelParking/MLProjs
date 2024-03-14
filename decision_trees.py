import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def processdata(file):
    df = pd.read_csv(file)
    targets = df['Threat']
    attributes = df.drop('Threat', axis=1)

    # rus = RandomUnderSampler()
    # attributes, targets = rus.fit_resample(attributes, targets)
    # TODO: why the fuck is this so much worse?

    smote = SMOTE()
    attributes, targets = smote.fit_resample(attributes, targets)

    train_attributes, test_attributes, train_targets, test_targets \
        = train_test_split(attributes, targets)

    return train_attributes, test_attributes, train_targets, test_targets


def trainmodel(train_attributes, train_targets):
    clf = DecisionTreeClassifier()
    clf.fit(train_attributes, train_targets)
    return clf


def testmodel(test_attributes, test_targets, clf):
    predictions = clf.predict(test_attributes)
    report = classification_report(test_targets, predictions)
    return report


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = processdata('threats.csv')
    classification_model = trainmodel(train_x, train_y)
    rep = testmodel(test_x, test_y, classification_model)
    print(rep)
