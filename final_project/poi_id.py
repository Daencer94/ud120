#!/usr/bin/python


import sys
import pickle
sys.path.append("../tools/")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import tester
import features

import pandas as pd
import pprint
import matplotlib.pyplot
pp = pprint.PrettyPrinter(indent=4)


def split_data(features, labels, test_size, random_state=42):
    """
    Support function for test_clf() that returns features and labels
    for both training and testing sets.
    Args:
    features: data features
    labels: data labels
    test_size: (float between 0 and 1) determines the fraction of points to
        be allocated to the test sample
    random_state: (int) ensures results are consistent across tests,
        recommended to drop on production
    Output: four sets of features and labels, for both training and testing
    """
    f_train, f_test, l_train, l_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state)

    return f_train, f_test, l_train, l_test



def get_size(dataset):
    """
    Returns the size of a list.
    """

    return len(dataset)


def count_poi(dataset):
    """
    Returns the number of POI from the Enron Dataset.
    """
    n = 0
    for person in dataset:
        if dataset[person]["poi"]:
            n += 1

    return n


def count_nan(dataset):
    """
    Returns a dictionary with these key-value pairs:
        key = feature name
        value = number of NaNs the feature has across the dataset
    """
    d = {}
    for person in dataset:
        for key, value in dataset[person].iteritems():
            if value == "NaN":
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1

    return d


def nan_replacer(dataset):
    ff = [
        "salary",
        "deferral_payments",
        "total_payments",
        "loan_advances",
        "bonus",
        "restricted_stock_deferred",
        "deferred_income",
        "total_stock_value",
        "expenses",
        "exercised_stock_options",
        "other",
        "long_term_incentive",
        "restricted_stock",
        "director_fees"
    ]
    for f in ff:
        for person in dataset:
            if dataset[person][f] == "NaN":
                dataset[person][f] = 0

    return dataset


def sort_data(dataset, feature, results, reverse=False):
    """
    Returns an array of sorted data.
    Args:
    dataset: a dictionary containing the data
    feature: the dictionary key indicating the feature to be sorted
    results: an integer indicating the number of results to be output
    reverse: a boolean indicating the order of the results (default: False)
    Output: an array with the sorted results
    """
    features = [feature]
    data = featureFormat(dataset, features)

    s = sorted(data, key=lambda x: x[0], reverse=reverse)[:int(results)]

    return s


def get_name(dataset, feature, value):
    """
    Returns the matching name of a person, given a feature and its value.
    """
    for p in dataset:
        if dataset[p][feature] == value:

            return p

    return


def get_incompletes(dataset, threshold):
    """
    Returns an array of person names that have no information (NaN) in a
    percentage of features above a given threshold (between 0 and 1).
    """
    incompletes = []
    for person in dataset:
        n = 0
        for key, value in dataset[person].iteritems():
            if value == 'NaN' or value == 0:
                n += 1
        fraction = float(n) / float(21)
        if fraction > threshold:
            incompletes.append(person)

    return incompletes


def scatterplot(dataset, var1, var2):
    """
    Creates and shows a scatterplot given a dataset and two features.
    """
    features_name = [str(var1), str(var2)]
    features = [var1, var2]
    data = featureFormat(dataset, features)

    for point in data:
        var1 = point[0]
        var2 = point[1]
        matplotlib.pyplot.scatter(var1, var2)

    matplotlib.pyplot.xlabel(features_name[0])
    matplotlib.pyplot.ylabel(features_name[1])
    matplotlib.pyplot.show()

    return


def create_feature(data, feat1, feat2, newfeat):
    """
    Creates a new numerical feature within a dataset resulting from dividing
    two already pre-existing features.
    Args:
    d: a dictionary containing the data
    f1: existing feature (numerator)
    f2: existing feature (denominator)
    f: resulting new feature
    Output: a dictionary with the new feature added, if the denominator is
    either is zero or NaN, it returns 0.0 or NaN.
    """
    for p in data:
        if data[p][feat2] == 0:
            data[p][newfeat] = 0.0
        elif data[p][feat1] == "NaN" or data[p][feat2] == "NaN":
            data[p][newfeat] = "NaN"
        else:
            data[p][newfeat] = float(data[p][feat1]) / float(data[p][feat2])

    return data

def kbest(data_dict, features_list):
    """
    Prints an ordered array with k best features based on SelectKBased.
    """
    # Keep only the values from features_list
    data = featureFormat(data_dict, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    # Set up the scaler
    minmax_scaler = preprocessing.MinMaxScaler()
    features_minmax = minmax_scaler.fit_transform(features)

    # Use SelectKBest to tune for k
    k_best = SelectKBest(chi2, k=10)

    # Use the instance to extract the k best features
    features_kbest = k_best.fit_transform(features_minmax, labels)

    feature_scores = ['%.2f' % elem for elem in k_best.scores_]

    # Get SelectKBest pvalues, rounded to 3 decimal places
    feature_scores_pvalues = ['%.3f' % elem for elem in k_best.pvalues_]

    # Get SelectKBest feature names, from 'K_best.get_support',
    # Create an array of feature names, scores and pvalues
    k_features = [(features_list[i+1],
                   feature_scores[i],
                   feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]

    # Sort the array by score
    k_features = sorted(k_features, key=lambda f: float(f[1]))

    print "# KBEST FEATURES:"
    print k_features
    print "\n"

    return


def feature_importances(d, features_list, test_size, random_state=42):
    """
    Prints an ordered list of the feature imporances for a given classifier.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    # Create both training and test sets through split_data()
    features_train, features_test, labels_train, labels_test = split_data(features, labels, test_size, random_state)

    classifier = ["ADA", "RF", "SVC"]
    for c in classifier:
        if c == "ADA":
            clf = AdaBoostClassifier()
        elif c == "RF":
            clf = RandomForestClassifier()
        elif c == "SVM":
            clf = SVC(kernel='linear', max_iter=1000)

        result = []
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        importances = clf.feature_importances_

        for i in range(len(importances)):
            t = [features_list[i], importances[i]]
            result.append(t)

        result = sorted(result, key=lambda x: x[1], reverse=True)

        print "# FEATURE IMPORTANCE:", c
        print result
        print "\n"

    return

def rf_tune(d, features_list, scaler):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    if scaler:
        rf = Pipeline([('scaler', StandardScaler()),
                       ('rf', RandomForestClassifier())])
    else:
        rf = Pipeline([('rf', RandomForestClassifier())])

    param_grid = ([{'rf__n_estimators': [4, 5, 10, 500]}])

    rf_clf = GridSearchCV(rf,
                          param_grid,
                          scoring='f1').fit(
                            features, labels).best_estimator_

    tester.test_classifier(rf_clf, d, features_list)

    return


def ab_tune(d, features_list, scaler):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    if scaler:
        ab = Pipeline([('scaler', StandardScaler()),
                       ('ab', AdaBoostClassifier())])
    else:
        ab = Pipeline([('ab', AdaBoostClassifier())])

    param_grid = ([{'ab__n_estimators': [1, 5, 10, 50]}])

    ab_clf = GridSearchCV(ab,
                          param_grid,
                          scoring='recall').fit(
                            features, labels).best_estimator_

    tester.test_classifier(ab_clf, d, features_list)

    return


def svc_tune(d, features_list, scaler=True):
    """
    Prints the key metrics of an algorithm after it has gone through the
    tuning process with Pipleine and GridSearchCV.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    param_grid = ([{'svm__C': [1, 50, 100, 1000],
                    'svm__gamma': [0.5, 0.1, 0.01],
                    'svm__degree': [1, 2],
                    'svm__kernel': ['rbf', 'poly', 'linear'],
                    'svm__max_iter': [1, 100, 1000]}])

    svm_clf = GridSearchCV(svm,
                           param_grid,
                           scoring='f1').fit(
                           features, labels).best_estimator_

    tester.test_classifier(svm_clf, d, features_list)

    return


def get_svc(d, features_list):
    """
    Generates the classifier for final submission.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    param_grid = ([{'svm__C': [50],
                    'svm__gamma': [0.1],
                    'svm__degree': [2],
                    'svm__kernel': ['poly'],
                    'svm__max_iter': [100]}])

    svm_clf = GridSearchCV(svm,
                           param_grid,
                           scoring='f1').fit(
                           features, labels).best_estimator_

    return svm_clf


def test_clf(d, features_list, random_state=42):
    """
    Returns the classifier performance under different train / test ratios.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    test_sizes = [0.2, 0.4, 0.6]

    for test_size in test_sizes:
        # Create both training and test sets through split_data()
        features_train, features_test, labels_train, labels_test = split_data(
            features,
            labels,
            test_size,
            random_state)

        clf = get_svc(d, features_list)

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        print "# METRICS FOR TEST SIZE OF:", test_size
        acc = accuracy_score(labels_test, pred)
        print "* Accuracy:", acc

        pre = precision_score(labels_test, pred)
        print "* Precision:", pre

        rec = recall_score(labels_test, pred)
        print "* Recall:", rec
        print "\n"

    return


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# Get the size of the dictionary containing the dataset
print "* Dataset length:", get_size(data_dict)
# Get the number of POI in the dataset
print "* Number of POI:", count_poi(data_dict)

print "* List of NaNs per feature once replaced NaNs for zeros:"
pp.pprint(count_nan(data_dict))

#all features
f = []
for person in data_dict:
    for key, value in data_dict[person].iteritems():
        if key not in f:
            f.append(key)

print f

# Replace all the NaNs in financial features with zeros
data_dict = nan_replacer(data_dict)

print "* List of NaNs per feature once replaced NaNs for zeros:"
pp.pprint(count_nan(data_dict))

### Task 2: Remove outliers

fig1 = scatterplot(data_dict, "salary", "bonus")

outlier = sort_data(data_dict, "salary", 1, reverse=True)

outlier_name = get_name(data_dict, "salary", outlier[0])
print "* Outlier found:", outlier_name

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

enronDataFrame = pd.DataFrame.from_dict(data_dict, orient='index')

errorsPayment = (enronDataFrame[enronDataFrame[['salary',
                                                'bonus',
                                                'long_term_incentive',
                                                'deferred_income',
                                                'deferral_payments',
                                                'loan_advances',
                                                'other',
                                                'expenses',
                                                'director_fees']].sum(axis='columns') != enronDataFrame['total_payments']])

errorsStock = (enronDataFrame[enronDataFrame[['exercised_stock_options',
                                              'restricted_stock',
                                              'restricted_stock_deferred']].sum(axis='columns') != enronDataFrame['total_stock_value']])

print('Payment errors:',errorsPayment.index)
print('Stock errors:',errorsStock.index)

data_dict.pop('BELFER ROBERT')
data_dict.pop('BHATNAGAR SANJAY')

# Get the size of the dictionary containing the dataset
print "* Dataset length:", get_size(data_dict)
# Get the number of POI in the dataset
print "* Number of POI:", count_poi(data_dict)

# print "* List of persons with more than 90% of data missing:"
# #pp.pprint(get_incompletes(data_dict, 0.90))
# outlieres = get_incompletes(data_dict, 0.90)
# pp.pprint(outlieres)
#
# for o in outlieres:
#     data_dict.pop(o)

feat_1 = features.feat_1


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# print "Settings: \n* Features: default \n* Tuning: default"
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, feat_1)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, feat_1)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, feat_1)


feat_2 = features.feat_2

# Get new performance with feat_2
print "Settings: \n* Features: remove features with +50% NaNs in POI \n* Tuning: default"
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, feat_2)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, feat_2)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, feat_2)
print "\n"

data_dict = create_feature(data_dict,
                                      "bonus",
                                      "total_payments",
                                      "f_bonus")
# f_salary
data_dict = create_feature(data_dict,
                                      "salary",
                                      "total_payments",
                                      "f_salary")
# f_stock
data_dict = create_feature(data_dict,
                                      "total_stock_value",
                                      "total_payments",
                                      "f_stock")
# r_from
data_dict = create_feature(data_dict,
                                      "from_this_person_to_poi",
                                      "from_messages",
                                      "r_from")
# r_to
data_dict = create_feature(data_dict,
                                      "from_poi_to_this_person",
                                      "to_messages",
                                      "r_to")

# Add new engineered features
feat_3 = features.feat_3

print "Settings: \n* Features: add engineered features \n* Tuning: default"
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, feat_3)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, feat_3)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, feat_3)
print "\n"

kbest(data_dict, feat_1)

feat_K = features.feat_K

# Get new performance with feat_K
print "Settings: \n* Features: SelectKBest \n* Tuning: default"
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, feat_K)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, feat_K)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, feat_K)
print "\n"


feature_importances(data_dict, feat_3, 0.35)

# Remove less important features from feature_importances_
feat_4AB = features.feat_4AB
feat_4RF = features.feat_4RF
feat_4SVC = features.feat_4SVC

# Get new performance with feat_4*
print "Settings: \n* Features: .feature_importances_ \n* Tuning: default"
clf_AB = AdaBoostClassifier()
tester.test_classifier(clf_AB, data_dict, feat_4AB)
clf_RF = RandomForestClassifier()
tester.test_classifier(clf_RF, data_dict, feat_4RF)
clf_SVC = SVC(kernel='linear', max_iter=1000)
tester.test_classifier(clf_SVC, data_dict, feat_4SVC)
print "\n"

my_dataset = data_dict

df = pd.DataFrame.from_dict(my_dataset, orient='index')
df.to_excel('enron.xlsx')


ab_tune(data_dict, feat_4AB, True)
rf_tune(data_dict, feat_4RF, True)
svc_tune(data_dict, feat_4SVC)

# Code Review: store to my_dataset and my_features for easy export below
my_clf = get_svc(data_dict, feat_4SVC)
my_dataset = data_dict
my_features = feat_4SVC

dump_classifier_and_data(my_clf, my_dataset, my_features)

# Validate the algorithm
test_clf(data_dict, feat_4SVC, random_state=42)