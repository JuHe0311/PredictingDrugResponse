# imports
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to dataset.",
                        type=str)
    parser.add_argument("-y", "--outcome", help="Give the path to the outcome vector.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
X = pd.read_csv(args.data)
X = X.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()

# list of hyperparameters
max_features = ['sqrt', 'log2', None]
n_estimators = [25,50, 100,125, 150]

# cross validation 
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)

auc_pr_scores = []

# impute missing values with mean (missing values are represented as -1)
imputer = SimpleImputer(missing_values=-1, strategy='mean')  # Replace -1 with the mean of the column
X = imputer.fit_transform(X)

# outer loop for performance evaluation
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    mean_errors = []
    current_best_error = 1000
    current_best_max_features = -1
    current_best_ne = -1
    X_train = X[outer_train]
    y_train = y[outer_train]
    for mf in max_features:
        for ne in n_estimators:
            # inner loop for hyperparameter fitting
            for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
                clf = RandomForestClassifier(n_estimators=ne, max_features=mf, max_depth=None, random_state=0, min_samples_split=2)
                errors = []
                # train model
                clf.fit(X[inner_train], y[inner_train])
                # append errors 
                errors.append(clf.score(X[inner_test], y[inner_test]))
            # calculate the mean error and update best hyperparameters
            if np.mean(errors) < current_best_error:
                current_best_max_features = mf
                current_best_ne = ne
            mean_errors.append(np.mean(errors))
    # calculate the best hyperparameter setting 
    min_error = min(mean_errors) 
    print(current_best_max_features,current_best_ne)
    clf = RandomForestClassifier(n_estimators=current_best_ne, max_features=current_best_max_features, max_depth=None,min_samples_split=2,
                                 random_state=0)
    # train model
    clf.fit(X[outer_train], y[outer_train])
    # predict on test set
    y_pred = clf.predict(X[outer_test])

    # calculate AUC-PR 
    auc_pr_scores.append(average_precision_score(y[outer_test], y_pred))
    
print("mean auc_pr score: %.2f" % np.mean(auc_pr_scores))
print('SD for auc_pr scores: %.2f' % np.std(auc_pr_scores))
