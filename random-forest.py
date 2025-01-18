# imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, balanced_accuracy_score, RocCurveDisplay, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the kernel matrix.",
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
n_splits = 5
# list of hyperparameters
max_features = ['sqrt', 'log2', None]
n_estimators = [25,50, 100,125, 150]
# do cross validation with only the train set 
cv = StratifiedKFold(n_splits=n_splits)
mcc, f1, auc_scores, balanced_accuracy, auc_pr_scores = [], [], [], [], []
imputer = SimpleImputer(missing_values=-1, strategy='mean')  # Replace -1 with the mean of the column
X = imputer.fit_transform(X)
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    # here starts the inner loop of the cross validation where we find the best hyperparameter setting
    mean_errors = []
    current_best_error = 1000
    current_best_max_features = -1
    current_best_ne = -1
    for mf in max_features:
        for ne in n_estimators:
            # loop through every parameter setting in C 
            X_train = X[outer_train]
            y_train = y[outer_train]
            # split the training data from the outer loop into train and test 
            for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
                # initialize a logistic regression model with the specific hyperparameter setting
                clf = RandomForestClassifier(n_estimators=ne, max_features=mf, max_depth=None, random_state=0, min_samples_split=2)
                errors = []
                clf.fit(X[inner_train], y[inner_train])
                # append errors for all folds of the inner cross validation
                errors.append(clf.score(X[inner_test], y[inner_test]))
            # calculate the mean error for one hyperparameter setting
            if np.mean(errors) < current_best_error:
                current_best_max_features = mf
                current_best_ne = ne
            mean_errors.append(np.mean(errors))
    # calculate the best hyperparameter setting (with the smallest mean error)
    min_error = min(mean_errors) 
    print(current_best_max_features,current_best_ne)
    # for every fold of the outer cross validation loop we use the best hyperparameters
    # we train and fit our model on the outer_train and outer_test data    
    clf = RandomForestClassifier(n_estimators=current_best_ne, max_features=current_best_max_features, max_depth=None,min_samples_split=2,
                                 random_state=0)
    clf.fit(X[outer_train], y[outer_train])
    y_pred = clf.predict(X[outer_test])

    # calculate mean accuracy over all cross validation splits and its standard deviation
    mcc.append(matthews_corrcoef(y[outer_test], y_pred))
    f1.append(f1_score(y[outer_test], y_pred))
    balanced_accuracy.append(balanced_accuracy_score(y[outer_test], y_pred))
    auc_scores.append(roc_auc_score(y[outer_test], y_pred))
    auc_pr_scores.append(average_precision_score(y[outer_test], y_pred))
    
# add some quality measures or visualization....
print("mean matthews correlation coefficient: %.2f" % np.mean(mcc))
print('SD for MCC: %.2f' % np.std(mcc))
print("mean f1 score: %.2f" % np.mean(f1))
print('SD for f1 score: %.2f' % np.std(f1))
print("mean balanced accuracy: %.2f" % np.mean(balanced_accuracy))
print('SD for balanced accuracy: %.2f' % np.std(balanced_accuracy))
print("mean auc score: %.2f" % np.mean(auc_scores))
print('SD for auc scores: %.2f' % np.std(auc_scores))
print("mean auc_pr score: %.2f" % np.mean(auc_pr_scores))
print('SD for auc_pr scores: %.2f' % np.std(auc_pr_scores))
