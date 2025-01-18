# imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, balanced_accuracy_score, RocCurveDisplay, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.impute import SimpleImputer

# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the kernel matrix.",
                        type=str)
    parser.add_argument("-y", "--outcome", help="Give the path to the outcome vector.",
                        type=str)
    parser.add_argument("-l", "--loss_error", help="Give the loss to use 'l1' 'l2' or 'elasticnet'.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
X = pd.read_csv(args.data)
X = X.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()
l = args.loss_error
n_splits = 5
# list of hyperparameters
C = [0.5,0.75,1.0,1.25,1.5]
# do cross validation with only the train set 
cv = StratifiedKFold(n_splits=n_splits)
# Assuming your data is a NumPy array or pandas DataFrame
imputer = SimpleImputer(missing_values=-1, strategy='mean')  # Replace -1 with the mean of the column
X = imputer.fit_transform(X)
# run SVM with the gram matrices
clf = LogisticRegression(random_state=0,penalty=l,solver='saga',l1_ratio=0.5 if l == 'elasticnet' else None)
mcc, balanced_accuracy, auc_scores, auc_pr_scores = [], [], [], []
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    # here starts the inner loop of the cross validation where we find the best hyperparameter setting
    mean_errors = []
    for parameter in C:
		# loop through every parameter setting in C 
        X_train = X[outer_train]
        y_train = y[outer_train]
        # split the training data from the outer loop into train and test 
        for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
            # initialize a logistic regression model with the specific hyperparameter setting
            clf = LogisticRegression(random_state=0,penalty=l,C = C[fold], solver='saga',l1_ratio=0.5 if l == 'elasticnet' else None)
            errors = []
            clf.fit(X[inner_train], y[inner_train])
            # append errors for all folds of the inner cross validation
            errors.append(1 - clf.score(X_train[inner_test], y_train[inner_test]))
        # calculate the mean error for one hyperparameter setting
        mean_errors.append(np.mean(errors))
    # calculate the best hyperparameter setting (with the smallest mean error)
    min_error = min(mean_errors)
    best_hyperparameter = C[mean_errors.index(min_error)]
    print(best_hyperparameter)
    # for every fold of the outer cross validation loop we use the best hyperparameters
	# we train and fit our model on the outer_train and outer_test data    
    clf = LogisticRegression(random_state=0,penalty=l,solver='saga',C = best_hyperparameter, l1_ratio=0.5 if l == 'elasticnet' else None)
    clf.fit(X[outer_train], y[outer_train])
    y_pred = clf.predict(X[outer_test])

# calculate mean accuracy over all cross validation splits and its standard deviation
    mcc.append(matthews_corrcoef(y[outer_test], y_pred))
    balanced_accuracy.append(balanced_accuracy_score(y[outer_test], y_pred))
    auc_scores.append(roc_auc_score(y[outer_test], y_pred))
    auc_pr_scores.append(average_precision_score(y[outer_test], y_pred))
	
# add some quality measures or visualization....
print("mean matthews correlation coefficient: %.2f" % np.mean(mcc))
print('SD for mcc: %.2f' % np.std(mcc))
print("mean balanced accuracy: %.2f" % np.mean(balanced_accuracy))
print('SD for balanced accuracy: %.2f' % np.std(balanced_accuracy))
print("mean auc score: %.2f" % np.mean(auc_scores))
print('SD for auc scores: %.2f' % np.std(auc_scores))
print("mean auc_pr score: %.2f" % np.mean(auc_pr_scores))
print('SD for auc_pr scores: %.2f' % np.std(auc_pr_scores))
