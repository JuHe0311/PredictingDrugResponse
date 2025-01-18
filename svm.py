# imports
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import PolynomialCountSketch
from collections import Counter
from sklearn.impute import SimpleImputer

# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the dataset.",
                        type=str)
    parser.add_argument("-y", "--outcome", help="Give the path to the outcome vector.",
                        type=str)
    parser.add_argument("-k", "--kernel", help="Specify the kernel to use.",
                        type=str)
    parser.add_argument("-degree", "--degree", help="Specify the degree for the polynomial kernel, is ignored when no polynomial kernel is used.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
X_df = pd.read_csv(args.data)
X = X_df.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()
kernel = args.kernel
degree = int(args.degree)

# AUC-PR scores
auc_pr_scores = []

# list of hyperparameters
C = [0.01, 0.1, 1, 10, 100]
current_best_error = 1000
current_best_C = 0

# impute missing data (represented as -1 in the data)
imputer = SimpleImputer(missing_values=-1, strategy='mean') 
X = imputer.fit_transform(X)

# cross-validation
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)

# outer loop performance evaluation
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    c_values = []
    # inner loop for hyperparameter settings
    for c in C:
        errors = []
        for fold, (inner_train, inner_test) in enumerate(cv.split(X[outer_train], y[outer_train])):
            clf = svm.SVC(kernel=kernel, degree=degree, C = c)
	    # train model
            clf.fit(X[inner_train], y[inner_train])
            errors.append(clf.score(X[inner_test], y[inner_test]))
        # update best hyperparameters
        if np.mean(errors) < current_best_error:
            current_best_C = c
            current_best_error = np.mean(errors)
        c_values.append(current_best_C)    
    # select best hyperparameters
    c_counts = Counter(c_values)
    final_best_C, count = c_counts.most_common(1)[0]
    clf = svm.SVC(kernel=kernel, degree=degree, C = final_best_C)
    # train model
    clf.fit(X[outer_train], y[outer_train])
    # predict on test set
    y_pred = clf.predict(X[outer_test])

    auc_pr_scores.append(average_precision_score(y[outer_test], y_pred))
    
print("mean auc_pr scores: %.2f" % np.mean(auc_pr_scores))
print('SD for auc_pr scores: %.2f' % np.std(auc_pr_scores))
