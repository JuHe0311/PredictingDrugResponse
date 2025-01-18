# imports
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import argparse
import xgboost as xgb
from scipy.special import expit  

# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path input data.",
                        type=str)
    parser.add_argument("-y", "--outcome", help="Give the path to the outcome vector.",
                        type=str)
    parser.add_argument("-c", "--categorical_columns", help="Give the list of column names that are categorical features for importance calculation",
                        type=str)
    return parser

# reading data and outcomes, transform for further use
parser = make_argparser()
args = parser.parse_args()
X = pd.read_csv(args.data,index_col=0)
feature_names = X.columns  # Save the feature names for later use
X = X.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()
y = (y == 1).astype(int)
categorical_columns = args.categorical_columns
# cross validation
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)

# list of hyperparameters
learning_rate = [0.05, 0.075, 0.1, 0.15,0.2]
n_estimators = [50, 75, 100, 125, 150]

# AUC-PR scores
pr_auc_scores = []

# outer loop for performance evaluation
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    mean_errors = []
    current_best_error = 1000
    current_best_learning_rate = -1
    current_best_ne = -1
    X_train = X[outer_train]
    y_train = y[outer_train]
    # loop through hyperparameter combinations
    for lr in learning_rate:
        for ne in n_estimators:
            errors = []  
            # inner loop for hyperparameter fitting  
            for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
                clf = xgb.XGBClassifier(n_estimators=ne, learning_rate=lr,objective="binary:logistic",missing=-1,importance_type='gain')
                # train classifier
                clf.fit(X[inner_train], y[inner_train])
                # append errors for inner folds
                errors.append(1 - clf.score(X_train[inner_test], y_train[inner_test]))
            mean_errors.append(np.mean(errors))
    # update best hyperparameters
    best_idx = np.argmin(mean_errors)
    best_lr = learning_rate[best_idx // len(n_estimators)]
    best_ne = n_estimators[best_idx % len(n_estimators)]
    best_params.append({'learning_rate': best_lr, 'n_estimators': best_ne})
    
    clf = xgb.XGBClassifier(n_estimators=best_ne, learning_rate=best_lr,objective="binary:logistic",missing=-1, importance_type='gain')
    # train classifier
    clf.fit(X[outer_train], y[outer_train])
    # predict on test set
    y_pred = clf.predict(X[outer_test])
    
    # calculate AUC-PR scores
    y_pred_test_prob = expit(y_pred)
    y_pred_test_bin = (y_pred_test_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
    pr_auc_scores.append(average_precision_score(y[outer_test], y_pred_test_bin))  
    

print('parameters:')
print(best_params)

print("mean AUC-PR score: %.2f" % np.mean(pr_auc_scores))
print('SD for AUC-PR score: %.2f' % np.std(pr_auc_scores))

# Calculate feature importances and print top 10 features and their importance
importances = clf.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = importance_df.sort_values(by='importance', ascending=False).head(10)
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = importance_df.sort_values(by='importance', ascending=False).head(10)    
print("\nTop 10 Most Important Features:")
print(top_features)

# Determine influence directions of top features
for feature in top_features['feature']:
    feature_idx = feature_names.get_loc(feature)  # Get the column index for the feature
    X_feature = X[:, feature_idx]  # Extract the feature values
    print(f"\nFeature: {feature}")

    # Continuous feature
    if feature not in categorical_columns:
        print("Type: Continuous")
        # Bin the feature values into quartiles
        bins = np.percentile(X_feature, [0, 25, 50, 75, 100])
        binned_feature = np.digitize(X_feature, bins, right=True)
        mean_preds = []
        
        for bin_label in range(1, len(bins)):
            bin_indices = np.where(binned_feature == bin_label)[0]
            if len(bin_indices) > 0:
                mean_preds.append(np.mean(clf.predict(X[bin_indices])))  # Direct probability for positive class
            else:
                mean_preds.append(np.nan)
        
        for i in range(len(bins) - 1):
            print(f"  Bin {i+1} ({bins[i]} to {bins[i+1]}): Mean prediction = {mean_preds[i]:.3f}")
        
        trend = "increasing" if np.nanmean(np.diff(mean_preds)) > 0 else "decreasing"
        print(f"  Overall trend: {trend}")
        
    # Categorical feature
    else:
        print("Type: Categorical")
        categories = np.unique(X_feature)
        for category in categories:
            cat_indices = np.where(X_feature == category)[0]
            mean_pred = np.mean(clf.predict(X[cat_indices]))  # Direct probability for positive class
            print(f"  Category {category}: Mean prediction = {mean_pred:.3f}")
