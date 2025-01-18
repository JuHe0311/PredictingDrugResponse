# imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from scipy.special import expit  # Sigmoid function
from sklearn.inspection import permutation_importance

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
X = pd.read_csv(args.data,index_col=0)
feature_names = X.columns  # Save the feature names for later use
X = X.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()
y = (y == 1).astype(int)

n_splits = 5
# list of hyperparameters
learning_rate = [0.05, 0.075, 0.1, 0.15,0.2]
n_estimators = [50, 75, 100, 125, 150]

cv = StratifiedKFold(n_splits=n_splits)
mcc, f1, balanced_accuracy, best_params, auc_scores, pr_auc_scores,all_importances = [], [], [], [], [], [], []

for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    # here starts the inner loop of the cross validation where we find the best hyperparameter setting
    mean_errors = []
    current_best_error = 1000
    current_best_learning_rate = -1
    current_best_ne = -1
    for lr in learning_rate:
        for ne in n_estimators:
            # loop through every parameter setting in C 
            X_train = X[outer_train]
            y_train = y[outer_train]
            errors = []  
            # split the training data from the outer loop into train and test 
            for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
                # initialize a logistic regression model with the specific hyperparameter setting
                #xgb.DMatrix(X, y)
                clf = xgb.XGBClassifier(n_estimators=ne, learning_rate=lr,objective="binary:logistic",missing=-1,importance_type='gain')
                clf.fit(X[inner_train], y[inner_train])
                # append errors for all folds of the inner cross validation
                errors.append(1 - clf.score(X_train[inner_test], y_train[inner_test]))
            mean_errors.append(np.mean(errors))
    # calculate the mean error for one hyperparameter setting
    best_idx = np.argmin(mean_errors)
    best_lr = learning_rate[best_idx // len(n_estimators)]
    best_ne = n_estimators[best_idx % len(n_estimators)]
    best_params.append({'learning_rate': best_lr, 'n_estimators': best_ne})
    
    # for every fold of the outer cross validation loop we use the best hyperparameters
    # we train and fit our model on the outer_train and outer_test data    
    clf = xgb.XGBClassifier(n_estimators=best_ne, learning_rate=best_lr,objective="binary:logistic",missing=-1, importance_type='gain')
    clf.fit(X[outer_train], y[outer_train])
    y_pred = clf.predict(X[outer_test])
    # calculate mean accuracy over all cross validation splits and its standard deviation
    y_pred_test_prob = expit(y_pred)
    y_pred_test_bin = (y_pred_test_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
    mcc.append(matthews_corrcoef(y[outer_test], y_pred_test_bin))
    f1.append(f1_score(y[outer_test], y_pred_test_bin))
    balanced_accuracy.append(balanced_accuracy_score(y[outer_test], y_pred_test_bin))
    auc_score = roc_auc_score(y[outer_test], y_pred_test_bin)
    auc_scores.append(auc_score)
    pr_auc_scores.append(average_precision_score(y[outer_test], y_pred_test_bin))  # Precision-recall AUC
    # Get the permutation importance for the current outer fold
    importances_perm = permutation_importance(clf, X[outer_test], y_pred_test_bin, n_repeats=10)
    # Accumulate the importances for this fold
    all_importances.append(importances_perm.importances_mean)

# Convert the list of importances into a numpy array (shape: [n_folds, n_features])
all_importances = np.array(all_importances)

# Calculate the mean importance across all folds for each feature
mean_importance = np.mean(all_importances, axis=0)

# Sort the features by their mean importance
sorted_idx = mean_importance.argsort()[::-1]  # Sort in descending order

# Print the top 10 features and their importance
top_10_features = [(feature_names[i], mean_importance[i]) for i in sorted_idx[:10]]

# Output the top 10 features with their importance values
print("Top 10 Features and Their Average Importance Across All Folds:")
for feature, importance in top_10_features:
    print(f"{feature}: {importance}")
    
print('parameters:')
print(best_params)
# add some quality measures or visualization....
print("mean matthews correlation coefficient: %.2f" % np.mean(mcc))
print('SD for mcc: %.2f' % np.std(mcc))
print("mean balanced accuracy: %.2f" % np.mean(balanced_accuracy))
print('SD for balanced accuracy: %.2f' % np.std(balanced_accuracy))
print("mean AUC_ROC score: %.2f" % np.mean(auc_scores))
print('SD for AUC_ROC score: %.2f' % np.std(auc_scores))
print("mean AUC-PR score: %.2f" % np.mean(pr_auc_scores))
print('SD for AUC-PR score: %.2f' % np.std(pr_auc_scores))

# Get feature importances and print top 10 features
importances = clf.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = importance_df.sort_values(by='importance', ascending=False).head(10)


importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
top_features = importance_df.sort_values(by='importance', ascending=False).head(10)    
print("\nTop 10 Most Important Features:")
print(top_features)
cat_cols = ['D120_AED_GENERIC_ABBREV', 'D121_NON_ADHERENT',
'D127_SEIZ_FRQ_BEFORE_COMB', 'D141_TRIAL_ADEQUATE', 'D140_OUTCOME',
'S102_SUBJ_ID', 'S103_GENDER', 'S140_ETHNICITY',
'S164_EPI_DIAGN_STATUS', 'S180_HIPPOCAMP_SCLEROSIS',
'S181_HIPPOCAMP_SCLER_LEFT', 'S182_HIPPOCAMP_SCLER_RIGHT',
'S188_POSITIVE_FAMHX', 'S190_NEUROL_PROGR_DISORDER',
'S200_NEUROL_EXAM_RESULT', 'S239_SEIZ_PHOTOSENSITIVE',
'S240_SEIZ_PRIM_GEN_TON_CLON', 'S241_SEIZ_ABSENCE', 'S242_SEIZ_CLONIC',
'S243_SEIZ_TONIC', 'S244_SEIZ_ATONIC', 'S245_SEIZ_MYOCLONIC',
'S246_SEIZ_SIMPL_PART', 'S247_SEIZ_COMPL_PART', 'S248_SEIZ_SEC_GTC',
'S249_SEIZ_UNCLASS_PART', 'S250_SEIZ_UNCLASS_GTC',
'S251_SEIZ_UNCERT_EPIL', 'S252_SEIZ_NON_EPIL', 'S253_SEIZ_FEBRILE',
'S254_FIRST_SEIZ_STATUS','S265_SEIZ_NAED_GTC_CAT','S267_SEIZ_NAED_NGTC_CAT',
'S269_SEIZ_NAED_COMB_CAT','S275_SEIZ_N12_GTC_CAT','S277_SEIZ_N12_NGTC_CAT', 'S279_SEIZ_N12_COMB_CAT', 'S290_SEIZ_REMISS_YES_NO',
'S305_AED_NON_EPI_SEIZ', 'diagnosis', 'number_treatment_episode','focal_epilepsy','generalized_epilepsy']

# Determine influence directions of top features
for feature in top_features['feature']:
    feature_idx = feature_names.get_loc(feature)  # Get the column index for the feature
    X_feature = X[:, feature_idx]  # Extract the feature values
    print(f"\nFeature: {feature}")

    if feature not in cat_cols:
        # Continuous feature
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

    else:
        # Categorical feature
        print("Type: Categorical")
        categories = np.unique(X_feature)
        for category in categories:
            cat_indices = np.where(X_feature == category)[0]
            mean_pred = np.mean(clf.predict(X[cat_indices]))  # Direct probability for positive class
            print(f"  Category {category}: Mean prediction = {mean_pred:.3f}")
