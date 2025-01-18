# imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import kernels as kn

# for clinical kernel calculation
nominal_variables=['D135_REASON_STOP','D141_TRIAL_ADEQUATE','S103_GENDER','S140_ETHNICITY','S180_HIPPOCAMP_SCLEROSIS','S181_HIPPOCAMP_SCLER_LEFT','S182_HIPPOCAMP_SCLER_RIGHT','S188_POSITIVE_FAMHX',
                  'S190_NEUROL_PROGR_DISORDER','S200_NEUROL_EXAM_RESULT','S239_SEIZ_PHOTOSENSITIVE','S240_SEIZ_PRIM_GEN_TON_CLON','S241_SEIZ_ABSENCE','S242_SEIZ_CLONIC','S243_SEIZ_TONIC',
                  'S244_SEIZ_ATONIC','S245_SEIZ_MYOCLONIC','S246_SEIZ_SIMPL_PART','S247_SEIZ_COMPL_PART','S248_SEIZ_SEC_GTC','S249_SEIZ_UNCLASS_PART','S250_SEIZ_UNCLASS_GTC','S251_SEIZ_UNCERT_EPIL',
                  'S252_SEIZ_NON_EPIL','S253_SEIZ_FEBRILE','S254_FIRST_SEIZ_STATUS','S305_AED_NON_EPI_SEIZ','S314_AED_RESP_VPT_LTG','AED_ct','diagnosis']
ordinal_variables = ['D127_SEIZ_FRQ_BEFORE_COMB','D132_SEIZ_FRQ_AFTER_COMB','S264_SEIZ_NAED_GTC_ABS','S265_SEIZ_NAED_GTC_CAT','S266_SEIZ_NAED_NGTC_ABS','S267_SEIZ_NAEDNGTC_CAT','S268_SEIZ_NAED_COMB_ABS',
                    'S269_SEIZ_NAED_COMB_CAT','S274_SEIZ_N12_GTC_ABS','S275_SEIZ_N12_GTC_CAT','S276_SEIZ_N12_NGTC_ABS','S277_SEIZ_N12_NGTC_CAT','S278_SEIZ_N12_COMB_ABS','S279_SEIZ_N12_COMB_CAT',
                    'age_diagnosis','age_first_seizure','seiz_frequency_before_ct','age_ct','number_treatment_episode']


# load data
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Give the path to the kernel matrix.",
                        type=str)
    parser.add_argument("-y", "--outcome", help="Give the path to the outcome vector.",
                        type=str)
    parser.add_argument("-k", "--kernel", help="Give the kernel, poly2, poly3, rbf, linear, custom'.",
                        type=str)
    return parser

parser = make_argparser()
args = parser.parse_args()
X = pd.read_csv(args.data)
#X = X.values
y = pd.read_csv(args.outcome)
y = y.values.flatten()
k = args.kernel
n_splits = 5
# list of hyperparameters
C = [0.5,0.75,1.0,1.25,1.5]
# do cross validation with only the train set 
cv = StratifiedKFold(n_splits=n_splits)

# define kernel and degree for the model based on input argument
kernel = ''
degree = ''
if k == 'poly2':
  kernel = 'poly'
  degree = 2
elif k == 'poly3':
  kernel = 'poly'
  degree = 3
elif k == 'rbf':
  kernel = 'rbf'
elif k == 'linear':
  kernel = 'linear'
else:
  kernel = 'precomputed'
print(kernel)
print(degree)
# run SVM 
mcc, balanced_accuracy, auc, auc_pr = [], [], [], []
if kernel == 'precomputed':
    kernel_matrix = np.zeros((len(X),len(X)))
    kernel_matrix = kn.clinical_kernel(kernel_matrix,X,ordinal_variables,nominal_variables)
for fold, (outer_train, outer_test) in enumerate(cv.split(X, y)):
    # here starts the inner loop of the cross validation where we find the best hyperparameter setting
    mean_errors = []
    for parameter in C:
		# loop through every parameter setting in C 
        X_train = X.iloc[outer_train]
        y_train = y[outer_train]
        # split the training data from the outer loop into train and test 
        for fold, (inner_train, inner_test) in enumerate(cv.split(X_train, y_train)):
            # initialize a svm with the specific hyperparameter setting
            if kernel == 'poly':
                clf = svm.SVC(kernel=kernel,C = C[fold], degree=degree)
            else:
                clf = svm.SVC(kernel=kernel,C = C[fold])
            errors = []
	    # calculate kernel matrix and train model if kernel = precomputed
            if kernel == 'precomputed':
                inner_train_matrix, inner_test_matrix = kn.split_gram2(kernel_matrix,inner_train, inner_test)
                clf.fit(inner_train_matrix,y[inner_train])
                errors.append(clf.score(inner_test_matrix, y[inner_test]))
            else:
                clf.fit(X.iloc[inner_train], y[inner_train])
            	# append errors for all folds of the inner cross validation
                errors.append(clf.score(X.iloc[inner_test], y[inner_test]))
        # calculate the mean error for one hyperparameter setting
        mean_errors.append(np.mean(errors))
    # calculate the best hyperparameter setting (with the smallest mean error)
    min_error = min(mean_errors)
    best_hyperparameter = C[mean_errors.index(min_error)]
    print(best_hyperparameter)
    # for every fold of the outer cross validation loop we use the best hyperparameters
  	# we train and fit our model on the outer_train and outer_test data    
    if kernel == 'poly':
        clf = svm.SVC(kernel=kernel,C = best_hyperparameter, degree=degree)
    else:
        clf = svm.SVC(kernel=kernel,C = best_hyperparameter)
    if kernel == 'precomputed':
        outer_train_matrix, outer_test_matrix = kn.split_gram2(kernel_matrix,outer_train, outer_test)
        clf.fit(outer_train_matrix,y[outer_train])
        y_pred = clf.predict(outer_test_matrix)

    else:
        clf.fit(X.iloc[outer_train], y[outer_train])
        y_pred = clf.predict(X.iloc[outer_test])

# calculate mean accuracy over all cross validation splits and its standard deviation
    mcc.append(matthews_corrcoef(y_pred,y[outer_test]))
    balanced_accuracy.append(balanced_accuracy_score(y_pred,y[outer_test]))
'''
# draw roc curve for every cross validation iteration
    viz = RocCurveDisplay.from_estimator(
        clf,
        X.iloc[outer_test],
        y[outer_test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(fold == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# add the mean reciever operating curve for all cross validation folds and the standard deviation
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Mean ROC curve with individual ROC curves for each cross validation fold",
)
ax.axis("square")
ax.legend(loc="lower right")
plt.show()
plt.savefig('/home/julia.hellmig/reports/epipgx/complete/Results/roc_curve_svm_kernel%s.png' % k)
'''
# add some quality measures or visualization....
print("mean matthews correlation coefficient: %.2f" % np.mean(mcc))
print('SD for mcc: %.2f' % np.std(mcc))
print("mean balanced accuracy: %.2f" % np.mean(balanced_accuracy))
print('SD for balanced accuracy: %.2f' % np.std(balanced_accuracy))
