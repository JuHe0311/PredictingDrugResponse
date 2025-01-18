import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd 
from sklearn.metrics import matthews_corrcoef,roc_auc_score, average_precision_score
from scipy.special import expit  
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold

def calculate_mcc(y_true, y_pred):
    """
    Calculate MCC 
    
    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
    :return: MCC value.
    """
    return matthews_corrcoef(y_true, y_pred)

def calculate_auc(y_true, y_pred):
    """
    Calculate AUC score

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
    :return: AUC value.
    """
    return roc_auc_score(y_true, y_pred)

def calculate_auc_pr(y_true, y_pred):
    """
    Calculate AUC score for the precision recall curve

    :param y_true: Array of true labels.
    :param y_pred: Array of predicted labels.
    :return: AUC value.
    """
    return average_precision_score(y_true, y_pred)
   
class CustomCART:
    def __init__(self, max_depth=3, min_samples_split=2, shrinkage=0.1, mu_g=0.01, mu_t=0.01):
        """
        Initialize the custom CART tree.
        
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum samples required to split a node.
        :param shrinkage: Learning rate (shrinkage for boosting).
        :param mu_g: Global penalty parameter for feature selection.
        :param mu_t: Task-specific penalty parameter for feature selection.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.shrinkage = shrinkage
        self.mu_g = mu_g
        self.mu_t = mu_t
        self.tree = None
        self.selected_features = set()  # Track selected features
        self.feature_importances_ = {}  # Track feature importance as a dictionary
        
    def fit(self, X, y, Ω_t, Ω_G):
        """
        Fit the decision tree on the data using a custom loss function.

        :param X: Feature matrix (n_samples x n_features).
        :param y: Target values (n_samples,).
        :param Ω_t: Task-specific selected features.
        :param Ω_G: Global selected features.
        """
        X = X.values
        self.tree = self._build_tree(X, y, depth=0, Ω_t=Ω_t, Ω_G=Ω_G)

    def _build_tree(self, X, y, depth, Ω_t, Ω_G):
        """
        Recursively build a CART tree with a custom loss function.

        :param X: Feature matrix (n_samples x n_features).
        :param y: Target values (n_samples,).
        :param depth: Current depth of the tree.
        :param Ω_t: Task-specific selected features.
        :param Ω_G: Global selected features.
        :return: A tree node (dictionary).
        """
        n_samples, n_features = X.shape
      
        # stop criterion - when max depth is reached or the samples for a split are not enough
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            # Create leaf node with prediction (mean of residuals)
            leaf_value = np.mean(y)  
            return {"leaf": leaf_value}

        # no stop scenario - split the tree into 2 children
        best_split, loss, old_min_loss = self._find_best_split(X, y, Ω_t, Ω_G)
        if best_split is None:
            # No valid split found, create leaf node
            leaf_value = np.mean(y)
            return {"leaf": leaf_value}
    
        # Create masks for splitting and recursively build left and right subtrees
        missing_mask = X[:, best_split["feature"]] == -1
        left_mask = (X[:, best_split["feature"]] <= best_split["threshold"]) & ~missing_mask
        right_mask = ~left_mask & ~missing_mask

        # if left or right tree are empty return a leaf
        if len(left_mask) == 0 or len(right_mask) == 0:
            leaf_value = np.mean(y)
            return {"leaf": leaf_value}

        if best_split is not None:    
            # Record the feature used for this split
            self.selected_features.add(best_split["feature"])

            self.feature_importances_[best_split["feature"]] = self.feature_importances_.get(best_split["feature"], 0) + 1

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1, Ω_t, Ω_G)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1, Ω_t, Ω_G)
        missing_node = self._build_tree(X[missing_mask], y[missing_mask], depth + 1, Ω_t, Ω_G)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_node,
            "right": right_node,
            "missing": missing_node
        }

    def _find_best_split(self, X, y, Ω_t, Ω_G):
        """
        Find the best feature and threshold to split the data based on custom loss function.

        :param X: Feature matrix (n_samples x n_features).
        :param y: Target values (n_samples,).
        :param Ω_t: Task-specific selected features.
        :param Ω_G: Global selected features.
        :return: Best split (dictionary with feature and threshold) or None if no valid split and the importance contribution of the 
        feature or 0 if there was no valid split.
        """
        n_samples, n_features = X.shape
        best_split = None
        min_loss = float("inf")
        
        # Create a shuffled order of features
        feature_indices = np.arange(n_features)
        np.random.seed(42)  # Set a fixed random seed for reproducibility
        np.random.shuffle(feature_indices)  # Shuffle the feature indices

        for feature in feature_indices:
            unique_values = np.unique(X[:, feature][X[:, feature] != -1])  # Exclude missing values (-1)
            for threshold in unique_values:
                left_mask = (X[:, feature] <= threshold) & (X[:, feature] != -1)
                right_mask = ~left_mask & ((X[:, feature] != -1))
                missing_mask = X[:, feature] == -1
                left_y, right_y, missing_y = y[left_mask], y[right_mask], y[missing_mask]

                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue  # Skip if the split doesn't meet the minimum sample requirement
                
                # Calculate loss function for this split
                loss = self.calculate_loss(left_y, right_y, missing_y, feature, threshold, Ω_t, Ω_G)
                if loss < min_loss:
                    importance = min_loss - loss
                    old_min_loss = min_loss
                    min_loss = loss
                    best_split = {"feature": feature, "threshold": threshold}
         # Calculate the importance for the best feature using the min_loss - loss approach
        if best_split:
            importance_contribution = old_min_loss - loss  # Feature importance from loss reduction
            return best_split, loss, old_min_loss

        return None, 0,0  # If no valid split is found, return no split and zero importance
    
    def calculate_loss(self, left_y, right_y, missing_y, feature, threshold, Ω_t, Ω_G):
        """
        Calculate the loss function (with penalties) for a given split.

        :param left_y: Target values for the left split.
        :param right_y: Target values for the right split.
        :param feature: Feature used for the split.
        :param threshold: Threshold for the split.
        :param Ω_t: Task-specific feature set.
        :param Ω_G: Global feature set.
        :return: The calculated loss (sum of squared error and penalties).
        """
        left_size, right_size, missing_size = len(left_y), len(right_y), len(missing_y)
        left_mean, right_mean, missing_mean = np.mean(left_y), np.mean(right_y), np.mean(missing_y)

        # Calculate sum of squared error (SSE) for the split
        sse_left = np.sum((left_y - left_mean) ** 2)
        sse_right = np.sum((right_y - right_mean) ** 2)
        see_missing = np.sum((missing_y - missing_mean) ** 2)
        
        # Calculate penalties
        penalty_g = self.mu_g * (1 if feature not in Ω_G else 0)
        penalty_t = self.mu_t * (1 if feature not in Ω_t else 0)

        # Return total loss (SSE + penalties)
        return (sse_left + sse_right + see_missing) + penalty_g + penalty_t

    def predict(self, X):
        """
        Make predictions for input features X based on the fitted tree.

        :param X: Feature matrix (n_samples x n_features).
        :return: Predictions (n_samples,).
        """
        X = X.values
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, node):
        """
        Predict the value for a single sample.

        :param x: Single sample (n_features,).
        :param node: Current tree node.
        :return: Predicted value.
        """
        if "leaf" in node:
            return node["leaf"]

        if x[node["feature"]] == -1:
            return self._predict_single(x, node["missing"])
        elif x[node["feature"]] <= node["threshold"]:
            return self._predict_single(x, node["left"])
        else:
            return self._predict_single(x, node["right"])
            
    def get_feature_importances(self):
        """
        Return the feature importances from the features selected during the tree construction.

        :return: 
        """
        return self.feature_importances_
       
    def get_selected_features(self):
        """
        Return the features selected during tree construction.

        :return: A set of selected feature indices.
        """
        return self.selected_features


class MultitaskAGBM:
    def __init__(self, shrinkage, iterations, alpha, mu_g, mu_t):
        """
        Initialize parameters for Multitask GBM

        :param shrinkage: Learning rate for gradient boosting.
        :param iterations: Number of boosting iterations.
        :param alpha: Tree growth parameter.
        :param mu_g: Group penalty parameter.
        :param mu_t: Individual task penalty parameter.
        """
        assert mu_g + mu_t < 1, "mu_g + mu_t must be less than 1."
        self.shrinkage = shrinkage
        self.iterations = iterations
        self.alpha = alpha
        self.mu_g = mu_g
        self.mu_t = mu_t
        self.models = []  # Store models for each task
        
    def extract_tasks(self,features_file, outcome_file, one_hot_columns):
        """
        Extract tasks from one-hot encoded features and combine with outcomes.

        :param features_file: Path to the CSV file containing feature data.
        :param outcome_file: Path to the CSV file containing outcome data.
        :param one_hot_columns: List of one-hot encoded drug feature columns.
        :return: Combined feature matrix (X), outcome vector (y), and task vector.
        """
        # Load feature and outcome data
        features_df = pd.read_csv(features_file)
        outcomes_df = pd.read_csv(outcome_file)
        features_df = features_df.loc[:, ~features_df.columns.str.contains('^Unnamed')]
        # Extract task names from one-hot encoded columns
        tasks = features_df[one_hot_columns].idxmax(axis=1)  # Get the column name with max value (1)
        # Convert outcomes to 1D array if necessary
        y = outcomes_df.squeeze()  # Ensures it's a 1D Series
        y = y.astype(int)
        y = y - 1  # Convert labels 1, 2 to 0, 1
        return features_df, y, tasks  
        

    def fit(self, X, y, tasks, n_splits):
        """
        Fit Multitask GBM to the data.

        :param X: Input features (array-like of shape [n_samples, n_features]).
        :param y: Target values (array-like of shape [n_samples]).
        :param tasks: Task identifiers (array-like of shape [n_samples]).
        :return: Predictions H_t, task-specific feature sets Ω_t, global feature set Ω_G.
        """
        T = len(set(tasks))  # Number of tasks
        task_data = {t: (X[tasks == t], y[tasks == t]) for t in set(tasks)}
        H_t = {t: np.zeros(len(task_data[t][0])) for t in task_data.keys()}  # Initialize predictions
        H_t_test = {t: [] for t in task_data.keys()}  # Store test set predictions
        H_t_test_indices = {t: [] for t in task_data.keys()}  # Store test set indices
        g_t = {t: task_data[t][1] - H_t[t] for t in task_data.keys()}  # Initialize residues
        Ω_t = {t: {} for t in task_data.keys()}  # Task-specific feature sets
        Ω_G = {}  # Global feature set
        
        # Create StratifiedKFold splits for each task
        skf_splits = {
            t: list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(task_data[t][0], task_data[t][1]))
            for t in task_data.keys()
        }
        # boosting and tree fitting part (k = no. of boosting and fitting iterations, t = no. of tasks)
        for k in range(self.iterations):
            for t in task_data.keys():
                # task specific data!
                X_t, y_t = task_data[t]
                # Perform cross-validation for this task
                for fold_idx, (train_idx, test_idx) in enumerate(skf_splits[t]):
                    X_train, X_test = X_t.iloc[train_idx], X_t.iloc[test_idx]
                    y_train, y_test = y_t.iloc[train_idx], y_t.iloc[test_idx]

                    
                    p_t = expit(H_t[t][train_idx]) # transform categorical label to probability?
                    g_t[t] = y_train - p_t  # Update residues
                    # Initialize and fit the tree
                    tree = CustomCART(max_depth=5, shrinkage=0.1, mu_g=self.mu_g, mu_t=self.mu_t)
                    tree.fit(X_train, g_t[t],  Ω_t[t], Ω_G)
                    selected_features = tree.get_selected_features()
                    feature_importances = tree.get_feature_importances()
                     # Update Ω_t and Ω_G with feature importances
                    for feature, importance in feature_importances.items():
                        # Update task-specific feature set
                        Ω_t[t][feature] = Ω_t[t].get(feature, 0) + importance
                        # Update global feature set
                        Ω_G[feature] = Ω_G.get(feature, 0) + importance
  
                    # Make predictions
                    predictions = tree.predict(X)
                    #H_t[t][train_idx] += self.shrinkage * tree.predict(X_train)
                    
                    test_predictions = self.shrinkage * tree.predict(X_test)
                    H_t_test[t].append(test_predictions)
                    H_t_test_indices[t].append(test_idx)
                    #Ω_t[t].update(selected_features)  # Update task-specific feature set
                    #Ω_G.update(selected_features)  # Update global feature set
                    
            # replace feature set indices with column names          
        #Ω_G = {X.columns[i]: importance for i, importance in Ω_G.items()}
        #Ω_t = {t: {X.columns[i]: importance for i, importance in features.items()}}
        # replace feature set indices with column names          
        Ω_G = {X.columns[i]: importance for i, importance in Ω_G.items()}
        Ω_t = {t: {X.columns[i]: importance for i, importance in Ω_t[t].items()} for t in Ω_t.keys()}

        return H_t_test, H_t_test_indices, Ω_t, Ω_G


# Example Usage
if __name__ == "__main__":
    feature_file = '/home/julia.hellmig/reports/epipgx/complete/data/genecounts/log_regression_learning_data_filtered_alldrugs.csv'
    outcome_file = '/home/julia.hellmig/reports/epipgx/complete/data/genecounts/log_regression_outcomes_filtered_alldrugs.csv'
    # Specify one-hot encoded drug columns
    '''one_hot_columns = ['D120_AED_GENERIC_ABBREV_CBZ','D120_AED_GENERIC_ABBREV_CLB',
                       'D120_AED_GENERIC_ABBREV_ESM','D120_AED_GENERIC_ABBREV_GBP',
                       'D120_AED_GENERIC_ABBREV_LEV','D120_AED_GENERIC_ABBREV_LTG','D120_AED_GENERIC_ABBREV_PB',
                       'D120_AED_GENERIC_ABBREV_PHT','D120_AED_GENERIC_ABBREV_PRM','D120_AED_GENERIC_ABBREV_TPM',
                       'D120_AED_GENERIC_ABBREV_VPA','D120_AED_GENERIC_ABBREV_ZNS']
    '''
    one_hot_columns = ['D120_AED_GENERIC_ABBREV_CBZ',
                       'D120_AED_GENERIC_ABBREV_LEV','D120_AED_GENERIC_ABBREV_LTG','D120_AED_GENERIC_ABBREV_TPM',
                       'D120_AED_GENERIC_ABBREV_VPA']
    mu_gs = [0.01,0.1]
    mu_ts = [0.1,0.5]
    for mu_g in mu_gs:
        for mu_t in mu_ts:
            if mu_g + mu_t >= 1:
                continue
            else:
                model = MultitaskAGBM(shrinkage=0.1, iterations=20, alpha=0.02, mu_g=mu_g, mu_t=mu_t)
                X, y, tasks = model.extract_tasks(feature_file, outcome_file, one_hot_columns)
                H_t_test, H_t_test_indices, Ω_t, Ω_G = model.fit(X, y, tasks, n_splits=5)
                print(mu_g,mu_t)
                #print("Predictions for each task:", H_t_test)
                print("Task-specific feature sets:", Ω_t)
                print("Global feature set:", Ω_G)

            # Evaluate MCC and AUC
            mcc_scores = {}
            auc_scores = {}
            auc_pr_scores = {}
            for t in H_t_test.keys():
                # Combine test set predictions and true labels for all folds
                y_true_test = np.concatenate([y[tasks == t].iloc[test_idx].values for test_idx in H_t_test_indices[t]])
                y_pred_test = np.concatenate(H_t_test[t])

                # Convert logits to probabilities
                y_pred_test_prob = expit(y_pred_test)
                y_pred_test_bin = (y_pred_test_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

                # Calculate MCC for the task
                mcc_scores[t] = calculate_mcc(y_true_test, y_pred_test_bin)
                auc_scores[t] = calculate_auc(y_true_test, y_pred_test_bin)
                auc_pr_scores[t] = calculate_auc_pr(y_true_test, y_pred_test_bin)
            print("MCC Scores for each task:", mcc_scores)
            print("AUC Scores for each task:", auc_scores)
            print("AUC PR Scores for each task:", auc_pr_scores)
