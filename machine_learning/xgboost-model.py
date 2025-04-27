import os
import json
import pathlib
import logging
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from utils import load_csv_data

###################  CONFIGURATIONS  ###################
# Logging
logger = logging.getLogger('svm_rbf_optimizer')

# Reproducibility
SEED = 21
np.random.seed(SEED)

# Paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR.parent / "data_analysis" / "output_data"
RESULTS_DIR = CURRENT_DIR / "trials-xgboost"
os.makedirs(RESULTS_DIR, exist_ok=True)

###################  LOAD DATA  ###################
# Load preprocessed dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data(DATA_DIR)

# Apply standardization using training statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Ensure label shape consistency
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Convert to XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Optimization parameters
N_TRIALS = 500
SAMPLER_SEED = 12

###################  OPTIMIZATION OBJECTIVE  ###################

# Store the results
RESULTS_LIST = []

def objective(trial):
    # Define the hyperparameters to tune
    param = {
        # Suppresses training logs
        "verbosity": 0,
        # Binary classification with logistic regression
        "objective": "binary:logistic",
        # Define booster: 'gbtree' uses decision trees, 'gblinear' uses linear models, 'dart' is Dropout Additive Regression Trees.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # Sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # Sampling features when building each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        # Number of boosting rounds
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        # GPU acceleration for training
        "device" : "cuda",
        # Histogram-based approximate algorithm
        "tree_method" : "hist",
        # Reproducibility
        "seed": SEED}
    
    # Additional params for tree-based booster
    if param["booster"] in ["gbtree", "dart"]:
        param.update({
            # Maximum depth of the tree.
            "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
            # Minimum child weight, the larger the more conservative is the tree.    
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            # Learning rate
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            # Minimum loss reduction for further partition.
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            # Tree growth policy: 'depthwise' splits the tree layer-by-layer, 'lossguide': splits nodes with highest loss change first.
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        })

    # Additional params for DART booster
    if param["booster"] == "dart":
        param.update({
            # Dropout type: 'uniform' dropout trees uniformly, 'weighted': dropout trees with probability proportional to weight.
            "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
            # Controls normalization of dropped trees’ predictions.
            "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),
            # Probability of dropping trees during training.
            "rate_drop": trial.suggest_float("rate_drop", 1e-8, 1.0, log=True),
            # Probability of skipping dropout during a boosting round.
            "skip_drop": trial.suggest_float("skip_drop", 1e-8, 1.0, log=True),
        })
    
    # Train model
    bst = xgb.train(param, dtrain)
    
    # Predictions
    val_preds = np.rint(bst.predict(dvalid))
    test_preds = np.rint(bst.predict(dtest))
    
    # Metrics
    accuracy_val = accuracy_score(y_val, val_preds)
    f1_val = f1_score(y_val, val_preds, average="macro")
    accuracy_test = accuracy_score(y_test, test_preds)
    f1_test = f1_score(y_test, test_preds, average="macro")
    
    # Store results
    RESULTS_LIST.append({
        "Trial": trial.number,
        "Validation_Acc": accuracy_val,
        "F1_validation": f1_val,
        "Test_Acc": accuracy_test,
        "F1_test": f1_test,
        "Parameters": trial.params
    })

    return accuracy_val

################### MAIN EXECUTION ###################
if __name__ == "__main__":
    
    # Set up Optuna study with TPE Sampler
    sampler = TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
        
    # Save all results
    pd.DataFrame(RESULTS_LIST).to_csv(RESULTS_DIR / "results.csv", index=False)
    
    # Save best trial
    best_trial = study.best_trial
    pd.DataFrame([r for r in RESULTS_LIST if r['Trial'] == best_trial.number]).to_csv(RESULTS_DIR / "best_trial.csv", index=False)

    # Save config
    with open(RESULTS_DIR / "experiment_details.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_trials": N_TRIALS,
            "sampler_seed": SAMPLER_SEED,
            "direction": "maximize",
            "goal": "validation_accuracy"
        }, f, indent=4)

    print(f"Best trial: {best_trial.number}, Accuracy: {best_trial.value:.4f}, Parameters: {best_trial.params}")
