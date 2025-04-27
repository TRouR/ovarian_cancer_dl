import os
import optuna
import pathlib
import json
import logging
import numpy as np
import pandas as pd

from utils import *
from optuna.samplers import TPESampler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

###################  CONFIGURATIONS  ###################
# Set up logging
logger = logging.getLogger('svm_rbf_optimizer')

# Set random seed for reproducibility
SEED = 21
np.random.seed(SEED)

# Define paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR.parent / "data_analysis" / "output_data"
RESULTS_DIR = CURRENT_DIR / "trials-svm"
os.makedirs(RESULTS_DIR, exist_ok=True)

###################  LOAD DATA  ###################
# Load preprocessed dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data(DATA_DIR)

# Apply standardization using training statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Optimization parameters
N_TRIALS = 500
SAMPLER_SEED = 12

###################  OPTIMIZATION OBJECTIVE  ###################

# Store the results
RESULTS_LIST = []

def objective(trial):
    """
    Objective function for hyperparameter tuning using Optuna.
    
    - Optimizes C (svc_c) and gamma (svc_gamma) for the RBF kernel SVM model.
    - Evaluates performance using validation accuracy.
    """
    
    svc_c = trial.suggest_float("svc_c", 1e-3, 1e3, log=True)    
    svc_gamma = trial.suggest_float("svc_gamma", 1e-3, 1e3, log=True)    
    
    # Initialize and train the model
    svm = SVC(kernel='rbf',C=svc_c, gamma=svc_gamma, random_state=42)
    svm.fit(X_train,y_train)

    # Predict on validation and test sets
    y_val_pred = svm.predict(X_val)
    y_test_pred = svm.predict(X_test)

    # Compute accuracy and F1-score
    accuracy_val = accuracy_score(y_val,y_val_pred)
    f1_val = f1_score(y_val,y_val_pred,average='macro')
    
    accuracy_test = accuracy_score(y_test,y_test_pred)
    f1_test = f1_score(y_test,y_test_pred,average='macro')
    
    # Store results
    results_dict = {
    'Trial': trial.number,
    'Validation_Acc': accuracy_val,
    'F1_validation' : f1_val,
    'Test_Acc': accuracy_test,
    'F1_test': f1_test,
    'Parameters': trial.params  # Store trial parameters
    }

    RESULTS_LIST.append(results_dict)
    
    return accuracy_val # Objective: maximize validation accuracy

################### MAIN EXECUTION ###################
if __name__ == "__main__":

    # Set up Optuna study with TPE Sampler
    sampler = TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Save results as DataFrame
    results_df = pd.DataFrame(RESULTS_LIST)
    results_df.to_csv(RESULTS_DIR / "results.csv", index=False)
    
    # Retrieve best trial
    best_trial = study.best_trial
    best_trial_df = results_df[results_df['Trial'] == best_trial.number]
    best_trial_df.to_csv(RESULTS_DIR / "best_trial.csv", index=False)

    # Save experimental details
    experiment_details = {
        'seed': SEED,
        'n_trials': N_TRIALS,
        'sampler_seed': 12,
        'direction': "maximize",
        'goal': "validation_accuracy"
    }
    
    with open(RESULTS_DIR / "experiment_details.json", 'w') as f:
        json.dump(experiment_details, f, indent=4)

    print(f"Best trial: {best_trial.number}, Accuracy: {best_trial.value:.4f}, Parameters: {best_trial.params}")    
