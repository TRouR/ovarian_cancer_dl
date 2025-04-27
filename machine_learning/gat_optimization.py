import sys
import os
import glob
import json
import time

import logging
import numpy as np
import pandas as pd
import torch
import optuna

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from optuna import Trial, create_study
from optuna.samplers import TPESampler
from gat_model import GAT
from utils_gat import gat_train_val, gat_test, optimized_gradients_each_class
from utils import load_csv_data, define_scheduler, plot_layerwise_importance

# ============================ CONFIGURATIONS ============================
# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Logging setup (optional, you can expand to file logging)
logger = logging.getLogger('gat_optimizer')

# Reproducibility
SEED = 21
np.random.seed(SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR.parent / "data_analysis" / "output_data"
RESULTS_DIR = CURRENT_DIR / "trials-gat"
os.makedirs(RESULTS_DIR, exist_ok=True)  

print(f"CURRENT_DIR: {CURRENT_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"Does CURRENT_DIR exist? {CURRENT_DIR.exists()}")
print(f"Does RESULTS_DIR exist? {RESULTS_DIR.exists()}")

# Optimization parameters
POWER = 6
NUM_EPOCHS = 2
BATCH_SIZE = 9
N_TRIALS = 2
PATIENCE = 7
INPUT_DIM = 1
LABEL_DIM = 2
SAMPLER_SEED = 12

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================ DATA LOADING ============================
logger.info("Loading datasets...")
X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data(DATA_DIR)

# Compute adjacency matrix on training data
from data_analysis.analysis_utils import compute_biweight_correlation
adj_matrix = compute_biweight_correlation(pd.DataFrame(X_train)).values ** POWER
NUM_FEATURES = adj_matrix.shape[0]
ADJ_TYPE = "unsigned"


# Standardize features using training statistics
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Prepare tensors
X_train = torch.FloatTensor(X_train).requires_grad_().reshape(-1,X_train.shape[1],1)
y_train = torch.LongTensor(y_train)

X_val = torch.FloatTensor(X_val).requires_grad_().reshape(-1,X_train.shape[1],1)
y_val = torch.LongTensor(y_val)

X_test = torch.FloatTensor(X_test).requires_grad_().reshape(-1,X_train.shape[1],1)
y_test = torch.LongTensor(y_test)

results_list = []

# ============================ HYPERPARAMETER SPACE ============================

def HypeParameterSpace(trial: Trial):
    # Define the hyperparameters to tune    
    params = {
        "adj_tresh": trial.suggest_float("adj_tresh", 1e-8, 1e-1, log=True),
        "num_gat_layers": trial.suggest_int("num_gat_layers", 1, 4),
        "nhids_1": trial.suggest_int("nhids_1", 2, 24),
        "nheads_1": trial.suggest_int("nheads_1", 1, 4),
        "num_fc_layers": trial.suggest_int("num_fc_layers", 1, 6),
        "fc_dim_1": trial.suggest_int("fc_dim_1", 8, 512),
        "act_fc": trial.suggest_categorical("act_fc", ["ReLU", "Elu"]),
        "act_gat_out": trial.suggest_categorical("act_gat_out", ["ReLU", "Elu"]),
        "act_attention": trial.suggest_categorical("act_attention", ["LeakyReLU"]),
        "dropout_gat": trial.suggest_float("dropout_gat", 0.1, 0.6),
        "dropout_fc": trial.suggest_float("dropout_fc", 0.1, 0.6),
        "dropout_att": trial.suggest_float("dropout_att", 0.1, 0.6),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        "which_layer": trial.suggest_categorical("which_layer", ["all", "final"]),
        "l1": trial.suggest_float("l1", 1e-5, 1e-1, log=True),
        "normalization": trial.suggest_categorical("normalization", ["batch", "layer", "none"])
    }

    for i in range(2, params["num_gat_layers"] + 1):
        params[f"nhids_{i}"] = trial.suggest_int(f"nhids_{i}", 2, 24)
        params[f"nheads_{i}"] = trial.suggest_int(f"nheads_{i}", 1, 4)
    for i in range(2, params["num_fc_layers"] + 1):
        params[f"fc_dim_{i}"] = trial.suggest_int(f"fc_dim_{i}", 8, 512)

    if params["act_attention"] == "LeakyReLU":
        params["negative_slope"] = trial.suggest_float("negative_slope", 0.05, 0.6, step=0.05)

    if params["use_scheduler"]:
        params["scheduler"] = trial.suggest_categorical("scheduler", ["exp", "step"])
        params["gamma"] = 0.95 if params["scheduler"] == "exp" else 0.1
        if params["scheduler"] == "step":
            params["step"] = trial.suggest_int("step", 3, 6)

    return params

# ============================ OBJECTIVE FUNCTION ============================
# Keep track of the start time of the first trial
start_time_first_trial = None
def objective(trial: Trial):
    global start_time_first_trial
    if start_time_first_trial is None:
        start_time_first_trial = time.time()
    
    start_time = time.time()
    
    params = HypeParameterSpace(trial)
    
    global adj
    adj = torch.LongTensor((adj_matrix > params["adj_tresh"]).astype(int))
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = GAT(INPUT_DIM, LABEL_DIM, NUM_FEATURES, params).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], betas=(0.9, 0.999), weight_decay=params["weight_decay"])
    
    # Define the scheduler
    if params['use_scheduler']:
        scheduler = define_scheduler(epochs=NUM_EPOCHS, optimizer=optimizer, params=params)
    else:
        scheduler = None
        
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_epoch = 0, 0
    step = 0

    for epoch in range(NUM_EPOCHS):
        
        train_loss, train_acc, val_loss, val_acc, train_f1, val_f1 = gat_train_val(
            model, train_loader, val_loader, optimizer, criterion, adj, epoch,
            params["which_layer"], params["l1"], use_scheduler=params["use_scheduler"], scheduler=scheduler
        )

        test_acc, test_f1, *_ = gat_test(model, test_loader, adj, params["which_layer"], LABEL_DIM, compute_grad=False)
        
        # Save model checkpoint if validation improves
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(best_epoch)
            step = 0
            # Save the current best model
            torch.save(model.state_dict(), RESULTS_DIR / f"model_trial_{trial.number}_best_epoch_{best_epoch}.pkl")
        else:
            step += 1
        if step >= PATIENCE:
            break

        # Store training results
        results_list.append({
            'Trial': trial.number,
            'Epoch': epoch + 1,
            'Training_Loss': train_loss,
            'Training_Acc': train_acc,
            'Validation_Loss': val_loss,
            'Validation_Acc': val_acc,
            'F1_validation': val_f1,
            'Test_Acc': test_acc,
            'F1_test': test_f1,
            'Parameters': trial.params
        })

    trial.set_user_attr("best_epoch", best_epoch)
    trial.report(best_val_acc, step)
    
    elapsed_time = (time.time() - start_time) / 60
    total_elapsed_time = (time.time() - start_time_first_trial) / 60
    print(f"Trial {trial.number} completed in {elapsed_time:.2f} min, total elapsed: {total_elapsed_time:.2f} min.")
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


    return best_val_acc

# ============================ OPTIMIZATION ============================

if __name__ == "__main__":
    # Ensure RESULTS_DIR exists inside objective
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    study = create_study(direction="maximize", sampler=TPESampler(seed=SAMPLER_SEED))

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_DIR / "results.csv", index=False)
    
    best_trial = study.best_trial
    best_trial_epoch = best_trial.user_attrs["best_epoch"]
    #logger.info(f"Best trial: {best_trial.number}, Epoch: {best_trial_epoch}, Params: {best_trial.params}")

    best_trial_df = results_df[(results_df['Trial'] == best_trial.number) & (results_df['Epoch'] == best_trial_epoch)]
    best_trial_df.to_csv(RESULTS_DIR / "best_trial.csv", index=False)
    
        
    # Load best model
    best_model = GAT(input_dim=INPUT_DIM, label_dim=LABEL_DIM, num_features=NUM_FEATURES, params=best_trial.params).cuda()
    best_model.load_state_dict(torch.load(RESULTS_DIR / f"model_trial_{best_trial.number}_best_epoch_{best_trial_epoch}.pkl"))
    
    # Evaluate best model (Validation and Test) and save gradients + attention coefficients
    for name, loader in zip(["val", "test"],
                            [DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE),
                            DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)]
                            ):
        
        acc, f1, gradients, gr_labels, attention_coef = gat_test(
            best_model, loader, adj, best_trial.params['which_layer'],
            LABEL_DIM, compute_grad=True, output_dir=RESULTS_DIR, set_name=name
        )

        gradients = np.maximum(gradients.numpy(), 0)
        optimized_gradients_each_class(
            gradients, gr_labels, LABEL_DIM, best_trial.params['num_gat_layers'],
            best_trial.params['which_layer'], set_name=name, output_dir=RESULTS_DIR
        )

        pd.DataFrame(attention_coef).to_csv(RESULTS_DIR / f"{name}_attention_coef.csv")

    # Save experiment details
    experiment_details = {
        'seed': SEED,
        'n_trials': N_TRIALS,
        'sampler_seed': SAMPLER_SEED,
        'direction': "maximize",
        'goal': "validation_accuracy",
        'validation_acc': acc,
        'validation_f1': f1,
        'test_acc': acc,
        'test_f1': f1
    }

    with open(RESULTS_DIR / "experiment_details.json", 'w') as f:
        json.dump(experiment_details, f, indent=4)

    # Clean up models, keeping only best
    for file in glob.glob(str(RESULTS_DIR / "*.pkl")):
        if f"model_trial_{best_trial.number}_best_epoch_{best_trial_epoch}" not in file:
            os.remove(file)

    print(f"Best trial: {best_trial.number} with accuracy: {best_trial.value:.4f}")

    for split in ["val", "test"]:
        plot_layerwise_importance(
            RESULTS_DIR / f"{split}_layerwise_importance.csv",
            title=f"{split.capitalize()} Set - Layer-wise Feature Importance",
            save_path=RESULTS_DIR / f"{split}_layerwise_importance.png"
        )

