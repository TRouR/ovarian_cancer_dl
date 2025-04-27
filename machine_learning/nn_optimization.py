import sys
import os
import pathlib
import optuna
import torch
import numpy as np
import pandas as pd
import glob
import time
import json
import logging
from utils import load_csv_data, define_scheduler, nn_train_val, nn_test, nn_gradients_each_class, plot_layerwise_importance
from sklearn.preprocessing import StandardScaler
from nn_model import FCNN
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset
from optuna.trial import Trial

# ============================ CONFIGURATIONS ============================

# Logging setup (optional, you can expand to file logging)
logger = logging.getLogger('nn_optimizer')

# Reproducibility
SEED = 21
np.random.seed(SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)

# Paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = CURRENT_DIR.parent / "data_analysis" / "output_data"
RESULTS_DIR = CURRENT_DIR / "trials-nn"
os.makedirs(RESULTS_DIR, exist_ok=True)  

print(f"CURRENT_DIR: {CURRENT_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"Does CURRENT_DIR exist? {CURRENT_DIR.exists()}")
print(f"Does RESULTS_DIR exist? {RESULTS_DIR.exists()}")



# Optimization parameters
NUM_EPOCHS = 3
PATIENCE = 7
N_TRIALS = 3
SAMPLER_SEED = 12

# ============================ DATA LOADING ============================

# Load preprocessed dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_csv_data(DATA_DIR)

# Standardize features using training statistics
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Prepare tensors
X_train = torch.FloatTensor(X_train).requires_grad_()
y_train = torch.LongTensor(y_train)

X_val = torch.FloatTensor(X_val).requires_grad_()
y_val = torch.LongTensor(y_val)

X_test = torch.FloatTensor(X_test).requires_grad_()
y_test = torch.LongTensor(y_test)

input_dim = X_train.shape[1]
label_dim = 2

results_list = []    
# ============================ HYPERPARAMETER SPACE ============================

def HypeParameterSpace(trial: Trial):
    # Define the hyperparameters to tune    
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "dim_1": trial.suggest_int("dim_1", 8, 256),
        "activation": trial.suggest_categorical("activation", ["ReLU", "Elu", "Tanh", "Sigmoid", "LeakyReLU"]),
        "normalization": trial.suggest_categorical("normalization", ["batch", "layer", "none"]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "dropout_type": trial.suggest_categorical("dropout_type", ["simple_drop", "alpha_drop"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        "l1": trial.suggest_float("l1", 1e-5, 1e-1, log=True),
    }

    # Layer dimensions for deeper networks
    if params["num_layers"] > 1:
        for i in range(2, params["num_layers"] + 1):
            params[f"dim_{i}"] = trial.suggest_int(f"dim_{i}", 8, 256)

    # Scheduler parameters
    if params['use_scheduler']:
        params['scheduler'] = trial.suggest_categorical('scheduler', ['exp', 'step'])
        if params['scheduler'] == 'exp':
            params['gamma'] = trial.suggest_float('gamma', 0.95, 0.95)  # Fixed gamma for stability
        else:
            params['step'] = trial.suggest_int('step', 3, 6, step=1)
            params['gamma'] = trial.suggest_float('gamma', 0.1, 0.1)

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
    batch_size = params['batch_size']
    l1_penalty = params['l1']
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initiate the model 
    model = FCNN(input_dim=input_dim, label_dim=label_dim, params=params).cuda()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.999), weight_decay=params['weight_decay'])
    
    # Define the scheduler
    if params['use_scheduler']:
        scheduler = define_scheduler(epochs=NUM_EPOCHS, optimizer=optimizer, params=params)
    else:
        scheduler = None
        
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    step = 0 # Early stopping counter
    best_val_acc = 0
    
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        
        train_loss, train_acc, val_loss, val_acc, train_f1, val_f1 = nn_train_val(
            model, train_loader, val_loader, optimizer, criterion, epoch,
            l1_penalty, params['use_scheduler'], scheduler
        )
                
        test_acc, test_f1, _, _ = nn_test(model, test_loader, label_dim, comp_grad=False)
        
        # Save model checkpoint if validation improves
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            step = 0
            # Save the current best model
            try:
                torch.save(model.state_dict(), RESULTS_DIR / f"model_trial_{trial.number}_best_epoch_{best_epoch}.pkl")
            except Exception as e:
                print(f"Saving test failed: {e}")
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

    # Report progress to Optuna
    trial.set_user_attr("best_epoch", best_epoch)
    trial.report(best_val_acc, step)
    
    elapsed_time = (time.time() - start_time) / 60
    total_elapsed_time = (time.time() - start_time_first_trial) / 60
    
    print(f"Trial {trial.number} completed in {elapsed_time:.2f} min, total elapsed: {total_elapsed_time:.2f} min.")

    # Optuna pruning 
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return best_val_acc

# ============================ OPTIMIZATION ============================

if __name__ == "__main__":
    sampler = TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_DIR / "results.csv", index=False)

    best_trial = study.best_trial
    best_trial_epoch = best_trial.user_attrs["best_epoch"]

    best_trial_df = results_df[(results_df['Trial'] == best_trial.number) & (results_df['Epoch'] == best_trial_epoch)]
    best_trial_df.to_csv(RESULTS_DIR / "best_trial.csv", index=False)

    # Load best model
    best_model = FCNN(input_dim=input_dim, label_dim=label_dim, params=best_trial.params).cuda()
    best_model.load_state_dict(torch.load(RESULTS_DIR / f"model_trial_{best_trial.number}_best_epoch_{best_trial_epoch}.pkl"))

    # Evaluate best model (Validation and Test) and save gradients
    for name, loader in zip(["val", "test"],
                            [DataLoader(TensorDataset(X_val, y_val), batch_size=best_trial.params['batch_size']),
                             DataLoader(TensorDataset(X_test, y_test), batch_size=best_trial.params['batch_size'])]
                            ):
        
        acc, f1, gradients, gr_labels = nn_test(
                                                best_model, loader, label_dim, 
                                                comp_grad=True,
                                                input_dim=input_dim,
                                                output_dir=RESULTS_DIR,
                                                set_name=name
                                            )

        gradients = np.maximum(gradients.detach().cpu().numpy(), 0)
        nn_gradients_each_class(gradients, gr_labels, label_dim, input_dim, set_name=name, dir=RESULTS_DIR)

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

        
    

    
    
    

