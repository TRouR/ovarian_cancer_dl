import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import f1_score

# ============================================================
# Data loading and preprocessing utilities
# ============================================================

def load_csv_data(data_dir, seed=16):
    '''
    Loads expression (X) and trait (y) data from CSV files,
    performs stratified train-validation-test split.

    :param data_dir: Path to the directory containing 'exprs.csv' and 'traits.csv'.
    :param seed: Random seed for reproducibility.
    :return: Tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    '''
    np.random.seed(seed)
    
    # Load features (X) and target (y)
    exprs_path = os.path.join(data_dir, "exprs.csv")
    traits_path = os.path.join(data_dir, "traits.csv")
    
    X = pd.read_csv(exprs_path, index_col=0).to_numpy(dtype=np.float32)
    y = pd.read_csv(traits_path, usecols=["Stage"])
    
    # Encode labels: 'low' → 0, 'high' → 1
    y.replace(['low', 'high'], [0, 1], inplace=True)
    y = y.to_numpy(dtype=np.float32).flatten()
    
    # Stratified split data into Training, Validation, Test sets   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=41, shuffle=True, stratify=y) 
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state=41, shuffle=True, stratify=y_test)
    
    # Balance data by transferring samples from Validation/Test to Training
    X_train, y_train, X_val, y_val = _balance_sample(X_train, y_train, X_val, y_val, label=0, set_name="Validation")
    X_train, y_train, X_test, y_test = _balance_sample(X_train, y_train, X_test, y_test, label=1, set_name="Test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def _balance_sample(X_train, y_train, X_other, y_other, label, num_samples=1, set_name="Validation"):
    '''
    Moves a number of randomly selected samples of a specific class from `X_other` (Validation or Test)
    to `X_train` to ensure class balance.

    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param X_other: Validation/Test feature set.
    :param y_other: Validation/Test labels.
    :param label: Class label (0 or 1) to balance.
    :param num_samples: Number of samples to move (default: 1).
    :param set_name: Name of the set being modified ('Validation' or 'Test').
    :return: Updated X_train, y_train, X_other, y_other.
    '''
    indices = np.where(y_other == label)[0]

    # Ensure we do not remove more samples than available
    num_samples = min(num_samples, len(indices))

    if num_samples == 0:
        print(f"No samples of class {label} found in {set_name}.")
        return X_train, y_train, X_other, y_other

    # Randomly select 'num_samples' indices
    selected_indices = np.random.choice(indices, size=num_samples, replace=False)

    # Extract selected samples
    X_selected, y_selected = X_other[selected_indices], y_other[selected_indices]

    # Remove selected samples from X_other and y_other
    X_other = np.delete(X_other, selected_indices, axis=0)
    y_other = np.delete(y_other, selected_indices, axis=0)

    # Append selected samples to training set
    X_train = np.vstack([X_train, X_selected])
    y_train = np.append(y_train, y_selected)

    print(f"Moved {num_samples} sample(s) of class {label} from {set_name} to Training set.")

    return X_train, y_train, X_other, y_other

# ============================================================
# Learning rate scheduler
# ============================================================

def define_scheduler(epochs, optimizer, params):
    """
    Defines learning rate scheduler based on hyperparameters.
    :param epochs: Number of training epochs.
    :param optimizer: Optimizer instance.
    :param params: Dictionary of parameters.
    :return: Scheduler object.
    """
    if params['scheduler'] == 'exp':
        return lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'], last_epoch=-1)
    elif params['scheduler'] == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=params['gamma'])
    elif params['scheduler'] == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif params['scheduler'] == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    else:
        raise NotImplementedError(f"Learning rate scheduler '{params['scheduler']}' is not implemented.")

# ============================================================
# Activation Functions
# ============================================================

def activation_fun(act=None, negative_slope=0.2, alpha=1.0):
    """
    Utility to select activation functions.

    :param act: Activation function name.
    :return: Corresponding PyTorch activation function.
    """
    act = act.lower() if act else 'none'
    
    activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'elu': nn.ELU(alpha=alpha),
    'leakyrelu': nn.LeakyReLU(negative_slope),
    'sigmoid': nn.Sigmoid(),
    'lsm': nn.LogSoftmax(dim=1),
    "none": nn.Identity()
    }
    
    if act not in activations:
        raise NotImplementedError(f"Activation function '{act}' is not found.")
    return activations[act]

# ============================================================
# Training & Evaluation Utilities
# ============================================================
    
def nn_train_val(model, train_loader, val_loader, optimizer, criterion, epoch, l1, use_scheduler=False, scheduler=None):
    """
    Trains and validates the neural network.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        epoch (int): Current epoch number.
        l1 (float): L1 regularization strength.
        use_scheduler (bool): Whether to use learning rate scheduler.
        scheduler: Scheduler object.

    Returns:
        Training loss, training accuracy, validation loss, validation accuracy, training F1, validation F1
    """
    model.train()
    total_loss, correct = 0.0, 0
    all_tr_preds, all_tr_labels = [], []

    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()

        batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda()
        
        _, outputs = model(batch_features)

        loss = criterion(outputs, batch_labels) + l1 * define_reg(model)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_labels).sum().item()

        all_tr_preds.extend(preds.detach().cpu().numpy())
        all_tr_labels.extend(batch_labels.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

    if use_scheduler:
        scheduler.step()

    train_accuracy = correct * 100 / len(all_tr_labels)
    train_f1 = f1_score(all_tr_labels, all_tr_preds, average='macro')

    # Validation
    val_loss, val_correct = 0.0, 0
    all_val_preds, all_val_labels = [], []

    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.cuda(), labels.cuda()
            _, outputs = model(features)

            val_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_accuracy = val_correct * 100 / len(all_val_labels)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

    return total_loss, train_accuracy, val_loss, val_accuracy, train_f1, val_f1

def nn_test(model, test_loader, label_dim, comp_grad=False, input_dim=None, output_dir=None, set_name=None):
    """
    Evaluates the model on the test dataset.
    Optionally computes gradients w.r.t. input features and internal layers for interpretability.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        label_dim: Number of classes.
        comp_grad (bool): If True, compute input and layer-wise gradients.
        input_dim (int): Input feature dimension (needed if comp_grad=True).
        output_dir (Path): Output directory to save layer-wise importance CSVs.
        set_name (str): Dataset name ('val' or 'test'), used in filenames.

    Returns:
        test_accuracy (float): Accuracy on test set.
        test_f1 (float): Macro F1-score on test set.
        gradients (torch.Tensor or None): Gradients w.r.t. input features (if comp_grad=True).
        all_test_labels (np.ndarray): True labels for the test set.
    """
    model.eval()  # Evaluation mode

    all_preds = []
    all_labels = []
    gradients = [] if comp_grad else None

    total_correct = 0
    total_samples = 0

    # Prepare hooks for per-layer gradients
    if comp_grad:
        activations_gradients = {}

        def register_hooks(model):
            hooks = []

            def hook_fn(layer_name):
                def hook(module, grad_input, grad_output):
                    if grad_input[0] is not None:
                        if layer_name not in activations_gradients:
                            activations_gradients[layer_name] = []
                        activations_gradients[layer_name].append(grad_input[0].detach().abs().cpu().numpy())
                return hook

            for idx, layer in enumerate(model.encoder):
                name = f'Layer_{idx + 1}'
                handle = layer.register_backward_hook(hook_fn(name))
                hooks.append(handle)

            return hooks

        hooks = register_hooks(model)

    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.cuda()
        batch_labels = batch_labels.cuda()

        # Forward pass
        input_features, outputs = model(batch_features)

        # Collect true labels and predictions
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())

        # Update accuracy counter
        total_correct += (preds == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

        if comp_grad:
            one_hot_labels = torch.zeros(batch_labels.size(0), label_dim).cuda().scatter_(1, batch_labels.unsqueeze(1), 1)
            y_c = torch.sum(one_hot_labels * outputs)

            batch_gradients = torch.autograd.grad(y_c, input_features, retain_graph=False, create_graph=False)[0]
            gradients.append(batch_gradients.detach().cpu())

    # After loop cleanup
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    test_accuracy = total_correct * 100 / total_samples
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    # Prepare final gradients
    gradients = torch.cat(gradients, dim=0) if comp_grad else None

    # === Process layer-wise gradients ===
    if comp_grad and output_dir and set_name:
        save_layerwise_gradients(activations_gradients, output_dir, set_name)

        # Free memory (optional)
        activations_gradients.clear()

    # Remove hooks
    if comp_grad:
        for h in hooks:
            h.remove()

    return test_accuracy, test_f1, gradients, all_labels

def define_reg(model):
    """
    Compute L1 regularization term.

    :param model: PyTorch model.
    :return: L1 regularization loss.
    """
    return sum(torch.abs(W).sum() for W in model.parameters())

# ============================================================
# Feature Importance Utilities
# ============================================================

def nn_gradients_each_class(gradients, labels, num_classes, input_dim, set_name=None, dir=None):
    """
    Compute average feature importance per class based on gradients.

    Args:
        gradients (np.array): Gradients computed from the model.
        labels (np.array): Ground truth class labels.
        num_classes (int): Number of output classes.
        input_dim (int): Input feature dimension.
        set_name (str): Dataset identifier ('val' or 'test').
        dir (Path): Output directory.

    Outputs:
        CSV file containing feature importance per class.
    """
    overall_importance = []

    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        class_grads = gradients[class_indices]
        class_importance = np.mean(class_grads, axis=0).reshape(-1, 1)
        normalized = normalize(class_importance, axis=0, norm='max')
        overall_importance.append(normalized)

    overall_mean = np.mean(overall_importance, axis=0)
    all_classes_importance = np.hstack(overall_importance + [overall_mean])

    # Save to CSV
    header = [f'class_{i}' for i in range(num_classes)] + ['all_classes']
    filename = f"{set_name}_all_layers_feature_importance.csv"
    df = pd.DataFrame(all_classes_importance, columns=header)
    df.to_csv(os.path.join(dir, filename), index=False)

def save_layerwise_gradients(activations_gradients, output_dir, set_name):
    """
    Save averaged layer-wise gradients to CSV.

    Args:
        activations_gradients (dict): Layer-wise gradients collected by hooks.
        input_dim (int): Number of input features.
        output_dir (Path): Output directory.
        set_name (str): Dataset identifier ('val' or 'test').
    """
    records = []
    for layer_name, grads in activations_gradients.items():
        grads_array = np.vstack(grads)  # Shape: (samples, input_dim)
        avg_importance = np.mean(grads_array, axis=0)
        normalized_importance = avg_importance / (np.max(avg_importance) + 1e-8)
        records.append(normalized_importance)

    df = pd.DataFrame(np.array(records).T, columns=activations_gradients.keys())
    filename = output_dir / f"{set_name}_layerwise_importance.csv"
    df.to_csv(filename, index=False)
    print(f"Layer-wise importance saved to: {filename}")
        
def plot_layerwise_importance(csv_path, title="Layer-wise Feature Importance", save_path=None):
    """
    Plot heatmap of layer-wise feature importance.

    Args:
        csv_path (Path or str): Path to the CSV file with importance values.
        title (str): Title of the plot.
        save_path (Path or str, optional): If provided, saves the plot to this path.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Plot
    plt.figure(figsize=(12, max(6, df.shape[0] // 10)))  # Dynamically adjust size

    sns.heatmap(df, cmap="YlGnBu", linewidths=0.5, linecolor='grey')

    plt.title(title, fontsize=16)
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Input Features", fontsize=14)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

    plt.close()
   

    