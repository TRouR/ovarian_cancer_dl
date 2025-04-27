import sys
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from torch.nn.utils import clip_grad_norm_
from utils import define_reg

def gat_train_val(
    model, train_loader, val_loader, optimizer, criterion,
    adj_matrix, epoch, which_layer, l1,
    use_scheduler=False, scheduler=None,
    accum_iter=1, clip_grad=False, max_grad_norm=1.0
):
    """
    Train and validate the GAT model for one epoch.

    Args:
        model (nn.Module): GAT model.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion: Loss function.
        adj_matrix (Tensor): Adjacency matrix (binary).
        epoch (int): Epoch number.
        which_layer (str): 'all' or 'final' to control layer outputs.
        l1 (float): L1 regularization coefficient.
        use_scheduler (bool): Whether to use a learning rate scheduler.
        scheduler: Learning rate scheduler.
        accum_iter (int): Number of steps to accumulate gradients.
        clip_grad (bool): Whether to apply gradient clipping.
        max_grad_norm (float): Maximum norm for gradient clipping.
    
    Returns:
        Tuple[float, float, float, float, float, float]:
        train_loss, train_acc, val_loss, val_acc, train_f1, val_f1
    """
    model.train()
    adj_matrix = adj_matrix.cuda()
    total_loss, correct = 0.0, 0
    all_tr_preds, all_tr_labels = [], []

    optimizer.zero_grad()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.cuda()
        labels = labels.squeeze(-1).cuda()
        outputs = model(features, adj_matrix, labels, which_layer)[2]
        loss = criterion(outputs, labels) + l1 * define_reg(model)
        loss = loss / accum_iter  # Normalize for accumulation

        # Track loss before backward for accurate logging
        total_loss += loss.item()

        loss.backward()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        
        all_tr_preds.extend(preds.detach().cpu().numpy())
        all_tr_labels.extend(labels.detach().cpu().numpy())

        # Perform optimizer step every `accum_iter` batches or at last batch
        if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(train_loader):
            if clip_grad:
                clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    if use_scheduler and scheduler:
        scheduler.step()

    train_acc = correct * 100 / len(all_tr_labels)
    train_f1 = f1_score(all_tr_labels, all_tr_preds, average='macro')

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    all_val_preds, all_val_labels = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.cuda()
            labels = labels.squeeze(-1).cuda()

            outputs = model(features, adj_matrix, labels, which_layer)[2]
            val_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_acc = val_correct * 100 / len(all_val_labels)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')

    return total_loss, train_acc, val_loss, val_acc, train_f1, val_f1

def gat_test(model, test_loader, adj_matrix, which_layer, label_dim, compute_grad=False, output_dir=None, set_name=None):
    """
    Evaluate the GAT model on the test set with optional gradient-based interpretability.

    Args:
        model (nn.Module): Trained GAT model.
        test_loader (DataLoader): Test dataset loader.
        adj_matrix (Tensor): Adjacency matrix for the graph.
        which_layer (str): 'all' or 'final' output pooling.
        label_dim (int): Number of output classes.
        compute_grad (bool): Whether to compute input and per-layer gradients.
        output_dir (str or Path): Where to save attention coefficients and gradients.
        set_name (str): Label for output files, e.g., 'val' or 'test'.

    Returns:
        Tuple of (accuracy, macro F1, input gradients (optional), labels, averaged attention map)
    """
    model.eval()
    adj_matrix = adj_matrix.cuda()

    all_preds, all_labels, attention_all = [], [], []
    gradients = [] if compute_grad else None
    total_correct, total_samples = 0, 0

    if compute_grad:
        activations_gradients = {}

        def hook_fn(layer_name):
            def hook(module, grad_input, _):
                if grad_input[0] is not None:
                    if layer_name not in activations_gradients:
                        activations_gradients[layer_name] = []
                    activations_gradients[layer_name].append(grad_input[0].detach().abs().cpu().numpy())
            return hook

        # Register hooks for each FC layer in the encoder
        hooks = [layer.register_backward_hook(hook_fn(f'Layer_{i+1}')) for i, layer in enumerate(model.encoder)]

    for features, labels in test_loader:
        features = features.cuda()
        labels = labels.squeeze(-1).cuda()

        gat_feat, _, logits, attention = model(features, adj_matrix, labels, which_layer)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        attention_all.append(attention.detach().cpu().numpy())

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if compute_grad:
            one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
            y_c = (logits * one_hot).sum()
            grad_batch = torch.autograd.grad(y_c, gat_feat, retain_graph=False)[0]
            gradients.append(grad_batch.detach().cpu())

    # Combine results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    attention_all = np.mean(np.vstack(attention_all), axis=0).squeeze(-1)
    
    gradients = torch.cat(gradients, dim=0) if compute_grad else None

    acc = total_correct * 100 / total_samples
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Save outputs if needed
    if compute_grad and output_dir and set_name:
        save_layerwise_gradients(activations_gradients, output_dir, set_name)
    
        # Free memory (optional)
        activations_gradients.clear()
    
    # Always clean up hooks if gradients were computed
    if compute_grad:
        for h in hooks:
            h.remove()
            
    return acc, f1, gradients, all_labels, attention_all

def optimized_gradients_each_class(
    gradients: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    num_layers: int,
    which_layer: str,
    set_name: str = None,
    output_dir: str = None
) -> None:
    """
    Compute and export feature importance per class based on input gradients.
    
    Args:
        gradients (np.ndarray): Gradients of shape (samples, features).
        labels (np.ndarray): Class labels corresponding to each sample.
        num_classes (int): Number of unique classes.
        num_layers (int): Number of GAT layers.
        which_layer (str): 'all' or 'final' - determine layer aggregation.
        set_name (str): 'val' or 'test' for naming output files.
        output_dir (str): Path to save feature importance CSVs.
    """
    overall_importance = []

    # Compute normalized average gradient per class
    for class_label in range(num_classes):
        idx = labels == class_label
        class_grads = gradients[idx]
        avg_grad = np.mean(class_grads, axis=0)

        if which_layer == "all":
            reshaped = avg_grad.reshape(num_layers + 1, -1).T
        else:
            reshaped = avg_grad.reshape(1, -1).T

        normalized = normalize(reshaped, axis=0, norm='max')
        overall_importance.append(normalized[np.newaxis])

    # Compute average across all classes
    mean_importance = np.mean(np.vstack(overall_importance), axis=0, keepdims=True)
    all_class_importance = np.vstack(overall_importance + [mean_importance])
    avg_layer_importance = np.mean(all_class_importance, axis=-1, keepdims=True)
    feature_importance = np.concatenate([all_class_importance, avg_layer_importance], axis=-1)

    # Prepare headers
    headers = [f'class_{i}' for i in range(num_classes)] + ['all_classes']
    headers = np.array(headers).reshape(1, -1)

    if which_layer == 'all':
        layer_names = [f'layer{i}' for i in range(num_layers + 1)] + ['overall']
        for i, layer_name in enumerate(layer_names):
            layer_data = np.concatenate([headers, feature_importance[:, :, i].T], axis=0)
            filename = f"{set_name}_{layer_name}_feature_importance.csv"
            pd.DataFrame(layer_data).to_csv(f"{output_dir}/{filename}", header=False, index=False)
    else:
        layer_data = np.concatenate([headers, all_class_importance.squeeze(-1).T], axis=0)
        filename = f"{set_name}_all_layers_feature_importance.csv"
        pd.DataFrame(layer_data).to_csv(f"{output_dir}/{filename}", header=False, index=False)

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