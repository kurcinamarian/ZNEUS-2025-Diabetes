import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def train_model(model, X_train, y_train, X_val, y_val, device, epochs=50, lr=0.001, batch_size=128, pos_weight=None, verbose=False, log_wandb=False, patience=None, min_delta=0.0):

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    # Early stopping tracking
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size()[0])
        running_train_loss = 0.0
        batches = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())
            batches += 1

        epoch_train_loss = running_train_loss / max(batches, 1)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            epoch_val_loss = float(val_loss.item())
            val_losses.append(epoch_val_loss)
        
        # Early stopping check
        if patience is not None:
            if epoch_val_loss < best_val_loss - min_delta:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
        
        if log_wandb:
            try:
                import wandb
                wandb.log({
                    "train/loss": epoch_train_loss,
                    "val/loss": epoch_val_loss,
                    "epoch": epoch + 1,
                    "lr": lr,
                    "batch_size": batch_size,
                })
            except Exception:
                pass
        
        if verbose and (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    
    # Restore best model if early stopping was used
    if patience is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Restored model to best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        y_true = y.cpu().numpy().ravel()
        y_pred = preds.cpu().numpy().ravel()
        y_probs = probs.cpu().numpy().ravel()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Curves
        fpr, tpr, roc_thr = roc_curve(y_true, y_probs)
        prec_curve, rec_curve, pr_thr = precision_recall_curve(y_true, y_probs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thr.tolist()},
        "pr_curve": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist(), "thresholds": pr_thr.tolist()},
    }

