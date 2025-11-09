
import torch
import yaml
import wandb

from pathlib import Path

from project.dataloader import DataLoader
from project.experiment import ExperimentManager


def load_config(config_path, n_features):
    path = Path(config_path)
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
        
    cfg["n_input"] = n_features
    
    cfg.setdefault("name", path.stem)
    return cfg


def run (config_path, project, entity = None):
    loader = DataLoader()
    X_train, X_test, y_train, y_test, n_features = loader.load_and_preprocess()
    
    cfg = load_config(config_path, n_features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(
        project = project,
        entity=entity,
        name=cfg["name"],
        config=cfg,
        reinit=True
    )
    
    manager = ExperimentManager()
    
    result = manager.run_experiment(
        X_train, y_train, X_test, y_test, config=cfg, device=device, log_wandb=True
    )
    
    wandb.log({
        "test/accuracy": result["accuracy"],
        "test/precision": result["precision"],
        "test/sensitivity": result["sensitivity"],
        "test/specificity": result["specificity"],
        "test/f1": result["f1"],
        "test/roc_auc": result["roc_auc"],
        "training_time_sec": result["training_time_sec"],
        "hyperparams/epochs": result["epochs"],
        "hyperparams/lr": result["lr"],
        "hyperparams/batch_size": result["batch_size"],
    })

    cm = result.get("confusion_matrix")
    if cm is not None:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = np.array(cm)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {cfg['name']}")
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()
        except Exception as e:
            print(f"Warning: failed to log confusion matrix image: {e}")

    # Log ROC curve image if available in results
    roc = result.get("roc_curve")
    if roc is not None and isinstance(roc, dict):
        try:
            import matplotlib.pyplot as plt

            fpr = roc.get("fpr", [])
            tpr = roc.get("tpr", [])
            if len(fpr) > 0 and len(tpr) > 0:
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, label=f"AUC = {result.get('roc_auc', 0):.3f}")
                plt.plot([0, 1], [0, 1], 'k--', linewidth=0.75)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {cfg['name']}")
                plt.legend(loc="lower right")
                wandb.log({"roc_curve": wandb.Image(plt)})
                plt.close()
        except Exception as e:
            print(f"Warning: failed to log ROC curve image: {e}")

    wandb.finish()
    
    print(f"\nExperiment: {result['experiment']}")
    for k in ["roc_auc", "sensitivity", "specificity", "accuracy", "precision", "f1"]:
        print(f"{k}: {result[k]:.4f}")
        
run(".scratch/experiment_configs/11.yaml", "zneus-project-1", entity="ZNEUS-Diabetes")
