import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml
import torch
from project.model import MultilayerPerceptron
from project.trainer import train_model, evaluate_model


class ExperimentManager:
    def __init__(self):
        self.results = []
    
    def run_experiment(self, X_train, y_train, X_test, y_test, config, device, log_wandb=False):
        cfg_dict = self._load_config_dict(config)
        
        exp_name = self._get_experiment_name(config, cfg_dict)
        
        epochs, lr, batch_size, verbose, patience, min_delta = self._get_training_params(cfg_dict)
        
        pos = int(y_train.sum().item()) if hasattr(y_train, "sum") else int(y_train.sum())
        total = int(y_train.numel()) if hasattr(y_train, "numel") else len(y_train)
        neg = max(total - pos, 0)
        pos_weight = None
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor(float(neg/pos), dtype=torch.float32).to(device)
        
        print(f"Running Experiment: {exp_name}")
        
        model = MultilayerPerceptron(config)
        
        start = time.time()
        train_losses, val_losses = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            device,
            epochs,
            lr,
            batch_size,
            pos_weight,
            verbose,
            log_wandb,
            patience=patience,
            min_delta=min_delta
        )
        elapsed = time.time() - start
        
        metrics = evaluate_model(model, X_test, y_test, device)
        
        if "confusion_matrix" in metrics and hasattr(metrics["confusion_matrix"], "tolist"):
            metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
            
        result = {
            "experiment": exp_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "training_time_sec": elapsed,
            **metrics,
        }
        
        self.results.append(result)
        return result
    
    
    def _load_config_dict(self, config):
        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.suffix.lower() in {".yaml", ".yml"}:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            raise ValueError(f"Unsupported config format: {path.suffix}. Expected .yaml or .yml")
        elif isinstance(config, dict):
            return config
        else:
            raise TypeError("Config must be a dict or a path to a .yaml/.yml file")
        
    def _get_experiment_name(self, config, cfg_dict):
        if isinstance(config, (str, Path)):
            return Path(config).stem
        return str(cfg_dict.get("name", "experiment"))
    
    def _get_training_params(self, cfg):
        training = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
        
        def _get(key, default):
            return training.get(key, cfg.get(key, default))
    
        epochs = int(_get("epochs", 50))
        lr = float(_get("lr", 0.001))
        batch_size = int(_get("batch_size", 128))
        verbose = bool(_get("verbose", False))
        patience = _get("patience", None)
        patience = int(patience) if patience is not None else None
        min_delta = float(_get("min_delta", 0.0))
        return epochs, lr, batch_size, verbose, patience, min_delta
    
