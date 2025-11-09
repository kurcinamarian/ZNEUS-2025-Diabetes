
import torch
import torch.nn as nn

from pathlib import Path

import yaml


class MultilayerPerceptron(nn.Module):
    def __init__(self, config):

        """
        config.yaml example:
        n_input: 16
        layers: [64, 128, 64, 10]
        activations: [ReLU, GELU, ReLU, None]
        output_activation: Sigmoid
        dropout: 0.3
        batch_norm: True
        skip_connections: True
        bottleneck: True
        """

        super().__init__()
        
        # Load config
        if isinstance(config, (str, Path)):
            config = self._load_config(config)
        
        n_input = config["n_input"]
        layers = config["layers"]
        self.layers = layers

        activations = config.get("activations", None)
        output_activation = config.get("output_activation", None)
        dropout = config.get("dropout", 0.0)
        batch_norm = config.get("batch_norm", False)

        self.skip_connections = config.get("skip_connections", False)
        bottleneck = config.get("bottleneck", False)

        n_layers = len(layers)
        activations = self._resolve_activations(activations, n_layers)
            
        hidden = []
        n_prev = n_input
        
        for i, n_out in enumerate(layers):

            if bottleneck and i > 0 and i < len(layers) - 1:
                bottleneck_dim = max(n_prev // 2, n_out // 2)
                hidden.append(nn.Linear(n_prev, bottleneck_dim))
                hidden.append(nn.ReLU())
                n_prev = bottleneck_dim

            hidden.append(nn.Linear(n_prev, n_out))
            
            if batch_norm and i < n_layers - 1:
                hidden.append(nn.BatchNorm1d(n_out))
            
            act = activations[i]
            if act is not None:
                hidden.append(act())
                
            if dropout > 0.0 and i < n_layers - 1:
                hidden.append(nn.Dropout(dropout))
                
            n_prev = n_out

        if output_activation is not None:
            hidden.append(self._get_activation(output_activation)())
            
        self.main = nn.Sequential(*hidden)




    def forward(self, x):
        return self.main(x)
    
    def _load_config(self, path):
        # Load a .yaml config file
        path = Path(path)
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
    def _get_activation(self, name_or_class):
        # Turn activation names like ReLU into nn.ReLU, or pass through a class
        if isinstance(name_or_class, str):
            if not hasattr(nn, name_or_class):
                raise ValueError(f"Unknown activation: {name_or_class}")
            return getattr(nn, name_or_class)
        return name_or_class
    
    def _resolve_activations(self, activations, n_layers):
        # Ensure activations list matches number of layers
        if activations is None:
            raise ValueError("Missing activations.")
        
        activations_resolved = []
        for a in activations:
            if a is None:
                activations_resolved.append(None)
            else:
                activations_resolved.append(self._get_activation(a))
        
        if len(activations_resolved) < n_layers:
            print(f"Number of activations: {len(activations)}, Number of layers: {n_layers}")
            raise ValueError("Missing activations.")
        
        return activations_resolved


# Try running with a config dictionary
# cfg = {
#     "n_input": 32,
#     "layers": [64, 128, 64, 10],
#     "activations": ["ReLU", "GELU", "ReLU", None],
#     "output_activation": "Sigmoid",
#     "dropout": 0.3,
#     "batch_norm": True,
# }
# model = MultilayerPerceptron(cfg)
# # model = MultilayerPerceptron(".scratch/config.yaml")
# x = torch.randn(8, 32)
# out = model(x)
# print(out.shape)

