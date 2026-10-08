"""Model architectures. Each returns a single logit; positive means benign."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_features: int = None, num_classes: int = 1):
        super().__init__()
        layers, in_channels = [], 1
        for _ in range(10):
            layers += [nn.Conv1d(in_channels, 108, kernel_size=5, padding=2),
                       nn.BatchNorm1d(108), nn.ReLU()]
            in_channels = 108
        self.cnn_backbone = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(108, num_classes)

    def forward(self, x, return_features: bool = False):
        x = x.unsqueeze(1)
        x = self.cnn_backbone(x)
        features = self.global_pool(x).squeeze(-1)
        logit = self.classifier(features)
        return (logit, features) if return_features else logit


class LSTMModel(nn.Module):
    def __init__(self, num_features: int = None, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features: bool = False):
        x, _ = self.lstm(x.unsqueeze(-1))
        features = x[:, -1, :]
        logit = self.classifier(features)
        return (logit, features) if return_features else logit


class CLSTMModel(nn.Module):
    def __init__(self, num_features: int = None, hidden_size: int = 128, num_classes: int = 1):
        super().__init__()
        layers, in_channels = [], 1
        for _ in range(5):
            layers += [nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                       nn.BatchNorm1d(64), nn.ReLU()]
            in_channels = 64
        self.cnn_backbone = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features: bool = False):
        x = x.unsqueeze(1)
        x = self.cnn_backbone(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        features = x[:, -1, :]
        logit = self.classifier(features)
        return (logit, features) if return_features else logit


MODEL_REGISTRY = {"cnn": CNNModel, "lstm": LSTMModel, "clstm": CLSTMModel}
RECURRENT = {"lstm", "clstm"}


def build_model(arch: str, num_features: int) -> nn.Module:
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[arch](num_features)
