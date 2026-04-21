# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Save PyTorch models (.pt) with embedded metadata."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hc=None):
        out, (h, c) = self.lstm(x, hc)
        return self.decoder(out[:, -1, :]), (h, c)


def save_with_metadata(model, path, metadata):
    """Save a model checkpoint with arbitrary metadata.

    Args:
        model: The nn.Module to save.
        path: Destination .pt file path.
        metadata: Dict of metadata to embed alongside the weights.
    """
    torch.save({"model": model, "metadata": metadata}, path)
    print(f"Saved to {path}  metadata={metadata}")


if __name__ == "__main__":
    mlp = MLP(input_dim=2, hidden_dim=32, output_dim=1)
    save_with_metadata(
        mlp,
        "mlp_controller.pt",
        {
            "model_type": "mlp",
            "input_dim": 2,
            "hidden_dim": 32,
            "output_dim": 1,
            "description": "PD-like MLP controller",
        },
    )

    lstm = LSTMNet(input_dim=2, hidden_dim=16, output_dim=1, num_layers=2)
    save_with_metadata(
        lstm,
        "lstm_controller.pt",
        {
            "model_type": "lstm",
            "input_dim": 2,
            "hidden_dim": 16,
            "output_dim": 1,
            "num_layers": 2,
            "description": "LSTM recurrent controller",
        },
    )
