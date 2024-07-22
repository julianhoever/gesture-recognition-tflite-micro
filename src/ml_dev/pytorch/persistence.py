from pathlib import Path

import torch


def save_weights(model: torch.nn.Module, weights_file: Path) -> None:
    state_dict = model.state_dict()
    torch.save(state_dict, weights_file)


def load_weights(model: torch.nn.Module, weights_file: Path) -> None:
    state_dict = torch.load(weights_file, weights_only=True)
    model.load_state_dict(state_dict)
