import torch

ABS_MAX_DATA = 20
ABS_MAX_DEVICE = 511


def _center_channels(data: torch.Tensor) -> torch.Tensor:
    mean = torch.unsqueeze(torch.mean(data, dim=-2), dim=-2)
    return data - mean


def preprocess(data: torch.Tensor) -> torch.Tensor:
    device_specific_scaling = ABS_MAX_DEVICE / ABS_MAX_DATA
    data = torch.clamp(data * device_specific_scaling, -ABS_MAX_DEVICE, ABS_MAX_DEVICE)
    data = _center_channels(data)
    return data
