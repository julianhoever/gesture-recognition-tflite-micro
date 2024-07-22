from functools import partial

import torch


class GestureCnnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conv_block = partial(_conv_block, kernel_size=4, pool_size=3)
        self.conv0 = conv_block(in_channels=3, out_channels=32)
        self.conv1 = conv_block(in_channels=32, out_channels=12)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.flatten = torch.nn.Flatten()
        self.clf = torch.nn.Linear(in_features=144, out_features=4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.clf(x)


def _conv_block(
    in_channels: int, out_channels: int, kernel_size: int, pool_size: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        _depthwise_separable_conv1d(in_channels, out_channels, kernel_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(kernel_size=pool_size),
        torch.nn.BatchNorm1d(num_features=out_channels),
    )


def _depthwise_separable_conv1d(
    in_channels: int, out_channels: int, kernel_size: int
) -> torch.nn.Sequential:
    depthwise_conv = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        groups=in_channels,
    )
    pointwise_conv = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        groups=1,
    )
    return torch.nn.Sequential(depthwise_conv, pointwise_conv)
