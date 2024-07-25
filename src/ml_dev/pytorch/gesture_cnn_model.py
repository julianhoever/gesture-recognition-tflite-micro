from functools import partial

import torch
import torch.ao.quantization


class GestureCnnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conv_block = partial(_ConvBlock, kernel_size=4, pool_size=3)
        self.quant = torch.ao.quantization.QuantStub()
        self.conv0 = conv_block(in_channels=3, out_channels=32)
        self.conv1 = conv_block(in_channels=32, out_channels=12)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.flatten = torch.nn.Flatten()
        self.clf = torch.nn.Linear(in_features=144, out_features=4)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.quant(inputs)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.clf(x)
        return self.dequant(x)


class _ConvBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ) -> None:
        super().__init__()
        self.dw_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
        )
        self.pw_conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
        )
        self.batchnorm = torch.nn.BatchNorm1d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(inputs)
        x = self.pw_conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
