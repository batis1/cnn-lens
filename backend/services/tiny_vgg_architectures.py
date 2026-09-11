from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn


TINY_VGG_CLASS_LABELS = [
    "goldfish",
    "tabby cat",
    "German shepherd",
    "monarch butterfly",
    "banana",
    "pomegranate",
    "bullet train",
    "lighthouse",
    "sunglasses",
    "refrigerator",
]


@dataclass(frozen=True)
class TinyVggSpec:
    key: str
    label: str
    file_name: str
    builder: Callable[[], nn.Module]
    hint: str
    flatten_shape: tuple[int, int, int]


class TinyVgg7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 6, kernel_size=3)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(6, 6, kernel_size=3)
        self.sigmoid_1_1 = nn.Sigmoid()
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(5400, 64)
        self.relu_dense_1 = nn.ReLU()
        self.dense_2 = nn.Linear(64, 32)
        self.relu_dense_2 = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.relu_1_1(x)
        x = self.conv_1_2(x)
        x = self.sigmoid_1_1(x)
        x = self.avg_pool_1(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.relu_dense_1(x)
        x = self.dense_2(x)
        x = self.relu_dense_2(x)
        return self.output(x)


class TinyVgg12(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 10, kernel_size=3)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(10, 10, kernel_size=3)
        self.relu_1_2 = nn.ReLU()
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = nn.Conv2d(10, 10, kernel_size=3)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(10, 10, kernel_size=3)
        self.relu_2_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(1690, 64)
        self.relu_dense_1 = nn.ReLU()
        self.dense_2 = nn.Linear(64, 32)
        self.relu_dense_2 = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.relu_1_1(x)
        x = self.conv_1_2(x)
        x = self.relu_1_2(x)
        x = self.avg_pool_1(x)
        x = self.conv_2_1(x)
        x = self.relu_2_1(x)
        x = self.conv_2_2(x)
        x = self.relu_2_2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.relu_dense_1(x)
        x = self.dense_2(x)
        x = self.relu_dense_2(x)
        return self.output(x)


class TinyVgg17(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 14, kernel_size=3)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(14, 14, kernel_size=3)
        self.relu_1_2 = nn.ReLU()
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_2_1 = nn.Conv2d(14, 14, kernel_size=3)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(14, 14, kernel_size=3)
        self.relu_2_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3_1 = nn.Conv2d(14, 14, kernel_size=3)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(14, 14, kernel_size=3)
        self.relu_3_2 = nn.ReLU()
        self.avg_pool_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(224, 64)
        self.relu_dense_1 = nn.ReLU()
        self.dense_2 = nn.Linear(64, 32)
        self.relu_dense_2 = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv_1_1(x)
        x = self.relu_1_1(x)
        x = self.conv_1_2(x)
        x = self.relu_1_2(x)
        x = self.avg_pool_1(x)
        x = self.conv_2_1(x)
        x = self.relu_2_1(x)
        x = self.conv_2_2(x)
        x = self.relu_2_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3_1(x)
        x = self.relu_3_1(x)
        x = self.conv_3_2(x)
        x = self.relu_3_2(x)
        x = self.avg_pool_3(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.relu_dense_1(x)
        x = self.dense_2(x)
        x = self.relu_dense_2(x)
        return self.output(x)


TINY_VGG_SPECS = {
    "tiny-vgg-7": TinyVggSpec(
        key="tiny-vgg-7",
        label="7-layer PyTorch",
        file_name="tiny_vgg_7.pt",
        builder=TinyVgg7,
        hint="Conv2d -> ReLU -> Conv2d -> Sigmoid -> AvgPool2d -> Flatten -> Linear",
        flatten_shape=(6, 30, 30),
    ),
    "tiny-vgg-12": TinyVggSpec(
        key="tiny-vgg-12",
        label="12-layer PyTorch",
        file_name="tiny_vgg_12.pt",
        builder=TinyVgg12,
        hint=(
            "Conv2d -> ReLU -> Conv2d -> ReLU -> AvgPool2d -> "
            "Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear"
        ),
        flatten_shape=(10, 13, 13),
    ),
    "tiny-vgg-17": TinyVggSpec(
        key="tiny-vgg-17",
        label="17-layer PyTorch",
        file_name="tiny_vgg_17.pt",
        builder=TinyVgg17,
        hint=(
            "Conv2d -> ReLU -> Conv2d -> ReLU -> AvgPool2d -> "
            "Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d -> "
            "Conv2d -> ReLU -> Conv2d -> ReLU -> AvgPool2d -> Flatten -> Linear"
        ),
        flatten_shape=(14, 4, 4),
    ),
}


def build_tiny_vgg_model(architecture_key: str) -> nn.Module:
    try:
        return TINY_VGG_SPECS[architecture_key].builder()
    except KeyError as exc:
        raise ValueError(f"Unknown TinyVGG architecture: {architecture_key}") from exc
