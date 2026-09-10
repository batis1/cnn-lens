from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    from services.model_loader import load_registered_model
    from services.path_resolver import resolve_explain_image_path
    from services.custom_image import _decode_data_url, _validate_image
except ImportError:  # pragma: no cover - used when imported as backend.services
    from .model_loader import load_registered_model
    from .path_resolver import resolve_explain_image_path
    from .custom_image import _decode_data_url, _validate_image


def build_explanation(model_spec: dict, image_path: str) -> dict:
    if not model_spec.get("supportedForExplanation"):
        raise ValueError(
            f"Model {model_spec['id']} is not supported by the CNN explanation path yet."
        )

    model = load_registered_model(model_spec)
    is_inline_image = image_path.startswith("data:image/")
    if is_inline_image:
        image_bytes = _decode_data_url(image_path)
        _validate_image(image_bytes)
        resolved_image_path = BytesIO(image_bytes)
    else:
        resolved_image_path = resolve_explain_image_path(image_path)
    image_tensor, input_array = load_image_tensor(
        resolved_image_path,
        model_spec.get("inputShape", [1, 28, 28]),
        model_spec.get("pixelValueScale", 1.0),
    )

    layers, logits = trace_model(model, image_tensor, input_array)
    probabilities = torch.softmax(logits, dim=0)
    prediction_index = int(torch.argmax(probabilities).item())
    class_labels = model_spec.get("classLabels", [])

    return {
        "model": {
            "id": model_spec["id"],
            "label": model_spec["label"],
            "framework": model_spec["framework"],
            "family": model_spec["family"],
            "sourcePath": model_spec["sourcePath"],
            "inputShape": model_spec["inputShape"],
            "classLabels": class_labels,
        },
        "image": {
            "requestedPath": "custom image" if is_inline_image else image_path,
            "resolvedPath": "custom image" if is_inline_image else str(resolved_image_path),
            "inferredLabel": None if is_inline_image else infer_label_from_filename(resolved_image_path),
            "preprocessing": {
                "mode": "grayscale" if model_spec.get("inputShape", [1])[0] == 1 else "rgb",
                "resize": model_spec.get("inputShape", [1, 28, 28])[1:],
                "normalization": normalization_label(model_spec.get("pixelValueScale", 1.0)),
                "valueRange": value_range(model_spec.get("pixelValueScale", 1.0)),
                "tensorOrder": "NCHW",
            },
        },
        "prediction": {
            "index": prediction_index,
            "label": class_labels[prediction_index]
            if prediction_index < len(class_labels)
            else str(prediction_index),
            "probability": float(probabilities[prediction_index].item()),
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
        },
        "graph": {
            "format": "cnn-explainer-pytorch-v1",
            "inputShape": model_spec["inputShape"],
            "layers": layers,
        },
    }


def load_image_tensor(image_path: Path, input_shape: list[int], pixel_value_scale: float = 1.0):
    channels, height, width = input_shape
    image_mode = "L" if channels == 1 else "RGB"
    image = Image.open(image_path).convert(image_mode).resize((width, height))
    image_array = np.asarray(image, dtype=np.float32)
    if pixel_value_scale and pixel_value_scale != 1.0:
        image_array = image_array / float(pixel_value_scale)

    if channels == 1:
        if image_array.ndim != 2:
            raise ValueError(f"Expected grayscale image array, got shape {image_array.shape}")
        chw_array = image_array[np.newaxis, :, :]
    else:
        if image_array.ndim != 3 or image_array.shape[2] != channels:
            raise ValueError(
                f"Expected {channels}-channel image array, got shape {image_array.shape}"
            )
        chw_array = np.transpose(image_array, (2, 0, 1))

    tensor = torch.from_numpy(chw_array).float().unsqueeze(0)
    return tensor, chw_array


def normalization_label(pixel_value_scale: float) -> str:
    if pixel_value_scale and pixel_value_scale != 1.0:
        return f"divide_by_{pixel_value_scale:g}"
    return "none"


def value_range(pixel_value_scale: float) -> list[float]:
    if pixel_value_scale and pixel_value_scale != 1.0:
        return [0.0, 255.0 / float(pixel_value_scale)]
    return [0, 255]


def trace_model(model: nn.Module, image_tensor: torch.Tensor, input_array: np.ndarray):
    leaf_modules = list(iter_leaf_modules(model))
    activations = []
    hooks = []

    for module_name, module in leaf_modules:
        hooks.append(module.register_forward_hook(make_activation_hook(module_name, activations)))

    try:
        with torch.no_grad():
            raw_output = model(image_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    logits = raw_output.detach().cpu().squeeze(0)
    layers = build_layers(input_array, activations)
    return layers, logits


def iter_leaf_modules(model: nn.Module):
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if len(list(module.children())) == 0:
            yield module_name, module


def make_activation_hook(module_name: str, activations: list[dict]):
    def hook(module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        activations.append(
            {
                "moduleName": module_name,
                "module": module,
                "output": output.detach().cpu().squeeze(0),
            }
        )

    return hook


def build_layers(input_array: np.ndarray, activations: list[dict]) -> list[dict]:
    layers = [build_input_layer(input_array)]

    for activation in activations:
        module = activation["module"]
        output = activation["output"]
        previous_layer = layers[-1]
        layer_index = len(layers)

        if isinstance(module, nn.Conv2d):
            layer = build_conv_layer(layer_index, activation, previous_layer)
        elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            layer = build_activation_layer(layer_index, activation, previous_layer)
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            layer = build_pool_layer(layer_index, activation, previous_layer)
        elif isinstance(module, nn.Flatten):
            layer = build_flatten_layer(layer_index, activation, previous_layer)
        elif isinstance(module, nn.Linear):
            layer = build_linear_layer(layer_index, activation, previous_layer)
        else:
            layer = build_unknown_layer(layer_index, activation, output)

        layers.append(layer)

    return layers


def build_input_layer(input_array: np.ndarray) -> dict:
    nodes = [
        {
            "index": channel_index,
            "bias": 0.0,
            "output": channel.tolist(),
            "inputLinks": [],
        }
        for channel_index, channel in enumerate(input_array)
    ]

    return {
        "index": 0,
        "name": "input",
        "type": "input",
        "outputShape": list(input_array.shape),
        "nodes": nodes,
    }


def build_conv_layer(layer_index: int, activation: dict, previous_layer: dict) -> dict:
    module: nn.Conv2d = activation["module"]
    output = as_channel_tensor(activation["output"], activation["moduleName"])
    weights = module.weight.detach().cpu()
    bias = module.bias.detach().cpu().tolist() if module.bias is not None else [0.0] * weights.shape[0]

    nodes = []
    for output_channel in range(output.shape[0]):
        nodes.append(
            {
                "index": output_channel,
                "bias": float(bias[output_channel]),
                "output": output[output_channel].tolist(),
                "inputLinks": [
                    {
                        "sourceLayerIndex": previous_layer["index"],
                        "sourceNodeIndex": input_channel,
                        "weight": weights[output_channel, input_channel].tolist(),
                    }
                    for input_channel in range(weights.shape[1])
                ],
            }
        )

    return {
        "index": layer_index,
        "name": activation["moduleName"],
        "type": "conv",
        "moduleClass": module.__class__.__name__,
        "outputShape": list(output.shape),
        "params": {
            "inChannels": module.in_channels,
            "outChannels": module.out_channels,
            "kernelSize": list(pair(module.kernel_size)),
            "stride": list(pair(module.stride)),
            "padding": list(pair(module.padding)),
            "dilation": list(pair(module.dilation)),
            "groups": module.groups,
        },
        "nodes": nodes,
    }


def build_activation_layer(layer_index: int, activation: dict, previous_layer: dict) -> dict:
    module = activation["module"]
    output = activation["output"]

    if output.ndim == 3:
        output_values = output
        nodes = [
            {
                "index": channel_index,
                "bias": 0.0,
                "output": output_values[channel_index].tolist(),
                "inputLinks": [
                    {
                        "sourceLayerIndex": previous_layer["index"],
                        "sourceNodeIndex": channel_index,
                        "weight": None,
                    }
                ],
            }
            for channel_index in range(output_values.shape[0])
        ]
    elif output.ndim == 1:
        nodes = [
            {
                "index": node_index,
                "bias": 0.0,
                "output": float(output[node_index].item()),
                "inputLinks": [
                    {
                        "sourceLayerIndex": previous_layer["index"],
                        "sourceNodeIndex": node_index,
                        "weight": None,
                    }
                ],
            }
            for node_index in range(output.shape[0])
        ]
    else:
        raise ValueError(
            f"Unsupported activation output shape for {activation['moduleName']}: {list(output.shape)}"
        )

    return {
        "index": layer_index,
        "name": activation["moduleName"],
        "type": "relu",
        "moduleClass": module.__class__.__name__,
        "activationName": activation_name(module),
        "outputShape": list(output.shape),
        "nodes": nodes,
    }


def build_pool_layer(layer_index: int, activation: dict, previous_layer: dict) -> dict:
    module = activation["module"]
    output = as_channel_tensor(activation["output"], activation["moduleName"])

    nodes = [
        {
            "index": channel_index,
            "bias": 0.0,
            "output": output[channel_index].tolist(),
            "inputLinks": [
                {
                    "sourceLayerIndex": previous_layer["index"],
                    "sourceNodeIndex": channel_index,
                    "weight": None,
                }
            ],
        }
        for channel_index in range(output.shape[0])
    ]

    return {
        "index": layer_index,
        "name": activation["moduleName"],
        "type": "pool",
        "moduleClass": module.__class__.__name__,
        "poolType": "avg" if isinstance(module, nn.AvgPool2d) else "max",
        "outputShape": list(output.shape),
        "params": {
            "kernelSize": list(pair(module.kernel_size)),
            "stride": list(pair(module.stride or module.kernel_size)),
            "padding": list(pair(module.padding)),
            "dilation": list(pair(getattr(module, "dilation", 1))),
            "ceilMode": bool(getattr(module, "ceil_mode", False)),
        },
        "nodes": nodes,
    }


def build_flatten_layer(layer_index: int, activation: dict, previous_layer: dict) -> dict:
    output = activation["output"].flatten()
    previous_shape = previous_layer.get("outputShape", [])
    nodes = []

    for flat_index, value in enumerate(output):
        link = {
            "sourceLayerIndex": previous_layer["index"],
            "sourceNodeIndex": flat_index,
            "weight": None,
        }

        if len(previous_shape) == 3:
            channels, height, width = previous_shape
            channel = flat_index // (height * width)
            remainder = flat_index % (height * width)
            row = remainder // width
            col = remainder % width
            link.update(
                {
                    "sourceNodeIndex": channel,
                    "position": [row, col],
                    "flattenOrder": "channel_row_col",
                }
            )

        nodes.append(
            {
                "index": flat_index,
                "bias": 0.0,
                "output": float(value.item()),
                "inputLinks": [link],
            }
        )

    return {
        "index": layer_index,
        "name": activation["moduleName"],
        "type": "flatten",
        "moduleClass": activation["module"].__class__.__name__,
        "outputShape": [len(nodes)],
        "nodes": nodes,
    }


def build_linear_layer(layer_index: int, activation: dict, previous_layer: dict) -> dict:
    module: nn.Linear = activation["module"]
    output = activation["output"].flatten()
    weights = module.weight.detach().cpu()
    bias = module.bias.detach().cpu().tolist() if module.bias is not None else [0.0] * weights.shape[0]
    is_output_layer = output.shape[0] == 10

    nodes = []
    for output_index in range(output.shape[0]):
        nodes.append(
            {
                "index": output_index,
                "bias": float(bias[output_index]),
                "output": float(output[output_index].item()),
                "logit": float(output[output_index].item()) if is_output_layer else None,
                "isOutputLayer": is_output_layer,
                "activationName": "linear",
                "inputLinks": [
                    {
                        "sourceLayerIndex": previous_layer["index"],
                        "sourceNodeIndex": input_index,
                        "weight": float(weights[output_index, input_index].item()),
                    }
                    for input_index in range(weights.shape[1])
                ],
            }
        )

    return {
        "index": layer_index,
        "name": "output" if is_output_layer else activation["moduleName"],
        "type": "fc",
        "moduleClass": module.__class__.__name__,
        "outputShape": [len(nodes)],
        "params": {
            "inFeatures": module.in_features,
            "outFeatures": module.out_features,
        },
        "nodes": nodes,
    }


def build_unknown_layer(layer_index: int, activation: dict, output: torch.Tensor) -> dict:
    return {
        "index": layer_index,
        "name": activation["moduleName"],
        "type": "unknown",
        "moduleClass": activation["module"].__class__.__name__,
        "outputShape": list(output.shape),
        "nodes": [],
    }


def as_channel_tensor(output: torch.Tensor, module_name: str) -> torch.Tensor:
    if output.ndim != 3:
        raise ValueError(f"Expected channel tensor for {module_name}, got shape {list(output.shape)}")
    return output


def pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def activation_name(module: nn.Module) -> str:
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    if isinstance(module, nn.Tanh):
        return "tanh"
    return "relu"


def infer_label_from_filename(image_path: Path):
    match = re.search(r"_(\d+)$", image_path.stem)
    return int(match.group(1)) if match else None
