from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.tiny_vgg_architectures import (  # noqa: E402
    TINY_VGG_CLASS_LABELS,
    TINY_VGG_SPECS,
    build_tiny_vgg_model,
)


TFJS_MODEL_ROOT = PROJECT_ROOT / "public" / "assets" / "data" / "models"
PYTORCH_MODEL_ROOT = PROJECT_ROOT / "backend" / "models"

EXPORT_SPECS = [
    {
        "architectureKey": "tiny-vgg-7",
        "tfjsDir": TFJS_MODEL_ROOT / "compact-7",
    },
    {
        "architectureKey": "tiny-vgg-12",
        "tfjsDir": TFJS_MODEL_ROOT / "balanced-12",
    },
    {
        "architectureKey": "tiny-vgg-17",
        "tfjsDir": TFJS_MODEL_ROOT / "deep-17",
    },
]


def load_tfjs_weight_map(model_dir: Path) -> dict[str, np.ndarray]:
    model_json_path = model_dir / "model.json"
    model_json = json.loads(model_json_path.read_text(encoding="utf-8"))
    manifest = model_json["weightsManifest"][0]

    raw_weights = b"".join((model_dir / rel_path).read_bytes() for rel_path in manifest["paths"])
    weight_map: dict[str, np.ndarray] = {}
    offset = 0

    for weight_spec in manifest["weights"]:
        if weight_spec["dtype"] != "float32":
            raise ValueError(f"Unsupported dtype in {model_json_path}: {weight_spec}")

        shape = tuple(weight_spec["shape"])
        item_count = int(np.prod(shape))
        byte_count = item_count * np.dtype("<f4").itemsize
        weight_name = normalize_tfjs_weight_name(weight_spec["name"])
        weight = np.frombuffer(raw_weights, dtype="<f4", count=item_count, offset=offset)
        weight_map[weight_name] = weight.reshape(shape).copy()
        offset += byte_count

    if offset != len(raw_weights):
        raise ValueError(f"Unused bytes in TFJS weight shard: {model_json_path}")

    return weight_map


def normalize_tfjs_weight_name(name: str) -> str:
    for prefix in ("sequential/", "sequential_1/"):
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


def convert_tfjs_weights_to_state_dict(
    architecture_key: str,
    weight_map: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    spec = TINY_VGG_SPECS[architecture_key]
    state_dict: dict[str, torch.Tensor] = {}

    for layer_name in sorted(
        name.removesuffix("/kernel")
        for name in weight_map
        if name.endswith("/kernel") and name.startswith("conv_")
    ):
        assign_conv2d(state_dict, weight_map, layer_name)

    assign_dense(
        state_dict,
        weight_map,
        "dense_1",
        flatten_shape_chw=spec.flatten_shape,
    )
    assign_dense(state_dict, weight_map, "dense_2")
    assign_dense(state_dict, weight_map, "output")

    return state_dict


def assign_conv2d(
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, np.ndarray],
    layer_name: str,
) -> None:
    kernel = weight_map[f"{layer_name}/kernel"]
    bias = weight_map[f"{layer_name}/bias"]

    # TF/Keras Conv2D stores [height, width, input_channels, output_channels].
    # PyTorch Conv2d stores [output_channels, input_channels, height, width].
    state_dict[f"{layer_name}.weight"] = torch.from_numpy(
        np.transpose(kernel, (3, 2, 0, 1)).copy()
    )
    state_dict[f"{layer_name}.bias"] = torch.from_numpy(bias.copy())


def assign_dense(
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, np.ndarray],
    layer_name: str,
    flatten_shape_chw: tuple[int, int, int] | None = None,
) -> None:
    kernel = weight_map[f"{layer_name}/kernel"]
    bias = weight_map[f"{layer_name}/bias"]

    if flatten_shape_chw is None:
        torch_kernel = np.transpose(kernel, (1, 0))
    else:
        torch_kernel = convert_first_dense_kernel(kernel, flatten_shape_chw)

    state_dict[f"{layer_name}.weight"] = torch.from_numpy(torch_kernel.copy())
    state_dict[f"{layer_name}.bias"] = torch.from_numpy(bias.copy())


def convert_first_dense_kernel(
    keras_kernel: np.ndarray,
    flatten_shape_chw: tuple[int, int, int],
) -> np.ndarray:
    channels, height, width = flatten_shape_chw
    input_features, output_features = keras_kernel.shape
    expected_features = channels * height * width

    if input_features != expected_features:
        raise ValueError(
            f"Dense input size mismatch: got {input_features}, expected {expected_features}"
        )

    # Keras Flatten on NHWC orders values as [row, col, channel]. PyTorch
    # Flatten on NCHW orders values as [channel, row, col].
    return keras_kernel.reshape(height, width, channels, output_features).transpose(3, 2, 0, 1).reshape(
        output_features,
        expected_features,
    )


def export_one(architecture_key: str, tfjs_dir: Path, output_dir: Path = PYTORCH_MODEL_ROOT) -> Path:
    spec = TINY_VGG_SPECS[architecture_key]
    weight_map = load_tfjs_weight_map(tfjs_dir)
    state_dict = convert_tfjs_weights_to_state_dict(architecture_key, weight_map)
    model = build_tiny_vgg_model(architecture_key)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 64, 64))
    if tuple(logits.shape) != (1, 10):
        raise ValueError(f"Unexpected output shape for {architecture_key}: {tuple(logits.shape)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.file_name
    torch.save(
        {
            "format": "cnn-explainer-tiny-vgg-state-dict-v1",
            "architectureKey": architecture_key,
            "inputShape": [3, 64, 64],
            "classLabels": TINY_VGG_CLASS_LABELS,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )
    return output_path


def export_all(project_root: Path | None = None) -> list[Path]:
    if project_root is None:
        project_root = PROJECT_ROOT

    output_dir = project_root / "backend" / "models"
    exported_paths = []
    for export_spec in EXPORT_SPECS:
        exported_paths.append(
            export_one(
                export_spec["architectureKey"],
                export_spec["tfjsDir"],
                output_dir,
            )
        )

    return exported_paths


if __name__ == "__main__":
    for path in export_all():
        print(path)
