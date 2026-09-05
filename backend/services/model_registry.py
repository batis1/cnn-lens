from __future__ import annotations

from pathlib import Path

try:
    from config import AI_TEST_UI_ROOT, PROJECT_MODEL_ROOT
    from services.tiny_vgg_architectures import TINY_VGG_CLASS_LABELS, TINY_VGG_SPECS
except ImportError:  # pragma: no cover - used when imported as backend.services
    from ..config import AI_TEST_UI_ROOT, PROJECT_MODEL_ROOT
    from .tiny_vgg_architectures import TINY_VGG_CLASS_LABELS, TINY_VGG_SPECS


CLASS_LABELS_10 = [str(index) for index in range(10)]
TINY_VGG_INPUT_SHAPE = [3, 64, 64]

_MODEL_SPECS = [
    {
        "id": "cnn-mynet28",
        "label": "CNN MyNet28",
        "fileName": "CNN_MyNet28.pt",
        "family": "cnn",
        "framework": "pytorch",
        "inputShape": [1, 28, 28],
        "classLabels": CLASS_LABELS_10,
        "supportedForExplanation": True,
        "architectureHint": "Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear",
    },
    {
        "id": "cnn-net-28-ori",
        "label": "CNN NET 28 original",
        "fileName": "CNN_NET_28_ori.pt",
        "family": "cnn",
        "framework": "pytorch",
        "inputShape": [1, 28, 28],
        "classLabels": CLASS_LABELS_10,
        "supportedForExplanation": True,
        "architectureHint": "Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear",
    },
    {
        "id": "cnn-net-28-pru1",
        "label": "CNN NET 28 pruned 1",
        "fileName": "CNN_NET_28_pru1.pt",
        "family": "cnn",
        "framework": "pytorch",
        "inputShape": [1, 28, 28],
        "classLabels": CLASS_LABELS_10,
        "supportedForExplanation": True,
        "architectureHint": "Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear",
    },
    {
        "id": "cnn-net-28-pru2",
        "label": "CNN NET 28 pruned 2",
        "fileName": "CNN_NET_28_pru2.pt",
        "family": "cnn",
        "framework": "pytorch",
        "inputShape": [1, 28, 28],
        "classLabels": CLASS_LABELS_10,
        "supportedForExplanation": True,
        "architectureHint": "Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear",
    },
    {
        "id": "mlp-mynet28",
        "label": "MLP MyNet28",
        "fileName": "MLP_MyNet28.pt",
        "family": "mlp",
        "framework": "pytorch",
        "inputShape": [1, 28, 28],
        "classLabels": CLASS_LABELS_10,
        "supportedForExplanation": False,
        "architectureHint": "Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear",
        "note": "Keep this for AI_Test_UI coverage parity; the CNN explainer path should start with CNN models.",
    },
    # PYTORCH_TINY_VGG_INTEGRATION:
    # Project-local PyTorch equivalents of the existing TensorFlow.js 7/12/17
    # TinyVGG models. To return to only the AI_Test_UI models, comment these
    # entries and leave the full-pickle entries above as they are.
    {
        "id": "tiny-vgg-7-pytorch",
        "label": TINY_VGG_SPECS["tiny-vgg-7"].label,
        "fileName": TINY_VGG_SPECS["tiny-vgg-7"].file_name,
        "sourceRoot": "project-backend-models",
        "family": "tiny-vgg",
        "framework": "pytorch",
        "architectureKey": "tiny-vgg-7",
        "inputShape": TINY_VGG_INPUT_SHAPE,
        "pixelValueScale": 255.0,
        "classLabels": TINY_VGG_CLASS_LABELS,
        "supportedForExplanation": True,
        "architectureHint": TINY_VGG_SPECS["tiny-vgg-7"].hint,
    },
    {
        "id": "tiny-vgg-12-pytorch",
        "label": TINY_VGG_SPECS["tiny-vgg-12"].label,
        "fileName": TINY_VGG_SPECS["tiny-vgg-12"].file_name,
        "sourceRoot": "project-backend-models",
        "family": "tiny-vgg",
        "framework": "pytorch",
        "architectureKey": "tiny-vgg-12",
        "inputShape": TINY_VGG_INPUT_SHAPE,
        "pixelValueScale": 255.0,
        "classLabels": TINY_VGG_CLASS_LABELS,
        "supportedForExplanation": True,
        "architectureHint": TINY_VGG_SPECS["tiny-vgg-12"].hint,
    },
    {
        "id": "tiny-vgg-17-pytorch",
        "label": TINY_VGG_SPECS["tiny-vgg-17"].label,
        "fileName": TINY_VGG_SPECS["tiny-vgg-17"].file_name,
        "sourceRoot": "project-backend-models",
        "family": "tiny-vgg",
        "framework": "pytorch",
        "architectureKey": "tiny-vgg-17",
        "inputShape": TINY_VGG_INPUT_SHAPE,
        "pixelValueScale": 255.0,
        "classLabels": TINY_VGG_CLASS_LABELS,
        "supportedForExplanation": True,
        "architectureHint": TINY_VGG_SPECS["tiny-vgg-17"].hint,
    },
]


def list_models():
    return [_with_file_status(spec) for spec in _MODEL_SPECS]


def get_model(model_id: str):
    for model in list_models():
        if model["id"] == model_id:
            return model

    return None


def find_model_by_source_path(path_text: str | None):
    if not path_text:
        return None

    requested = Path(path_text)
    requested_name = requested.name

    for model in list_models():
        source_path = Path(model["sourcePath"])

        if requested_name and requested_name == source_path.name:
            return model

        if requested.is_absolute() and requested.resolve() == source_path.resolve():
            return model

    return None


def _with_file_status(spec):
    model_root = PROJECT_MODEL_ROOT if spec.get("sourceRoot") == "project-backend-models" else AI_TEST_UI_ROOT
    model_path = model_root / spec["fileName"]
    enriched = dict(spec)
    enriched.update(
        {
            "sourceRoot": spec.get("sourceRoot", "AI_Test_UI"),
            "sourcePath": str(model_path),
            "exists": model_path.exists(),
            "sizeBytes": model_path.stat().st_size if model_path.exists() else None,
        }
    )
    return enriched
