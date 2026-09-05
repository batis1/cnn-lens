from __future__ import annotations

from functools import lru_cache

import torch

try:
    from services.tiny_vgg_architectures import build_tiny_vgg_model
except ImportError:  # pragma: no cover - used when imported as backend.services
    from .tiny_vgg_architectures import build_tiny_vgg_model


@lru_cache(maxsize=8)
def load_registered_model_cached(
    model_id: str,
    source_path: str,
    architecture_key: str | None = None,
):
    if architecture_key:
        checkpoint = load_state_dict_checkpoint(source_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = build_tiny_vgg_model(architecture_key)
        model.load_state_dict(state_dict)
        model.eval()
        model.cpu()
        return model

    try:
        model = torch.load(source_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(source_path, map_location="cpu")

    model.eval()
    model.cpu()
    return model


def load_registered_model(model_spec: dict):
    if not model_spec.get("exists"):
        raise ValueError(f"Model file is missing: {model_spec.get('sourcePath')}")

    # The current AI_Test_UI files are trusted local full-model pickles, not
    # generic user uploads. Arbitrary uploads should use state_dict contracts.
    #
    # PYTORCH_TINY_VGG_INTEGRATION:
    # New project-local TinyVGG exports use state_dict checkpoints with an
    # explicit architecture key. Comment this branch if you only want the old
    # full-pickle AI_Test_UI model behavior.
    return load_registered_model_cached(
        model_spec["id"],
        model_spec["sourcePath"],
        model_spec.get("architectureKey"),
    )


def load_state_dict_checkpoint(source_path: str):
    try:
        return torch.load(source_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(source_path, map_location="cpu")
