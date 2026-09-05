from __future__ import annotations

import re
from pathlib import Path

import torch

try:
    from config import AI_TEST_UI_ROOT
    from services.explain import load_image_tensor
    from services.model_loader import load_registered_model
    from services.path_resolver import IMAGE_SUFFIXES
except ImportError:  # pragma: no cover - used when imported as backend.services
    from ..config import AI_TEST_UI_ROOT
    from .explain import load_image_tensor
    from .model_loader import load_registered_model
    from .path_resolver import IMAGE_SUFFIXES


ROBUSTNESS_GROUPS = [
    "gauss_noise",
    "poisson_noise",
    "salt_pepper_noise",
    "rotation",
    "scale",
    "translation",
]

ADVERSARY_GROUPS = [
    "epsilon_0.1",
    "epsilon_0.2",
    "epsilon_0.3",
]

DISPLAY_NAMES = {
    "accuracy": "accuracy",
    "gauss_noise": "gauss noise",
    "poisson_noise": "poisson noise",
    "salt_pepper_noise": "salt pepper noise",
    "rotation": "rotation",
    "scale": "scale",
    "translation": "translation",
    "epsilon_0.1": "epsilon 0.1",
    "epsilon_0.2": "epsilon 0.2",
    "epsilon_0.3": "epsilon 0.3",
}


# PYTORCH_BACKEND_INTEGRATION:
# This ports AI_Test_UI/test.py accuracy, robustness, and adversary testing into
# a reusable service so Flask returns clean JSON instead of subprocess stdout.
def run_model_test(
    model_spec: dict,
    test_data_path: str | None,
    test_type: str | None,
    limit: int | str | None = None,
) -> dict:
    resolved_test_type = normalize_test_type(test_type)
    resolved_limit = normalize_limit(limit)
    resolved_data_path = resolve_ai_test_data_dir(test_data_path, resolved_test_type)

    model = load_registered_model(model_spec)
    input_shape = model_spec.get("inputShape", [1, 28, 28])
    class_labels = model_spec.get("classLabels", [])

    if resolved_test_type == "accuracy":
        accuracy_result = evaluate_image_directory(
            model=model,
            image_dir=resolved_data_path,
            input_shape=input_shape,
            class_labels=class_labels,
            limit=resolved_limit,
        )
        return build_base_response(
            model_spec=model_spec,
            test_type=resolved_test_type,
            requested_data_path=test_data_path,
            resolved_data_path=resolved_data_path,
            limit=resolved_limit,
        ) | {
            "success": True,
            "accuracy": accuracy_result["accuracy"],
            "correct": accuracy_result["correct"],
            "incorrect": accuracy_result["incorrect"],
            "total": accuracy_result["total"],
            "samples": accuracy_result["samples"],
            "output": f"accuracy: {accuracy_result['accuracy']:.4f}",
            "type": "accuracy",
        }

    group_names = ROBUSTNESS_GROUPS if resolved_test_type == "robustness" else ADVERSARY_GROUPS
    group_results = evaluate_grouped_directory(
        model=model,
        root_dir=resolved_data_path,
        group_names=group_names,
        input_shape=input_shape,
        class_labels=class_labels,
        limit=resolved_limit,
    )
    accuracy_by_group = {
        group["name"]: group["accuracy"]
        for group in group_results
    }
    total = sum(group["total"] for group in group_results)
    correct = sum(group["correct"] for group in group_results)
    aggregate_accuracy = correct / total if total else 0.0

    return build_base_response(
        model_spec=model_spec,
        test_type=resolved_test_type,
        requested_data_path=test_data_path,
        resolved_data_path=resolved_data_path,
        limit=resolved_limit,
    ) | {
        "success": True,
        resolved_test_type: accuracy_by_group,
        "groups": group_results,
        "accuracy": aggregate_accuracy,
        "correct": correct,
        "incorrect": total - correct,
        "total": total,
        "output": build_group_output(group_results),
        "type": resolved_test_type,
    }


def build_base_response(
    model_spec: dict,
    test_type: str,
    requested_data_path: str | None,
    resolved_data_path: Path,
    limit: int | None,
) -> dict:
    return {
        "model": {
            "id": model_spec["id"],
            "label": model_spec["label"],
            "framework": model_spec["framework"],
            "family": model_spec["family"],
            "sourcePath": model_spec["sourcePath"],
            "inputShape": model_spec["inputShape"],
        },
        "testType": test_type,
        "testDataPath": {
            "requestedPath": requested_data_path,
            "resolvedPath": str(resolved_data_path),
        },
        "limit": limit,
    }


def evaluate_grouped_directory(
    model: torch.nn.Module,
    root_dir: Path,
    group_names: list[str],
    input_shape: list[int],
    class_labels: list[str],
    limit: int | None,
) -> list[dict]:
    results = []

    for group_name in group_names:
        group_dir = root_dir / group_name
        if not group_dir.is_dir():
            continue

        group_result = evaluate_image_directory(
            model=model,
            image_dir=group_dir,
            input_shape=input_shape,
            class_labels=class_labels,
            limit=limit,
        )
        results.append(
            {
                "name": group_name,
                "displayName": DISPLAY_NAMES.get(group_name, group_name),
                "path": str(group_dir),
                **group_result,
            }
        )

    if not results:
        expected = ", ".join(group_names)
        raise ValueError(f"No supported test folders found in {root_dir}. Expected one of: {expected}")

    return results


def evaluate_image_directory(
    model: torch.nn.Module,
    image_dir: Path,
    input_shape: list[int],
    class_labels: list[str],
    limit: int | None,
) -> dict:
    image_paths = list_image_files(image_dir)
    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise ValueError(f"No test images found in {image_dir}")

    correct = 0
    samples = []

    for image_path in image_paths:
        expected_label = infer_label_from_filename(image_path)
        if expected_label is None:
            raise ValueError(f"Cannot infer numeric label from filename: {image_path.name}")

        prediction = predict_image(
            model=model,
            image_path=image_path,
            input_shape=input_shape,
            class_labels=class_labels,
        )
        is_correct = prediction["index"] == expected_label
        correct += 1 if is_correct else 0

        if len(samples) < 25:
            samples.append(
                {
                    "fileName": image_path.name,
                    "expected": expected_label,
                    "predicted": prediction["index"],
                    "predictedLabel": prediction["label"],
                    "confidence": prediction["confidence"],
                    "correct": is_correct,
                }
            )

    total = len(image_paths)
    return {
        "accuracy": correct / total,
        "correct": correct,
        "incorrect": total - correct,
        "total": total,
        "samples": samples,
    }


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    input_shape: list[int],
    class_labels: list[str],
) -> dict:
    image_tensor, _input_array = load_image_tensor(image_path, input_shape)

    with torch.no_grad():
        logits = model(image_tensor).detach().cpu().reshape(-1)

    probabilities = torch.softmax(logits, dim=0)
    prediction_index = int(torch.argmax(probabilities).item())

    return {
        "index": prediction_index,
        "label": class_labels[prediction_index]
        if prediction_index < len(class_labels)
        else str(prediction_index),
        "confidence": float(probabilities[prediction_index].item()),
    }


def resolve_ai_test_data_dir(path_text: str | None, test_type: str) -> Path:
    if not path_text:
        path_text = test_type

    requested = Path(path_text)
    ai_root = AI_TEST_UI_ROOT.resolve()

    if requested.is_absolute():
        candidates = [requested]
    elif requested.parts and requested.parts[0].lower() == "generate_data":
        candidates = [AI_TEST_UI_ROOT / requested]
    else:
        candidates = [
            AI_TEST_UI_ROOT / "generate_data" / requested,
            AI_TEST_UI_ROOT / requested,
        ]

    resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())

    if resolved != ai_root and ai_root not in resolved.parents:
        raise ValueError(f"Test data path must stay inside AI_Test_UI: {path_text}")

    if not resolved.exists():
        raise ValueError(f"Test data directory does not exist: {path_text}")

    if not resolved.is_dir():
        raise ValueError(f"Test data path is not a directory: {path_text}")

    return resolved


def list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.name,
    )


def infer_label_from_filename(image_path: Path) -> int | None:
    match = re.search(r"_(\d+)$", image_path.stem)
    if match:
        return int(match.group(1))

    fallback = re.search(r"(\d)$", image_path.stem)
    return int(fallback.group(1)) if fallback else None


def normalize_test_type(test_type: str | None) -> str:
    resolved = (test_type or "accuracy").strip().lower()
    if resolved not in {"accuracy", "robustness", "adversary"}:
        raise ValueError(
            f"Unsupported testType: {test_type}. Supported values: accuracy, robustness, adversary"
        )

    return resolved


def normalize_limit(limit: int | str | None) -> int | None:
    if limit in (None, ""):
        return None

    try:
        resolved = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"limit must be a positive integer, got: {limit}") from exc

    if resolved < 1:
        raise ValueError(f"limit must be a positive integer, got: {limit}")

    return resolved


def build_group_output(group_results: list[dict]) -> str:
    return " ; ".join(
        f"{group['displayName']} accuracy: {group['accuracy']:.4f}"
        for group in group_results
    )
