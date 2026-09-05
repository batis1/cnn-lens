from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from config import AI_TEST_UI_ROOT
except ImportError:  # pragma: no cover - used when imported as backend.services
    from ..config import AI_TEST_UI_ROOT


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SUPPORTED_OPTIONS = {
    "noise": {"gauss_noise", "poisson_noise", "salt_pepper_noise"},
    "transform": {"rotation", "scale", "translation"},
}


def generate_test_data(
    data_path: str | None,
    output_path: str | None,
    generation_method: str | None,
    option: str | None,
) -> dict:
    method = (generation_method or "").strip().lower()
    selected_option = (option or "").strip().lower()
    if method not in SUPPORTED_OPTIONS or selected_option not in SUPPORTED_OPTIONS[method]:
        raise ValueError(f"Unsupported generation method: {method}/{selected_option}")

    source_dir = _resolve_directory(data_path, must_exist=True)
    destination_dir = _resolve_directory(output_path, must_exist=False)
    if source_dir == destination_dir:
        raise ValueError("The source and output folders must be different.")

    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            transformed = _transform_image(image, selected_option)
            transformed.save(destination_dir / image_path.name)
            generated += 1

    return {
        "success": True,
        "generationMethod": method,
        "option": selected_option,
        "sourcePath": _relative_display_path(source_dir),
        "outputPath": _relative_display_path(destination_dir),
        "generated": generated,
        "output": f"Generated {generated} images in {_relative_display_path(destination_dir)}",
    }


def _resolve_directory(path_text: str | None, *, must_exist: bool) -> Path:
    if not path_text:
        raise ValueError("A source and output folder must be selected.")

    requested = Path(path_text.replace("\\", "/").removeprefix("./"))
    candidate = requested if requested.is_absolute() else AI_TEST_UI_ROOT / requested
    resolved = candidate.resolve()
    try:
        resolved.relative_to(AI_TEST_UI_ROOT)
    except ValueError as exc:
        raise ValueError("Data folders must be inside the configured AI_Test_UI root.") from exc

    if must_exist and not resolved.is_dir():
        raise ValueError(f"Data folder does not exist: {path_text}")
    return resolved


def _relative_display_path(path: Path) -> str:
    return "./" + path.relative_to(AI_TEST_UI_ROOT).as_posix()


def _transform_image(image: Image.Image, option: str) -> Image.Image:
    original_mode = image.mode
    working = image.convert("RGB") if image.mode not in {"L", "RGB", "RGBA"} else image.copy()
    pixels = np.asarray(working).astype(np.float32)

    if option == "gauss_noise":
        output = np.clip(pixels + np.random.normal(0, 25, pixels.shape), 0, 255)
        return Image.fromarray(output.astype(np.uint8), mode=working.mode).convert(original_mode)
    if option == "poisson_noise":
        normalized = pixels / 255.0
        output = np.clip(np.random.poisson(normalized * 255.0), 0, 255)
        return Image.fromarray(output.astype(np.uint8), mode=working.mode).convert(original_mode)
    if option == "salt_pepper_noise":
        output = pixels.copy()
        height, width = output.shape[:2]
        count = max(1, int(height * width * 0.05))
        salt_y = np.random.randint(0, height, count)
        salt_x = np.random.randint(0, width, count)
        pepper_y = np.random.randint(0, height, count)
        pepper_x = np.random.randint(0, width, count)
        output[salt_y, salt_x] = 255
        output[pepper_y, pepper_x] = 0
        return Image.fromarray(output.astype(np.uint8), mode=working.mode).convert(original_mode)
    if option == "rotation":
        angle = float(np.random.uniform(-30, 30))
        return working.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0).convert(original_mode)
    if option == "scale":
        scale = float(np.random.uniform(0.9, 0.99))
        width, height = working.size
        resized = working.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new(working.mode, working.size, 0)
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas.convert(original_mode)
    if option == "translation":
        width, height = working.size
        offset = (round(width * 0.1), round(height * 0.1))
        return working.transform(
            working.size,
            Image.Transform.AFFINE,
            (1, 0, -offset[0], 0, 1, -offset[1]),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        ).convert(original_mode)

    raise ValueError(f"Unsupported generation option: {option}")
