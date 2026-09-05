from __future__ import annotations

from pathlib import Path

try:
    from config import AI_TEST_UI_ROOT, PROJECT_ROOT
except ImportError:  # pragma: no cover - used when imported as backend.services
    from ..config import AI_TEST_UI_ROOT, PROJECT_ROOT


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PUBLIC_ROOT = (PROJECT_ROOT / "public").resolve()


def resolve_ai_test_image_path(path_text: str | None) -> Path:
    return resolve_explain_image_path(path_text)


def resolve_explain_image_path(path_text: str | None) -> Path:
    if not path_text:
        path_text = "test_original/1_7.jpg"

    candidate = Path(path_text)
    resolved = resolve_candidate(candidate)
    allowed_roots = [AI_TEST_UI_ROOT.resolve(), PUBLIC_ROOT]

    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(
            "Image path must stay inside AI_Test_UI or project public assets: "
            f"{path_text}"
        )

    if not resolved.exists():
        raise ValueError(f"Image file does not exist: {path_text}")

    if not resolved.is_file():
        raise ValueError(f"Image path is not a file: {path_text}")

    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type: {resolved.suffix}")

    return resolved


def resolve_candidate(candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate.resolve()

    for root in (AI_TEST_UI_ROOT, PUBLIC_ROOT, PROJECT_ROOT):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (AI_TEST_UI_ROOT / candidate).resolve()
