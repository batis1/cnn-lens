from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERPRETABILITY_ROOT = PROJECT_ROOT.parent
PROJECT_MODEL_ROOT = Path(
    os.environ.get("CNN_EXPLAINER_MODEL_ROOT", PROJECT_ROOT / "backend" / "models")
).resolve()

API_PREFIX = "/api"
HOST = os.environ.get("CNN_EXPLAINER_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("CNN_EXPLAINER_BACKEND_PORT", "8000"))
DEBUG = os.environ.get("CNN_EXPLAINER_BACKEND_DEBUG", "0") == "1"

AI_TEST_UI_ROOT = Path(
    os.environ.get("AI_TEST_UI_ROOT", INTERPRETABILITY_ROOT / "AI_Test_UI")
).resolve()
