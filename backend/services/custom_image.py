from __future__ import annotations

import base64
import hashlib
import ipaddress
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

try:
    from config import PROJECT_ROOT
except ImportError:  # pragma: no cover - used when imported as backend.services
    from ..config import PROJECT_ROOT


MAX_IMAGE_BYTES = 10 * 1024 * 1024
CUSTOM_IMAGE_ROOT = PROJECT_ROOT / "public" / "custom-inputs"
FORMAT_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png"}


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def prepare_custom_image(source: str) -> dict[str, str]:
    if not source or not isinstance(source, str):
        raise ValueError("Choose an image file or enter an image link.")

    if source.startswith("data:image/"):
        image_bytes = _decode_data_url(source)
    else:
        image_bytes = _download_image(source)

    image_format = _validate_image(image_bytes)
    suffix = FORMAT_SUFFIXES.get(image_format)
    if suffix is None:
        raise ValueError("Use a PNG or JPEG image.")

    CUSTOM_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(image_bytes).hexdigest()[:20]
    output_path = CUSTOM_IMAGE_ROOT / f"input-{digest}{suffix}"
    if not output_path.exists():
        output_path.write_bytes(image_bytes)

    relative_path = output_path.relative_to(PROJECT_ROOT / "public").as_posix()
    return {"imagePath": relative_path}


def _decode_data_url(source: str) -> bytes:
    try:
        header, encoded = source.split(",", 1)
        if ";base64" not in header:
            raise ValueError
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("The uploaded image could not be read.") from exc

    _check_size(image_bytes)
    return image_bytes


def _download_image(source: str) -> bytes:
    _validate_remote_url(source)
    request = Request(source, headers={"User-Agent": "CNN-Lens/1.0"})
    opener = build_opener(_SafeRedirectHandler())

    try:
        with opener.open(request, timeout=12) as response:
            final_url = response.geturl()
            _validate_remote_url(final_url)
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise ValueError("The link does not point to an image.")
            image_bytes = response.read(MAX_IMAGE_BYTES + 1)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The image link could not be loaded.") from exc

    _check_size(image_bytes)
    return image_bytes


def _validate_remote_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete HTTP or HTTPS image link.")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("Private or local image links are not allowed.")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return

    if not ip.is_global:
        raise ValueError("Private or local image links are not allowed.")


def _validate_image(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            return image.format or ""
    except Exception as exc:
        raise ValueError("The selected file is not a valid image.") from exc


def _check_size(image_bytes: bytes) -> None:
    if not image_bytes:
        raise ValueError("The image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Images must be smaller than 10 MB.")
