"""Extract a reproducible gallery from the bundled Tiny ImageNet test split."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "public/assets/img"
LABELS = ["lifeboat", "ladybug", "pizza", "bell pepper", "school bus",
          "koala", "espresso", "red panda", "orange", "sport car"]


def pixels(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((64, 64)), dtype=float)


originals = [pixels(p.read_bytes()) for p in DEST.glob("*_1.jpeg")]
gallery = []
with zipfile.ZipFile(ROOT / "tiny-vgg/data.zip") as archive:
    labels = json.loads(archive.read("data/val_class_dict_10.json"))
    candidates = sorted(name for name in archive.namelist()
                        if name.startswith("data/class_10_val/test_images/") and name.endswith(".JPEG"))
    for index, label in enumerate(LABELS):
        count = 0
        for source in candidates:
            if labels[Path(source).name]["index"] != index:
                continue
            data = archive.read(source)
            sample = pixels(data)
            if any(np.abs(sample - old).mean() < 8 for old in originals):
                continue
            filename = "sample_" + Path(source).name
            (DEST / filename).write_bytes(data)
            originals.append(sample)
            gallery.append({"file": filename, "class": label, "source": source})
            count += 1
            if count == 2:
                break
        assert count == 2, label
(DEST / "gallery.json").write_text(json.dumps(gallery, indent=2) + "\n", encoding="utf-8")
print(f"Extracted {len(gallery)} distinct images from the archive test split.")
