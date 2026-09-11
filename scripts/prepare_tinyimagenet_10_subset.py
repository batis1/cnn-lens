from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "tiny-vgg" / "tiny-imagenet-200" / "tiny-imagenet-200"
OUTPUT_ROOT = PROJECT_ROOT / "tiny-vgg" / "data"

SELECTED_CLASSES = [
    ("n01443537", "goldfish"),
    ("n02123045", "tabby cat"),
    ("n02106662", "German shepherd"),
    ("n02279972", "monarch butterfly"),
    ("n07753592", "banana"),
    ("n07768694", "pomegranate"),
    ("n02917067", "bullet train"),
    ("n02814860", "lighthouse"),
    ("n04356056", "sunglasses"),
    ("n04070727", "refrigerator"),
]


def read_validation_annotations() -> dict[str, str]:
    annotations_path = SOURCE_ROOT / "val" / "val_annotations.txt"
    image_to_wnid: dict[str, str] = {}
    with annotations_path.open("r", encoding="utf-8") as file:
        for line in file:
            image_name, wnid, *_ = line.strip().split("\t")
            image_to_wnid[image_name] = wnid
    return image_to_wnid


def copy_training_images(class_dict: dict[str, dict[str, int | str]]) -> None:
    for wnid in class_dict:
        source_dir = SOURCE_ROOT / "train" / wnid
        destination_dir = OUTPUT_ROOT / "class_10_train" / wnid
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet train class: {source_dir}")
        shutil.copytree(source_dir, destination_dir)


def copy_validation_and_test_images(
    class_dict: dict[str, dict[str, int | str]],
    image_to_wnid: dict[str, str],
) -> dict[str, dict[str, int | str]]:
    val_source_dir = SOURCE_ROOT / "val" / "images"
    val_destination_dir = OUTPUT_ROOT / "class_10_val" / "val_images"
    test_destination_dir = OUTPUT_ROOT / "class_10_val" / "test_images"
    val_destination_dir.mkdir(parents=True)
    test_destination_dir.mkdir(parents=True)

    per_class_seen = {wnid: 0 for wnid in class_dict}
    val_class_dict: dict[str, dict[str, int | str]] = {}

    for image_name, wnid in sorted(image_to_wnid.items()):
        if wnid not in class_dict:
            continue

        source_path = val_source_dir / image_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet validation image: {source_path}")

        destination_dir = val_destination_dir if per_class_seen[wnid] < 25 else test_destination_dir
        shutil.copy2(source_path, destination_dir / image_name)
        per_class_seen[wnid] += 1
        val_class_dict[image_name] = {
            "class": class_dict[wnid]["class"],
            "index": class_dict[wnid]["index"],
        }

    missing = {wnid: count for wnid, count in per_class_seen.items() if count != 50}
    if missing:
        raise ValueError(f"Expected 50 validation images per selected class, got: {missing}")

    return val_class_dict


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Tiny ImageNet source folder was not found: {SOURCE_ROOT}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    class_dict = {
        wnid: {"class": label, "index": index}
        for index, (wnid, label) in enumerate(SELECTED_CLASSES)
    }
    image_to_wnid = read_validation_annotations()

    copy_training_images(class_dict)
    val_class_dict = copy_validation_and_test_images(class_dict, image_to_wnid)

    (OUTPUT_ROOT / "class_dict_10.json").write_text(
        json.dumps(class_dict, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "val_class_dict_10.json").write_text(
        json.dumps(val_class_dict, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_ROOT}")
    print("train images:", len(list((OUTPUT_ROOT / "class_10_train").glob("*/images/*.JPEG"))))
    print("validation images:", len(list((OUTPUT_ROOT / "class_10_val" / "val_images").glob("*.JPEG"))))
    print("test images:", len(list((OUTPUT_ROOT / "class_10_val" / "test_images").glob("*.JPEG"))))


if __name__ == "__main__":
    main()
