from pathlib import Path

from PIL import Image


def make_state_farm_fixture(root: Path, images_per_class: int, subjects: int) -> Path:
    train = root / "train"
    for class_id in range(10):
        class_dir = train / f"c{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            image = Image.new("RGB", (32, 24), (class_id * 20, index, 10))
            image.save(class_dir / f"img_{class_id}_{index}.jpg")
    rows = [
        {
            "subject": f"p{index % subjects:02d}",
            "classname": f"c{class_id}",
            "img": f"img_{class_id}_{index}.jpg",
        }
        for class_id in range(10)
        for index in range(images_per_class)
    ]
    import csv

    with (root / "driver_imgs_list.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "classname", "img"])
        writer.writeheader()
        writer.writerows(rows)
    return root
