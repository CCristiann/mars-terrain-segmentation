import argparse
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

SOL_SECONDS = 88775  # duration of a Martian sol in seconds
N_CLASSES = 4  # 0...3 are classes, 255 = no label


def group_key(stem, group_sols):
    m = re.search(r"(?<!\d)(\d{9,10})(?!\d)", stem)
    if m:
        return int(m.group(1)) // (SOL_SECONDS * group_sols)
    return stem  # no timestamp found: each image is its own group


def has_valid_pixels(mask_path):
    """True if the mask contains at least one labeled pixel (value < 4)"""
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    return m is not None and bool((m < N_CLASSES).any())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--group-sols",
        type=int,
        default=5,
        help="consecutive sols that always end up in the same split",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    names = sorted(
        f.name
        for f in a.img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (a.mask_dir / f"{f.stem}.png").exists()
    )

    # drop images whose mask is entirely 255: there is nothing to learn from them
    with ThreadPoolExecutor(8) as ex:
        ok = list(
            ex.map(
                lambda n: has_valid_pixels(a.mask_dir / f"{Path(n).stem}.png"), names
            )
        )

    n_empty = ok.count(False)
    names = [n for n, good in zip(names, ok) if good]
    print(f"Discarded {n_empty} images without any labeled pixels")

    groups = defaultdict(list)
    for n in names:
        groups[group_key(Path(n).stem, a.group_sols)].append(n)

    n_unparsed = sum(1 for k in groups if isinstance(k, str))
    print(
        f"{len(names)} images, {len(groups)} groups, "
        f"{n_unparsed} without a recognised timestamp"
    )

    keys = sorted(groups, key=str)
    random.Random(a.seed).shuffle(keys)

    val, train = [], []
    target = a.val_frac * len(names)
    for k in keys:
        (val if len(val) < target else train).extend(groups[k])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "train.txt").write_text("\n".join(sorted(train)) + "\n")
    (a.out_dir / "val.txt").write_text("\n".join(sorted(val)) + "\n")
    print(f"train: {len(train)}  val: {len(val)}")


if __name__ == "__main__":
    main()
