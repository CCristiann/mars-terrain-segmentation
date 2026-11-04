import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

SOL_SECONDS = 88775  # durata di un sol marziano in secondi


def group_key(stem, group_sols):
    m = re.search(r"(?<!\d)(\d{9,10})(?!\d)", stem)
    if m:
        return int(m.group(1)) // (SOL_SECONDS * group_sols)
    return stem  # timestamp non trovate: ogni immagine è un gruppo a sé


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
        help="sol consecutivi che finiscono sempre nello stesso split",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    names = sorted(
        f.name
        for f in a.img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (a.mask_dir / f"{f.stem}.png").exists()
    )

    groups = defaultdict(list)
    for n in names:
        groups[group_key(Path(n).stem, a.group_sols)].append(n)

    n_unparsed = sum(1 for k in groups if isinstance(k, str))
    print(
        f"{len(names)} immagini, {len(groups)} gruppi, "
        f"{n_unparsed} senza timestamp riconosciuto"
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
