import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mask-glob", default=None, help="glob relativo a --root che seleziona le maschere")
    p.add_argument("--n", type=int, default=500, help="maschere da campionare")
    a = p.parse_args()
    
    counts = Counter()
    for f in a.root.rglob("*"):
        if f.is_file():
            folder = "/".join(f.relative_to(a.root).parts[:-1][:4]) or "."
            counts[(folder, f.suffix.lower())] += 1
    
    for (folder, ext), n in sorted(counts.items()):
        print(f"{n:8d}  {ext or '(no ext)':8s} {folder}")
    
    if not a.mask_glob:
        return

    files = sorted(a.root.glob(a.mask_glob))
    print(f"\nMaschere trovate: {len(files)}")
    random.seed(0)
    files = random.sample(files, min(a.n, len(files)))

    hist = np.zeros(256, dtype=np.int64)
    formats = Counter()
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if m is None:
            print("Illegibile:", f)
            continue
        formats[(m.shape, str(m.dtype))] += 1
        if m.ndim != 2:
            print("Maschera non a canale singolo:", f, m.shape)
            continue
        hist += np.bincount(m.ravel(), minlength=256)

    print("\nFormati (shape, dtype):", dict(formats))
    total = hist.sum()
    print("\nvalore pixel      %")
    for v in np.nonzero(hist)[0]:
        print(f"{v:6d}  {hist[v]:12d}  {100 * hist[v] / total:6.2f}")
    
    freq = hist[:4] / hist[:4].sum()
    w = 1 / np.sqrt(freq)
    w = w / w.mean()
    print("\nclass_weights suggeriti per config.yaml:", np.round(w, 3).tolist())


if __name__ == "__main__":
    main()
