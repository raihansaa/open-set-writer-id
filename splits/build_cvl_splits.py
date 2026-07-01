"""
Usage (PowerShell):
    python build_cvl_splits.py `
        --images-root "image directory" `
        --out-root    "cvl_splits" `
        --pseudo-unknown-frac 0.05 `
        --seed 0
"""

import argparse
import csv
import math
import random
import re
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
WRITER_RE = re.compile(r"^(?P<writer>\d+)-(?P<page>\d+)-cropped", re.IGNORECASE)


def parse_filename(name: str):
    """Return (original_writer_id_int, page_index_int) or None."""
    m = WRITER_RE.match(name)
    if not m:
        return None
    return int(m.group("writer")), int(m.group("page"))


def collect(folder: Path):
    """Return {original_writer_id: [(page_index, original_name)...]} sorted by page_index."""
    groups = defaultdict(list)
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_filename(p.name)
        if parsed is None:
            print(f"  SKIPPED (unparsed filename): {p.name}")
            continue
        writer, page = parsed
        groups[writer].append((page, p.name))
    for w in groups:
        groups[w].sort(key=lambda t: t[0])
    return dict(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--pseudo-unknown-frac", type=float, default=0.05,
                    help="Fraction of 5-page writers to use as Pseudo-unknown")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARSING SANITY CHECK")
    print("=" * 70)
    print(f"\nImages root: {args.images_root}")
    sample = [p for p in sorted(args.images_root.iterdir())
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS][:5]
    for p in sample:
        parsed = parse_filename(p.name)
        print(f"  {p.name}  -->  writer={parsed[0] if parsed else None}, page_idx={parsed[1] if parsed else None}")

    print("\n" + "=" * 70)
    print("COLLECTING")
    print("=" * 70)
    writers = collect(args.images_root)
    n_writers = len(writers)
    n_pages = sum(len(v) for v in writers.values())
    print(f"  Total writers: {n_writers}")
    print(f"  Total pages:   {n_pages}")

    # Split writers by page count
    seven_page_writers = sorted([w for w, pages in writers.items() if len(pages) == 7])
    five_page_writers  = sorted([w for w, pages in writers.items() if len(pages) == 5])
    other_writers      = sorted([w for w, pages in writers.items() if len(pages) not in (5, 7)])

    print(f"\n  7-page writers (-> Unknown): {len(seven_page_writers)}  (expected 27)")
    print(f"  5-page writers:               {len(five_page_writers)}  (expected 283)")
    if other_writers:
        print(f"  WARNING: {len(other_writers)} writers have unexpected page counts:")
        for w in other_writers[:5]:
            print(f"    writer {w}: {len(writers[w])} pages")

    # Pseudo-unknown selection from 5-page writers
    rng = random.Random(args.seed)
    five_shuffled = list(five_page_writers)
    rng.shuffle(five_shuffled)
    n_pseudo = math.floor(args.pseudo_unknown_frac * len(five_shuffled))
    pseudo_unknown_writers = sorted(five_shuffled[:n_pseudo])
    known_writers          = sorted(five_shuffled[n_pseudo:])

    print(f"\n  Split of 5-page writers (frac={args.pseudo_unknown_frac}, floor):")
    print(f"    Pseudo-unknown: {len(pseudo_unknown_writers)} writers (expected ~14)")
    print(f"    Known:          {len(known_writers)} writers")

    # Build CSV rows
    train_rows, val_rows, test_rows = [], [], []
    img_id = 0

    def add(rows, name, writer_label):
        nonlocal img_id
        rel_path = f"cvl/{name}"
        rows.append((img_id, rel_path, writer_label))
        img_id += 1

    # Known writers (5-page): 3 train / 1 val / 1 test
    for w in known_writers:
        pages = writers[w]
        if len(pages) != 5:
            print(f"  WARNING: Known writer {w} has {len(pages)} pages; skipping.")
            continue
        prefixed_id = f"C{w:04d}"
        train_pages = pages[:3]
        val_pages   = pages[3:4]
        test_pages  = pages[4:5]
        for _, name in train_pages:
            add(train_rows, name, prefixed_id)
        for _, name in val_pages:
            add(val_rows, name, prefixed_id)
        for _, name in test_pages:
            add(test_rows, name, prefixed_id)

    # Pseudo-unknown writers: ALL pages -> val.csv as -1
    for w in pseudo_unknown_writers:
        for _, name in writers[w]:
            add(val_rows, name, "-1")

    # Unknown writers (the 27 seven-page writers): ALL pages -> test.csv as -1
    for w in seven_page_writers:
        for _, name in writers[w]:
            add(test_rows, name, "-1")

    # Write CSVs
    columns = ["image_id", "image_path", "writer_id"]
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out = args.out_root / f"{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)

    # Manifests (prefixed_id <TAB> original_id)
    def manifest_lines(writer_ids):
        return "\n".join(f"C{w:04d}\t{w}" for w in sorted(writer_ids)) + "\n"

    (args.out_root / "writers_known.txt").write_text(
        manifest_lines(known_writers), encoding="utf-8")
    (args.out_root / "writers_pseudo_unknown.txt").write_text(
        manifest_lines(pseudo_unknown_writers), encoding="utf-8")
    (args.out_root / "writers_unknown.txt").write_text(
        manifest_lines(seven_page_writers), encoding="utf-8")

    # Summary
    def counts(rows):
        k = sum(1 for r in rows if r[2] != "-1")
        n = sum(1 for r in rows if r[2] == "-1")
        return k, n

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'File':<12} {'known':>8} {'-1 rows':>10} {'total':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, rows in [("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]:
        k, n = counts(rows)
        print(f"{name:<12} {k:>8} {n:>10} {len(rows):>8}")

    print(f"\nFiles written under: {args.out_root.resolve()}/")
    for f in sorted(args.out_root.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
