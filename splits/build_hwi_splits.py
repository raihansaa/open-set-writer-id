"""
Usage (PowerShell):
    python build_hwi_splits.py `
        --images-root "image directory" `
        --out-root    "hwi_splits" `
        --pseudo-unknown-frac 0.30 `
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
KNOWN_PAGES = 5  # ScriptNet test-set writers have exactly 5 pages -> Known pool
# writer = leading digits; page = number after IMG_MAX_  (e.g. "1-IMG_MAX_960337.png")
WRITER_RE = re.compile(r"^(?P<writer>\d+).*?IMG_MAX_(?P<page>\d+)", re.IGNORECASE)


def parse_filename(name: str):
    """Return (original_writer_id_int, page_index_int) or None."""
    m = WRITER_RE.match(name)
    if not m:
        return None
    return int(m.group("writer")), int(m.group("page"))


def collect(root: Path):
    """Return {writer_id: [(page_index, rel_path_str)...]} sorted by page_index.

    Recurses (rglob) so a single root that contains sub-folders still works.
    """
    groups = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = parse_filename(p.name)
        if parsed is None:
            print(f"  SKIPPED (unparsed filename): {p.name}")
            continue
        writer, page = parsed
        rel = p.relative_to(root).as_posix()
        groups[writer].append((page, rel))
    for w in groups:
        groups[w].sort(key=lambda t: t[0])
    return dict(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=Path,
                    help="One HWI directory (flat or with sub-folders) containing all pages")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--pseudo-unknown-frac", type=float, default=0.30,
                    help="Fraction of the fewer-page (unknown-pool) writers used as Pseudo-unknown")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARSING SANITY CHECK")
    print("=" * 70)
    print(f"\nImages root: {args.images_root}")
    sample = [p for p in sorted(args.images_root.rglob("*"))
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS][:5]
    for p in sample:
        parsed = parse_filename(p.name)
        print(f"  {p.name}  -->  writer={parsed[0] if parsed else None}, page_idx={parsed[1] if parsed else None}")

    print("\n" + "=" * 70)
    print("COLLECTING")
    print("=" * 70)
    writers = collect(args.images_root)
    n_pages = sum(len(v) for v in writers.values())
    print(f"  Total writers: {len(writers)}  (expected 1114 = 720 + 394)")
    print(f"  Total pages:   {n_pages}  (expected 4782)")

    # Split writers by page count: 5-page -> Known, fewer-page -> unknown pool.
    known_writers = sorted([w for w, pages in writers.items() if len(pages) == KNOWN_PAGES])
    pool_writers  = sorted([w for w, pages in writers.items() if 0 < len(pages) < KNOWN_PAGES])
    odd_writers   = sorted([w for w, pages in writers.items() if len(pages) > KNOWN_PAGES])

    print(f"\n  {KNOWN_PAGES}-page writers (-> Known):       {len(known_writers)}  (expected 720)")
    print(f"  fewer-page writers (-> unknown pool): {len(pool_writers)}  (expected 394)")
    if odd_writers:
        print(f"  WARNING: {len(odd_writers)} writers have >{KNOWN_PAGES} pages "
              f"(possible merged-ID collision); excluded:")
        for w in odd_writers[:5]:
            print(f"    writer {w}: {len(writers[w])} pages")

    # Pseudo-unknown vs Unknown selection from the fewer-page pool (seeded, deterministic)
    rng = random.Random(args.seed)
    pool_shuffled = list(pool_writers)
    rng.shuffle(pool_shuffled)
    n_pseudo = math.floor(args.pseudo_unknown_frac * len(pool_shuffled))
    pseudo_unknown_writers = sorted(pool_shuffled[:n_pseudo])
    unknown_writers        = sorted(pool_shuffled[n_pseudo:])

    print(f"\n  Split of {len(pool_shuffled)} unknown-pool writers (frac={args.pseudo_unknown_frac}, floor):")
    print(f"    Pseudo-unknown: {len(pseudo_unknown_writers)} writers (expected 118)")
    print(f"    Unknown:        {len(unknown_writers)} writers (expected 276)")

    # Build CSV rows
    train_rows, val_rows, test_rows = [], [], []
    img_id = 0

    def add(rows, rel_path, writer_label):
        nonlocal img_id
        rows.append((img_id, f"hwi/{rel_path}", writer_label))
        img_id += 1

    # Known writers (5-page): 3 train / 1 val / 1 test
    for w in known_writers:
        pages = writers[w]
        prefixed_id = f"H{w:04d}"
        for _, rel in pages[:3]:
            add(train_rows, rel, prefixed_id)
        for _, rel in pages[3:4]:
            add(val_rows, rel, prefixed_id)
        for _, rel in pages[4:5]:
            add(test_rows, rel, prefixed_id)

    # Pseudo-unknown writers: ALL pages -> val.csv as -1
    for w in pseudo_unknown_writers:
        for _, rel in writers[w]:
            add(val_rows, rel, "-1")

    # Unknown writers: ALL pages -> test.csv as -1
    for w in unknown_writers:
        for _, rel in writers[w]:
            add(test_rows, rel, "-1")

    # Write CSVs
    columns = ["image_id", "image_path", "writer_id"]
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out = args.out_root / f"{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(columns)
            wr.writerows(rows)

    # Manifests: prefixed_id <TAB> original_id
    def write_manifest(fname, writer_ids):
        lines = "\n".join(f"H{w:04d}\t{w}" for w in sorted(writer_ids)) + "\n"
        (args.out_root / fname).write_text(lines, encoding="utf-8")

    write_manifest("writers_known.txt", known_writers)
    write_manifest("writers_pseudo_unknown.txt", pseudo_unknown_writers)
    write_manifest("writers_unknown.txt", unknown_writers)

    # Summary
    def counts(rows):
        k = sum(1 for r in rows if r[2] != "-1")
        n = sum(1 for r in rows if r[2] == "-1")
        return k, n

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'File':<12} {'known':>8} {'-1 rows':>10} {'total':>8}   {'expected':>20}"
    print("\n" + header)
    print("-" * len(header))
    expected = {"train.csv": "2160 / 0 / 2160", "val.csv": "720 / 354 / 1074", "test.csv": "720 / 828 / 1548"}
    for name, rows in [("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]:
        k, n = counts(rows)
        print(f"{name:<12} {k:>8} {n:>10} {len(rows):>8}   {expected[name]:>20}")

    print(f"\nFiles written under: {args.out_root.resolve()}/")
    for f in sorted(args.out_root.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
