"""
Checks the structural invariants that must hold for ANY valid open-set split,
making NO assumption about writer-ID prefixes (C / H / T / ...) or image_path
roots (cvl/, hwi_test/, ...). It auto-detects the known population and the
per-writer enroll/val/test allocation and reports the derived structure, so the
same script verifies CVL, HWI, or any future dataset without edits.

"""
import csv
import sys
from collections import Counter
from pathlib import Path

UNKNOWN = "-1"
REQUIRED = {"image_id", "image_path", "writer_id"}


def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def known_ids(rows):
    return set(r["writer_id"] for r in rows if r["writer_id"] != UNKNOWN)


def load_manifest(p):
    if not p.exists():
        return None
    return set(line.split("\t")[0].strip()
              for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def verify(dirpath):
    d = Path(dirpath)
    checks = []  # (ok, name, detail)

    def chk(ok, name, detail=""):
        checks.append((bool(ok), name, str(detail)))

    for n in ("train.csv", "val.csv", "test.csv"):
        if not (d / n).exists():
            chk(False, f"{n} exists", "MISSING")
            return checks, {}
    tr, va, te = (load_csv(d / n) for n in ("train.csv", "val.csv", "test.csv"))
    splits = {"train": tr, "val": va, "test": te}

    # 1. columns
    for nm, rows in splits.items():
        cols = set(rows[0].keys()) if rows else set()
        chk(REQUIRED <= cols, f"{nm}.csv has required columns", sorted(cols))

    # 2. no page leakage
    allp = [r["image_path"] for rows in splits.values() for r in rows]
    dup = sum(1 for _, c in Counter(allp).items() if c > 1)
    chk(dup == 0, "no image_path appears in >1 split (no leakage)", f"{dup} duplicated")

    # 3. image_id globally unique
    ids = [r["image_id"] for rows in splits.values() for r in rows]
    chk(len(ids) == len(set(ids)), "image_id globally unique", f"{len(ids) - len(set(ids))} dup")

    # 4. known set identical across splits
    kt, kv, ke = known_ids(tr), known_ids(va), known_ids(te)
    chk(kt == kv == ke, "known-writer set identical in train/val/test",
        f"train={len(kt)} val={len(kv)} test={len(ke)}")
    known = kt

    # 5. enrollment known-only
    n_unk_tr = sum(1 for r in tr if r["writer_id"] == UNKNOWN)
    chk(n_unk_tr == 0, "train.csv has zero unknown(-1) rows", f"found {n_unk_tr}")

    # 6. uniform per-known-writer allocation (auto-detected)
    def per(rows):
        return Counter(r["writer_id"] for r in rows if r["writer_id"] != UNKNOWN)
    ctr, cva, cte = per(tr), per(va), per(te)
    alloc = Counter((ctr.get(w, 0), cva.get(w, 0), cte.get(w, 0)) for w in known)
    top = alloc.most_common(1)[0][0] if alloc else (0, 0, 0)
    chk(len(alloc) == 1,
        f"every known writer has identical allocation (enroll/val/test = {top[0]}/{top[1]}/{top[2]})",
        f"{len(alloc)} distinct allocations: {dict(alloc)}")

    # 7. manifests (optional)
    mk = load_manifest(d / "writers_known.txt")
    mp = load_manifest(d / "writers_pseudo_unknown.txt")
    mu = load_manifest(d / "writers_unknown.txt")
    if mk is not None:
        chk(mk == known, "writers_known.txt matches CSV known set",
            f"manifest={len(mk)} csv={len(known)} sym_diff={len(mk ^ known)}")
    pools = {"known": mk, "pseudo": mp, "unknown": mu}
    present = {k: v for k, v in pools.items() if v is not None}
    if len(present) >= 2:
        names = list(present)
        overlaps = {f"{a}&{b}": len(present[a] & present[b])
                    for i, a in enumerate(names) for b in names[i + 1:]}
        chk(all(v == 0 for v in overlaps.values()),
            "known/pseudo/unknown manifests mutually disjoint", overlaps)

    structure = {
        "rows": {k: len(v) for k, v in splits.items()},
        "known_writers": len(known),
        "allocation": f"{top[0]}/{top[1]}/{top[2]}",
        "unknown_pages": {"val": sum(1 for r in va if r["writer_id"] == UNKNOWN),
                          "test": sum(1 for r in te if r["writer_id"] == UNKNOWN)},
        "manifests": {k: (len(v) if v is not None else None) for k, v in pools.items()},
    }
    return checks, structure


def main():
    dirs = sys.argv[1:] or ["cvl_splits", "hwi_splits"]
    all_ok = True
    for dirpath in dirs:
        checks, structure = verify(dirpath)
        print("=" * 64)
        print(f"  VERIFY: {dirpath}")
        print("=" * 64)
        for ok, name, detail in checks:
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name:<52} {detail if not ok else ''}")
        npass = sum(1 for ok, _, _ in checks if ok)
        if structure:
            print(f"  -- structure: rows={structure['rows']} known={structure['known_writers']} "
                  f"alloc={structure['allocation']} unknown_pages={structure['unknown_pages']} "
                  f"manifests={structure['manifests']}")
        ok_all = npass == len(checks)
        all_ok &= ok_all
        print(f"  RESULT: {npass}/{len(checks)} passed" + ("" if ok_all else "  <-- FAILURES") + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
