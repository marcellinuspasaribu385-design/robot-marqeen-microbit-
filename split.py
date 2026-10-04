import random
from pathlib import Path
import shutil
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.2
SRC = Path("data/dataset_raw")
DST = Path("data/split")

random.seed(SEED)

classes = [p.name for p in SRC.iterdir() if p.is_dir()]
if not classes:
    raise RuntimeError("Dataset kosong. Isi data/dataset_raw/<kelas>/ terlebih dahulu.")

for cls in classes:
    files = [p for p in (SRC / cls).glob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if len(files) < 2:
        print(f"[WARNING] Kelas {cls} hanya memiliki {len(files)} file.")
        continue

    # Jika nama file diawali sessionXX_, file dikelompokkan berdasarkan sesi
    groups = defaultdict(list)
    for p in files:
        parts = p.stem.split("_")
        session = parts[0] if parts[0].lower().startswith("session") else p.stem
        groups[session].append(p)

    group_names = list(groups.keys())
    random.shuffle(group_names)

    target_val = max(1, int(len(files) * VAL_RATIO))
    val_groups, val_count = [], 0
    for g in group_names:
        if val_count >= target_val:
            break
        val_groups.append(g)
        val_count += len(groups[g])

    val_set = {p for g in val_groups for p in groups[g]}
    train_set = [p for p in files if p not in val_set]

    for split, items in [("train", train_set), ("val", list(val_set))]:
        out = DST / split / cls
        out.mkdir(parents=True, exist_ok=True)
        for p in items:
            shutil.copy2(p, out / p.name)

    print(f"{cls}: train={len(train_set)}, val={len(val_set)}")

print("Split selesai.")
