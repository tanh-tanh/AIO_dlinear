"""SGD đã tinh chỉnh learning rate (docs/NHIEM_VU_7.md, Phần D), chạy trên CPU bằng scripts/sgd_etth2.py.

    python scripts/sgd_lr_tune.py                       # cả 36 ô, 5 tiến trình song song, mỗi tiến trình 2 luồng
    python scripts/sgd_lr_tune.py --dry-run             # in danh sách việc còn thiếu
    python scripts/sgd_lr_tune.py --dataset ETTh2 --workers 3

Mỗi (dataset, mô hình, H), như nhau cho mọi ô:
    1. Chọn lr: lr ∈ {0,5; 1; 2; 4} × lr của script gốc (HPS của sgd_etth2.py), mỗi mức seed 2021–2025. Chọn mức
       có trung bình val_mse (float64, W_eff chấm trên đủ cửa sổ val) thấp nhất.
    2. Mức chọn là ×4 (đầu mút trên) thì thử thêm ×8 một lần với cùng 5 seed, rồi chọn lại trên 5 mức.
    3. Chạy cuối: seed 2026–2040 (15 seed mới) ở lr đã chọn. Mọi thống kê của "SGD đã tinh chỉnh" chỉ dùng 15 seed này.
Các tham số khác (batch, epoch, patience, lradj) giữ như script gốc. Test không dùng để chọn gì.

Dùng lại kết quả có sẵn (cùng code SGD; JSON phải ghi đúng lr): mức ×1 ở results/sgd_b1_b4/{ETTh1,ETTm1}/ và
results/sgd_etth2_20seed/ (ETTh2; NLinear chỉ có seed 2021–2030); Linear ETTm1 ×2, ×4 ở results/outliers/d2_lr/
(NHIEM_VU_6 D.2, seed 2021–2040, 2 luồng). Lần chạy mới ghi vào results/sgd_lr/<dataset>/x<mức>/.
Ghi results/sgd_lr/selection.json (trung bình val theo mức, mức chọn, nguồn từng seed), log ở results/sgd_lr/logs/.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import git_commit, hw_info       # noqa: E402
from sgd_etth2 import HPS                         # noqa: E402

PY = sys.executable
OUT = ROOT / "results" / "sgd_lr"
DATASETS = ["ETTh2", "ETTh1", "ETTm1"]            # thứ tự NHIEM_VU_7 D.2
MODELS = ["Linear", "DLinear", "NLinear"]
HORIZONS = [96, 192, 336, 720]
MULTS = [0.5, 1.0, 2.0, 4.0]
EXTRA = 8.0
SEL_SEEDS = list(range(2021, 2026))
FINAL_SEEDS = list(range(2026, 2041))
OLD = {"ETTh1": [ROOT / "results" / "sgd_b1_b4" / "ETTh1"], "ETTm1": [ROOT / "results" / "sgd_b1_b4" / "ETTm1"],
       "ETTh2": [ROOT / "results" / "sgd_etth2_20seed"]}
D2 = {2.0: ROOT / "results" / "outliers" / "d2_lr" / "ETTm1_lr0.0002",
      4.0: ROOT / "results" / "outliers" / "d2_lr" / "ETTm1_lr0.0004"}
LOCK = threading.Lock()


def new_dir(ds, mult):
    return OUT / ds / f"x{mult:g}"


def sources(ds, model, mult):
    dirs = [new_dir(ds, mult)]
    if mult == 1.0:
        dirs = OLD[ds] + dirs
    if ds == "ETTm1" and model == "Linear" and mult in D2:
        dirs = [D2[mult]] + dirs
    return dirs


def find(ds, model, H, mult, seed):
    """JSON của một lần chạy (đúng lr), hoặc None."""
    lr = HPS[ds]["lr"] * mult
    for d in sources(ds, model, mult):
        f = d / f"{model}_H{H}_seed{seed}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("dataset", "ETTh2") == ds and abs(r["lr"] - lr) <= 1e-12 * lr:
                return f, r
    return None


def run(ds, model, H, mult, seeds, threads, dry, log):
    missing = [s for s in seeds if find(ds, model, H, mult, s) is None]
    if not missing:
        return
    lr = HPS[ds]["lr"] * mult
    cmd = [PY, str(ROOT / "scripts" / "sgd_etth2.py"), "--step", "run", "--dataset", ds, "--model", model,
           "--H", str(H), "--seed", *map(str, missing), "--lr", repr(lr), "--threads", str(threads),
           "--out", str(new_dir(ds, mult).relative_to(ROOT / "results")).replace("\\", "/")]
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {ds} {model} H={H} ×{mult:g} (lr {lr:g}): {len(missing)} seed"
              + (" (dry-run)" if dry else ""), flush=True)
    if dry:
        return
    with open(log, "a", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def mean_val(ds, model, H, mult, seeds):
    rs = [find(ds, model, H, mult, s) for s in seeds]
    if any(r is None for r in rs):
        return None, []
    return float(np.mean([r[1]["val_mse"] for r in rs])), [str(r[0].relative_to(ROOT)) for r in rs]


def cell(ds, model, H, threads, dry):
    log = OUT / "logs" / f"{ds}_{model}_H{H}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for m in MULTS:
        run(ds, model, H, m, SEL_SEEDS, threads, dry, log)
    if dry:
        return None
    vals = {m: mean_val(ds, model, H, m, SEL_SEEDS) for m in MULTS}
    best = min(MULTS, key=lambda m: vals[m][0])
    extended = False
    if best == MULTS[-1]:
        run(ds, model, H, EXTRA, SEL_SEEDS, threads, dry, log)
        vals[EXTRA] = mean_val(ds, model, H, EXTRA, SEL_SEEDS)
        best = min(vals, key=lambda m: vals[m][0])
        extended = True
    run(ds, model, H, best, FINAL_SEEDS, threads, dry, log)
    final_val, final_files = mean_val(ds, model, H, best, FINAL_SEEDS)
    res = {"dataset": ds, "model": model, "H": H, "base_lr": HPS[ds]["lr"],
           "mean_val_by_mult": {f"{m:g}": v[0] for m, v in vals.items()},
           "sel_files": {f"{m:g}": v[1] for m, v in vals.items()},
           "chosen_mult": best, "chosen_lr": HPS[ds]["lr"] * best, "extended_x8": extended,
           "at_lower_edge": best == MULTS[0], "final_seeds": FINAL_SEEDS, "final_files": final_files,
           "final_mean_val": final_val}
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] xong {ds} {model} H={H}: chọn ×{best:g}"
              + (" (đã thử ×8)" if extended else ""), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--model", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--H", nargs="+", type=int, default=HORIZONS)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = [(ds, m, H) for ds in DATASETS if ds in args.dataset for m in args.model for H in args.H]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(cell, ds, m, H, args.threads, args.dry_run) for ds, m, H in jobs]
        results = [f.result() for f in futs]
    if args.dry_run:
        return
    f = OUT / "selection.json"
    old = json.loads(f.read_text(encoding="utf-8")).get("cells", {}) if f.exists() else {}
    old.update({f"{r['dataset']} {r['model']} H={r['H']}": r for r in results})
    hw = hw_info()
    f.write_text(json.dumps({"hardware": {"cpu": hw["cpu"], "torch": hw["torch"], "os": hw["os"]},
                             "git_commit": git_commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "workers": args.workers, "threads": args.threads,
                             "wall_seconds_last_call": time.time() - t0, "cells": old},
                            indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {f} ({(time.time() - t0) / 3600:.2f} giờ)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
