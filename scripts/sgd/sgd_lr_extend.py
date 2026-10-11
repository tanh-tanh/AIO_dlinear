"""Mở rộng lưới lr ở các ô chọn mức biên (docs/tasks/NHIEM_VU_9.md, Phần A), chạy trên CPU bằng sgd_etth2.py.

    python scripts/sgd/sgd_lr_extend.py --dry-run           # in 15 ô biên và việc còn thiếu
    python scripts/sgd/sgd_lr_extend.py                     # ETTm1 trước, 5 tiến trình × 2 luồng như NHIEM_VU_7
    python scripts/sgd/sgd_lr_extend.py --max-new ETTh2=1   # hết ngân sách: chỉ một mức mới cho ETTh2

Ô biên lấy từ results/sgd_lr/selection.json (không sửa file đó): mức chọn ×8 (biên trên) hoặc ×0,5 (biên dưới).
Mỗi ô, như nhau cho mọi ô:
    1. Chạy mức kế tiếp theo hướng biên (×16 rồi ×32; ×0,25 rồi ×0,125) với seed chọn lr 2021–2025.
    2. Mức mới tốt hơn mức đang chọn (trung bình val_mse thấp hơn) thì thử thêm một mức; tối đa 2 mức mới.
       Dừng ngay khi mức mới không tốt hơn.
    3. lr chọn đổi thì chạy 15 seed cuối 2026–2040 ở lr mới; không đổi thì giữ kết quả cũ (results/sgd_lr/).
    4. Sau 2 mức mới vẫn chọn mức xa nhất: ô "còn ở biên".
    5. Phân kỳ (val_mse không hữu hạn, mô hình không lưu được checkpoint nào, hoặc val_mse > 10 × trung bình val
       của mức ×1): lần chạy đó thất bại; mức có lần thất bại không được chọn. Giữ trong JSON.
Test không dùng để chọn gì. Kết quả: results/sgd_lr_ext/<dataset>/x<mức>/, results/sgd_lr_ext/selection.json,
log ở results/sgd_lr_ext/logs/.
"""
import argparse
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / d) for d in ("grid", "sgd", "checks", "report")]
from bench_5060ti import git_commit, hw_info       # noqa: E402
from sgd_etth2 import HPS                         # noqa: E402
from sgd_lr_tune import SEL_SEEDS, FINAL_SEEDS, find as old_find   # noqa: E402

PY = sys.executable
OUT = ROOT / "results" / "sgd_lr_ext"
OLD_SEL = ROOT / "results" / "sgd_lr" / "selection.json"
ORDER = ["ETTm1", "ETTh1", "ETTh2"]               # ETTm1 đắt nhất, chạy trước (A.3)
NEXT = {8.0: [16.0, 32.0], 0.5: [0.25, 0.125]}
DIVERGE = 10.0
LOCK = threading.Lock()


def new_dir(ds, mult):
    return OUT / ds / f"x{mult:g}"


def find(ds, model, H, mult, seed):
    """JSON của một lần chạy (đúng lr): mức mới ở results/sgd_lr_ext/, mức cũ qua sgd_lr_tune.find."""
    if mult not in NEXT[8.0] + NEXT[0.5]:
        return old_find(ds, model, H, mult, seed)
    lr = HPS[ds]["lr"] * mult
    f = new_dir(ds, mult) / f"{model}_H{H}_seed{seed}.json"
    if f.exists():
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["dataset"] == ds and abs(r["lr"] - lr) <= 1e-12 * lr:
            return f, r
    return None


def fail_file(ds, model, H, mult, seed):
    return new_dir(ds, mult) / f"{model}_H{H}_seed{seed}.failed.json"


def call(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def run(ds, model, H, mult, seeds, threads, dry, log):
    missing = [s for s in seeds if find(ds, model, H, mult, s) is None and not fail_file(ds, model, H, mult, s).exists()]
    if not missing:
        return
    lr = HPS[ds]["lr"] * mult
    out = str(new_dir(ds, mult).relative_to(ROOT / "results")).replace("\\", "/")
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {ds} {model} H={H} ×{mult:g} (lr {lr:g}): {len(missing)} seed"
              + (" (dry-run)" if dry else ""), flush=True)
    if dry:
        return
    base = [PY, str(ROOT / "scripts" / "sgd" / "sgd_etth2.py"), "--step", "run", "--dataset", ds, "--model", model,
            "--H", str(H), "--lr", repr(lr), "--threads", str(threads), "--out", out]
    if call(base + ["--seed", *map(str, missing)], log) == 0:
        return
    # một seed lỗi (vd. NaN: không epoch nào có val hữu hạn nên không có checkpoint) làm dừng cả tiến trình:
    # chạy lại từng seed còn thiếu, seed vẫn lỗi ghi thành lần chạy thất bại
    for s in missing:
        if find(ds, model, H, mult, s) is not None:
            continue
        rc = call(base + ["--seed", str(s)], log)
        if rc != 0 and find(ds, model, H, mult, s) is None:
            fail_file(ds, model, H, mult, s).write_text(json.dumps(
                {"dataset": ds, "model": model, "H": H, "seed": s, "lr": lr, "mult": mult, "returncode": rc,
                 "reason": "sgd_etth2.py lỗi (xem log)", "log": str(log.relative_to(ROOT))},
                indent=1, ensure_ascii=False), encoding="utf-8")


def level(ds, model, H, mult, seeds, ref1):
    """Tóm tắt một mức: trung bình val (None nếu mức thất bại), file, các lần thất bại."""
    files, vals, failed = [], [], []
    for s in seeds:
        r = find(ds, model, H, mult, s)
        if r is None:
            ff = fail_file(ds, model, H, mult, s)
            failed.append({"seed": s, "reason": "lỗi khi chạy", "file": str(ff.relative_to(ROOT))})
            continue
        v = r[1]["val_mse"]
        files.append(str(r[0].relative_to(ROOT)))
        vals.append(v)
        if not math.isfinite(v) or v > DIVERGE * ref1:
            failed.append({"seed": s, "reason": f"val_mse {v:g} > {DIVERGE:g} × {ref1:.5f} (mức ×1) hoặc không hữu hạn",
                           "file": files[-1]})
    mean = float(np.mean(vals)) if vals and not failed else None
    return {"mean_val": mean, "files": files, "vals": vals, "failed": failed}


def cell(key, old, threads, dry, max_new):
    ds, model, H = old["dataset"], old["model"], old["H"]
    log = OUT / "logs" / f"{ds}_{model}_H{H}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    edge = old["chosen_mult"]
    side = "trên" if edge == 8.0 else "dưới"
    ref1 = old["mean_val_by_mult"]["1"]
    best, best_val = edge, old["mean_val_by_mult"][f"{edge:g}"]
    tried = {}
    for mult in NEXT[edge][:max_new]:
        run(ds, model, H, mult, SEL_SEEDS, threads, dry, log)
        if dry:
            break
        lv = level(ds, model, H, mult, SEL_SEEDS, ref1)
        tried[f"{mult:g}"] = lv
        if lv["mean_val"] is None or lv["mean_val"] >= best_val:
            break
        best, best_val = mult, lv["mean_val"]
    if dry:
        return None
    changed = best != edge
    final_files, final_mean_val = old["final_files"], old["final_mean_val"]
    if changed:
        run(ds, model, H, best, FINAL_SEEDS, threads, dry, log)
        fl = level(ds, model, H, best, FINAL_SEEDS, ref1)
        final_files, final_mean_val = fl["files"], fl["mean_val"]
        assert len(final_files) == len(FINAL_SEEDS), f"{key}: chạy cuối thiếu seed"
    still_edge = changed and best == NEXT[edge][-1] and len(tried) == len(NEXT[edge])
    res = {"dataset": ds, "model": model, "H": H, "base_lr": old["base_lr"], "edge": side,
           "old_chosen_mult": edge, "old_mean_val_by_mult": old["mean_val_by_mult"],
           "new_levels": {m: {"mean_val": t["mean_val"], "vals": t["vals"], "failed": t["failed"]}
                          for m, t in tried.items()},
           "sel_files": {m: t["files"] for m, t in tried.items()},
           "max_new_levels": max_new, "chosen_mult": best, "chosen_lr": old["base_lr"] * best, "changed": changed,
           "still_at_edge": still_edge, "at_edge_budget_cut": changed and best == NEXT[edge][max_new - 1]
           and max_new < len(NEXT[edge]),
           "final_seeds": FINAL_SEEDS, "final_files": final_files, "final_mean_val": final_mean_val}
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] xong {key}: ×{edge:g} → ×{best:g}"
              + (" (còn ở biên)" if still_edge else ""), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dataset", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--max-new", nargs="*", default=[], help="vd. ETTh2=1: số mức mới tối đa của một dataset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    max_new = {ds: 2 for ds in ORDER}
    max_new.update({k: int(v) for k, v in (a.split("=") for a in args.max_new)})
    cells = json.loads(OLD_SEL.read_text(encoding="utf-8"))["cells"]
    edges = {k: c for k, c in cells.items() if c["chosen_mult"] in NEXT}
    print(f"{len(edges)} ô biên trong {OLD_SEL.relative_to(ROOT)}")
    jobs = sorted(((k, c) for k, c in edges.items() if c["dataset"] in args.dataset),
                  key=lambda kc: (ORDER.index(kc[1]["dataset"]), kc[1]["model"], kc[1]["H"]))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(cell, k, c, args.threads, args.dry_run, max_new[c["dataset"]]) for k, c in jobs]
        results = [f.result() for f in futs]
    if args.dry_run:
        return
    f = OUT / "selection.json"
    old = json.loads(f.read_text(encoding="utf-8")).get("cells", {}) if f.exists() else {}
    old.update({k: r for (k, _), r in zip(jobs, results)})
    hw = hw_info()
    f.write_text(json.dumps({"hardware": {"cpu": hw["cpu"], "torch": hw["torch"], "os": hw["os"]},
                             "git_commit": git_commit(), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "workers": args.workers, "threads": args.threads, "n_edge_cells": len(edges),
                             "wall_seconds_last_call": time.time() - t0, "cells": old},
                            indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {f} ({(time.time() - t0) / 3600:.2f} giờ)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
