"""Ablation δ theo từng bước h cho Huber (docs/tasks/NHIEM_VU_9.md, C.3), chạy trên GPU.

    python scripts/checks/huber_delta_h.py                  # ETTh2 H = 720 Linear, ETTh1 H = 96 Linear
    python scripts/checks/huber_delta_h.py --cell ETTh1:96:Linear

Cách làm (góp ý của TA): chuẩn hóa phần dư theo σ̂_h của từng bước rồi dùng một δ chung trên thang đã chuẩn hóa,
tương đương δ_h = 1,345·σ̂_h, σ̂_h = 1,4826·MAD của phần dư train ở bước h của nghiệm MSE λ = 0 (Cell.huber_delta với
residual_stats=True, cùng phép tính của scripts/grid/huber_deltas.py). irls nhận δ là vectơ [H]: hàm mục tiêu
Σ_h Σ_i ρ_{δ_h}(r_ih)/δ_h + λ Σ_h w_hᵀ Pen w_h (src/solvers.py).

Đường λ như grid (scripts/grid/runner.py): λ ∈ IRLS_LAMS tăng dần, warm start từ λ trước, λ = 0 khởi tạo từ nghiệm
MSE λ = 0 (Cell.closed_form); chế độ pha (A_h float32, phần còn lại float64), chunk của ô, tol = 1e-9, patience 1;
mở rộng tối đa 2 giá trị nếu λ* theo val_mse hoặc val_mae ở đầu mút. val_huber tính bằng δ_h.
δ_MAD chung lấy từ results/huber_mad/lambda_path.csv (cùng thiết lập, irls vô hướng trùng từng bit với bản trước).

Kết quả: results/huber_delta_h/lambda_path.csv (mỗi λ một dòng), deltas.csv (δ_h theo h), summary.json.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("grid", "sgd", "checks", "report")]
import runner as R                                    # noqa: E402  (đặt PYTORCH_CUDA_ALLOC_CONF, tắt TF32)
import bench_5060ti as B                              # noqa: E402
from bench_5060ti import ROOT, git_commit, hw_info    # noqa: E402
from src.solvers import HUBER_K, irls                 # noqa: E402

OUT = ROOT / "results" / "huber_delta_h"
CELLS = [("ETTh2", 720, "Linear"), ("ETTh1", 96, "Linear")]
TOL = 1e-9
COLS = ["dataset", "H", "model", "objective", "lam", "delta_rule", "delta_mean", "delta_min", "delta_max",
        "train_obj", "val_mse", "val_mae", "val_huber", "test_mse", "test_mae", "n_iter", "converged", "seconds",
        "init", "tol", "patience", "chunk", "gram_dtype", "n_fallback", "gpu", "torch_version", "git_commit",
        "timestamp"]


def metrics(cell, W, model, lam, delta):
    """Như Cell.evaluate của runner, với δ là vectơ [H] (phạt chia δ_h theo cột)."""
    d = cell.d
    Rt = d["Y"] - d["Xt"] @ W.T
    a = np.abs(Rt)
    q = np.minimum(a, delta)
    loss = (q * (a - q / 2) / delta).sum()
    pen = lam * np.einsum("hi,ij,hj->", W[:, :-1], cell.pens[model], W[:, :-1]) if lam else 0.0
    out = {"train_obj": (loss + pen) / Rt.size}
    for tag, name in (("va", "val"), ("te", "test")):
        Rs = d[f"Y_{tag}"] - d[f"Xt_{tag}"] @ W.T
        out[f"{name}_mse"] = float((Rs ** 2).mean())
        out[f"{name}_mae"] = float(np.abs(Rs).mean())
        if name == "val":
            out["val_huber"] = B.huber_np(Rs, delta)
    return out


def run_cell(ds, H, model, hw, commit):
    R.HUBER_RULE.update(delta="mad", scale=1.0)
    cell = R.Cell(ds, H)
    delta_mad, sigma, _, sigma_h = cell.huber_delta(model, residual_stats=True)
    delta_h = HUBER_K * np.asarray(sigma_h)
    print(f"[{ds} H={H} {model}] δ_MAD = {delta_mad:.5f}; δ_h từ {delta_h.min():.5f} đến {delta_h.max():.5f} "
          f"(trung bình {delta_h.mean():.5f})", flush=True)
    rows, Ws = {}, {}

    def one(lam):
        below = [l for l in rows if l < lam]
        W0, init = (cell.closed_form(model, 0), "ols") if lam == 0 else (Ws[max(below)], "warm")
        W, n, conv, sec = cell.run_irls(model, W0, delta_h, lam, TOL, 1)
        row = {"dataset": ds, "H": H, "model": model, "objective": "Huber", "lam": lam,
               "delta_rule": "1.345*sigma_h (MAD theo h)", "delta_mean": float(delta_h.mean()),
               "delta_min": float(delta_h.min()), "delta_max": float(delta_h.max()),
               **metrics(cell, W, model, lam, delta_h), "n_iter": n, "converged": conv, "seconds": sec,
               "init": init, "tol": TOL, "patience": 1, "chunk": cell.chunk, "gram_dtype": "float32",
               "n_fallback": irls.last_fallbacks, "gpu": hw.get("gpu", ""), "torch_version": torch.__version__,
               "git_commit": commit, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rows[lam], Ws[lam] = row, W
        print(f"  λ = {lam:g}: {n} vòng, {sec:.1f} s, val MSE {row['val_mse']:.6f} MAE {row['val_mae']:.6f}, "
              f"test MSE {row['test_mse']:.6f} MAE {row['test_mae']:.6f}", flush=True)

    for lam in R.IRLS_LAMS:
        one(lam)
    for _ in range(R.MAX_EXTEND):
        lams = sorted(rows)
        new = None
        for metric in ("val_mse", "val_mae"):
            best = min(lams, key=lambda l: rows[l][metric])
            if best == lams[-1]:
                new = lams[-1] * 3
                break
            if best == lams[0]:
                new = [l for l in lams if l > 0][0] / 3
                break
        if new is None:
            break
        print(f"  λ* ở đầu mút: mở rộng λ = {new:g}", flush=True)
        one(new)
    cell.free()
    return [rows[l] for l in sorted(rows)], {"delta_mad": delta_mad, "sigma_mad": sigma,
                                             "sigma_h": [float(s) for s in sigma_h],
                                             "delta_h": [float(d) for d in delta_h]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs="+", default=[f"{d}:{h}:{m}" for d, h, m in CELLS])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hw, commit = hw_info(), git_commit()
    t0 = time.time()
    all_rows, deltas = [], {}
    for c in args.cell:
        ds, H, model = c.split(":")
        rows, info = run_cell(ds, int(H), model, hw, commit)
        all_rows += rows
        deltas[c] = info
    with open(OUT / "lambda_path.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(all_rows)
    with open(OUT / "deltas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "h", "sigma_h", "delta_h", "delta_mad"])
        for c, info in deltas.items():
            for h, (s, d) in enumerate(zip(info["sigma_h"], info["delta_h"])):
                w.writerow([c, h + 1, s, d, info["delta_mad"]])
    (OUT / "summary.json").write_text(json.dumps(
        {"hardware": hw, "git_commit": commit, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
         "wall_seconds": time.time() - t0, "tol": TOL, "cells": args.cell,
         "deltas": {c: {k: v for k, v in i.items() if k in ("delta_mad", "sigma_mad")} for c, i in deltas.items()}},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {OUT} ({(time.time() - t0) / 60:.1f} phút)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
