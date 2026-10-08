"""δ của Huber theo thang phần dư cho mọi tập train của NHIEM_VU_7 (Phần A.1), ghi results/huber_mad/deltas.csv.

    python scripts/huber_deltas.py

Mỗi dòng: một (dataset, H, mô hình, tập train). δ = 1,345·σ̂, σ̂ = 1,4826·MAD của mọi phần tử phần dư train của
nghiệm MSE λ = 0 của chính mô hình đó (NLinear: nghiệm có ràng buộc) trên chính tập train đó, đúng hàm runner dùng
(runner.Cell.huber_delta). Cột sigma_h_*: σ̂ tính riêng từng bước h (min, trung vị, max trên H bước), để tham khảo.

Tập train: grid gốc (3 dataset × 4 H × 3 mô hình), các tập đã chèn outlier của results/outliers/ (series 1/5/10%,
target 1/10% ở H = 96, 720, B.5 seed 2027–2028), và các tập của Phần C (độ lớn 3, 5). Chỉ dùng CPU. Cột mask_sha1:
băm của mặt nạ vị trí outlier, để kiểm cùng seed thì cùng vị trí ở mọi độ lớn.
"""
import csv
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner as R                                     # noqa: E402
from bench_5060ti import ROOT, git_commit, hw_info     # noqa: E402

OUT = ROOT / "results" / "huber_mad" / "deltas.csv"
LOW, HIGH = 0.05, 5.0                                  # NHIEM_VU_7 mục 7: ngoài khoảng này là bất thường


def configs():
    """(dataset, H, models, outlier dict hoặc None)."""
    for ds in R.DATASETS:
        for H in R.HORIZONS:
            yield ds, H, R.MODELS, None
    for rate in (0.01, 0.05, 0.1):
        for H in R.HORIZONS:
            yield "ETTh1", H, ["Linear", "NLinear"], dict(mode="series", rate=rate, seed=2026, magnitude=10.0)
    for rate in (0.01, 0.1):
        for H in (96, 720):
            yield "ETTh1", H, ["Linear", "NLinear"], dict(mode="target", rate=rate, seed=2026, magnitude=10.0)
    for seed in (2027, 2028):
        for rate in (0.01, 0.1):
            yield "ETTh1", 96, ["Linear"], dict(mode="series", rate=rate, seed=seed, magnitude=10.0)
    for mag in (3.0, 5.0):
        for mode in ("series", "target"):
            for rate in (0.01, 0.1):
                for H in (96, 720):
                    yield "ETTh1", H, ["Linear"], dict(mode=mode, rate=rate, seed=2026, magnitude=mag)


def main():
    R.HUBER_RULE["delta"] = "mad"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hw = hw_info()
    env = {"hardware": f"CPU {hw['cpu']}; numpy {np.__version__}; {hw['os']}", "git_commit": git_commit()}
    cols = ["dataset", "H", "model", "outlier_mode", "outlier_rate", "outlier_seed", "outlier_magnitude",
            "n_train", "delta", "sigma_hat", "sigma_h_min", "sigma_h_median", "sigma_h_max", "delta_ratio_to_1",
            "in_range", "mask_sha1", "hardware", "git_commit", "timestamp"]
    rows, bad = [], []
    for ds, H, models, out in configs():
        R.OUTLIER = out
        cell = R.Cell(ds, H)
        sha = hashlib.sha1(np.packbits(cell.outlier_mask).tobytes()).hexdigest()[:12] if out else ""
        got = {}
        for m in models:
            delta, sigma, _, sig_h = cell.huber_delta(m, residual_stats=True)
            got[m] = (delta, sigma)
            ok = LOW <= delta <= HIGH
            row = {"dataset": ds, "H": H, "model": m, "n_train": cell.n, "delta": delta, "sigma_hat": sigma,
                   "sigma_h_min": float(sig_h.min()), "sigma_h_median": float(np.median(sig_h)),
                   "sigma_h_max": float(sig_h.max()), "delta_ratio_to_1": delta / 1.0, "in_range": ok,
                   "mask_sha1": sha, **env, "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
            if out:
                row.update(outlier_mode=out["mode"], outlier_rate=out["rate"], outlier_seed=out["seed"],
                           outlier_magnitude=out["magnitude"])
            rows.append(row)
            if not ok:
                bad.append(row)
            print(f"  {ds} H={H} {m:<7} {out or ''}: δ = {delta:.4f} (σ̂ {sigma:.4f}; theo h {sig_h.min():.4f}–"
                  f"{sig_h.max():.4f})", flush=True)
        if "DLinear" in got:
            assert got["DLinear"] == got["Linear"], f"{ds} H={H}: δ của DLinear khác Linear"
        del cell
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} dòng → {OUT}")
    d = np.array([r["delta"] for r in rows])
    print(f"δ: {d.min():.4f}–{d.max():.4f}; ngoài [{LOW}, {HIGH}]: {len(bad)}")
    if bad:
        sys.exit("có δ ngoài khoảng: dừng theo NHIEM_VU_7 mục 7")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
