"""Đọc kết quả đã lưu trong results/ cho báo cáo LaTeX (NHIEM_VU_8, B.3) và notebook 02, 03.

Chỉ đọc, không ghi gì vào results/. Mọi số trong báo cáo đi qua các hàm ở đây, rồi make_numbers.py (macro),
make_tables.py (bảng) hoặc make_figures.py (hình).

Nguồn:
- B2: results/three_predictions_sweep.csv, results/B2_three_predictions.md (notebook 01).
- B3: results/huber_mad/results_v2.csv (MSE, MAE từ grid; Huber δ_MAD), results/results_v2.csv (Huber δ = 1).
  Tiêu chí chọn λ* (BAO_CAO_NV7 B.1): MSE test ở λ*_MSE (selection = val_mse), MAE test ở λ*_MAE (val_mae).
- B1, B4 (lr gốc), C.3: results/sgd_b1_b4/summary.json.
- A (δ_MAD), C (độ lớn outlier): results/nv7/summary.json.
- B4-tinh chỉnh, C.3-tinh chỉnh (lưới lr đã mở rộng, NHIEM_VU_9 A), drop_last (B), họ ridge ETTh2 (C1), đoạn hằng (C2),
  δ theo bước h (C3): results/nv9/summary.json; lr đã chọn: results/sgd_lr/selection.json gộp với
  results/sgd_lr_ext/selection.json (report_nv9.merged_selection). Bản trước mở rộng (nv7) vẫn đọc được qua nv7().
- Outlier: results/outliers/results_v2.csv (MSE, MAE), results/huber_mad/outliers/results_v2.csv (Huber δ_MAD).
- ETTh2: results/etth2_report_2/summary.json, results/constant_windows/summary.json.
"""
import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["Linear", "DLinear", "NLinear"]
OBJECTIVES = ["MSE", "MAE", "Huber"]
METRICS = ["mse", "mae"]
SEL = {"mse": "val_mse", "mae": "val_mae"}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(None)
def read_json(rel):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# B3: nghiệm tối ưu, ba mục tiêu
# ---------------------------------------------------------------------------

def _okey(r):
    if not r.get("outlier_mode"):
        return ("-", 0.0, 10.0)
    return (r["outlier_mode"], float(r["outlier_rate"]), float(r.get("outlier_magnitude") or 10))


@lru_cache(None)
def _index(rel):
    """(okey, seed, dataset, H, model, objective, selection, level) → dòng."""
    out = {}
    for r in read_csv(RES / rel):
        seed = int(r["outlier_seed"]) if r.get("outlier_seed") else 0
        out[(_okey(r), seed, r["dataset"], int(r["H"]), r["model"], r["objective"], r["selection"], r["level"])] = r
    return out


def b3_row(ds, H, m, ob, metric, level="lam_star", huber="mad"):
    """Dòng của B3 theo tiêu chí mới. huber = 'mad' (bảng chính) hoặc 'one' (δ = 1, results/results_v2.csv)."""
    rel = "results_v2.csv" if (ob == "Huber" and huber == "one") else "huber_mad/results_v2.csv"
    return _index(rel)[(("-", 0.0, 10.0), 0, ds, H, m, ob, SEL[metric], level)]


def b3(ds, H, m, ob, metric, level="lam_star", huber="mad"):
    return float(b3_row(ds, H, m, ob, metric, level, huber)[f"test_{metric}"])


def b3_lam(ds, H, m, ob, metric, huber="mad"):
    return float(b3_row(ds, H, m, ob, metric, "lam_star", huber)["lam"])


def b3_delta(ds, H, m):
    return float(b3_row(ds, H, m, "Huber", "mse")["delta"])


def cells():
    return [(ds, H, m) for ds in DATASETS for H in HORIZONS for m in MODELS]


def ranking(ds, H, m, metric, huber="mad"):
    v = {ob: b3(ds, H, m, ob, metric, huber=huber) for ob in OBJECTIVES}
    return sorted(OBJECTIVES, key=lambda o: v[o]), v


def best_counts(metric):
    c = {ob: 0 for ob in OBJECTIVES}
    for k in cells():
        c[ranking(*k, metric)[0][0]] += 1
    return c


def huber_rank_counts(metric):
    c = {1: 0, 2: 0, 3: 0}
    for k in cells():
        c[ranking(*k, metric)[0].index("Huber") + 1] += 1
    return c


def mae_minus_mse_gain(ds, m):
    """MSE test(MSE) − MSE test(MAE) theo H (dương: huấn luyện bằng MAE cho MSE test tốt hơn)."""
    return [b3(ds, H, m, "MSE", "mse") - b3(ds, H, m, "MAE", "mse") for H in HORIZONS]


# ---------------------------------------------------------------------------
# Outlier (ETTh1, seed 2026, ±10 sd trừ khi ghi khác)
# ---------------------------------------------------------------------------

def outlier(mode, rate, H, m, ob, metric, seed=2026, mag=10.0):
    if rate == 0:
        return b3("ETTh1", H, m, ob, metric)
    if mag == 10.0:
        rel = "huber_mad/outliers/results_v2.csv" if ob == "Huber" else "outliers/results_v2.csv"
    else:
        rel = "outliers_mag/results_v2.csv"
    r = _index(rel).get(((mode, float(rate), float(mag)), seed, "ETTh1", H, m, ob, SEL[metric], "lam_star"))
    return None if r is None else float(r[f"test_{metric}"])


# ---------------------------------------------------------------------------
# B2 (RQ1)
# ---------------------------------------------------------------------------

@lru_cache(None)
def sweep():
    rows = read_csv(RES / "three_predictions_sweep.csv")
    for r in rows:
        for k in ("k", "lam", "rel_dW", "rms_gap_same", "rms_gap_best", "lam_best", "mse_val", "mse_test", "mae_test",
                  "rowsum_mean"):
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


def sweep_at(model, lam, k=None):
    for r in sweep():
        if r["model"] == model and r["lam"] == lam and (k is None or r["k"] == k):
            return r
    raise KeyError((model, lam, k))


@lru_cache(None)
def b2_md_numbers():
    """Các số trong results/B2_three_predictions.md mà CSV không có (notebook 01 tự sinh file này)."""
    t = (RES / "B2_three_predictions.md").read_text(encoding="utf-8")
    g = lambda pat: re.search(pat, t).group(1)                              # noqa: E731
    return {
        "rankZ": g(r"rank Z = (\d+)/"), "ncolZ": g(r"rank Z = \d+/(\d+)"),
        "dlin_dyhat": g(r"max\\\|Δŷ\\\| test ≤ ([0-9e.\-]+)"),
        "dlin_dW": g(r"‖ΔW‖/‖W‖ ≤ ([0-9e.\-]+)"),
        "dlin_grad": g(r"‖∇‖ qua module ≈ ([0-9e.\-]+)"),
        "nlin_dWF": g(r"‖ΔW‖_F = ‖r‖·‖u‖/s = ([0-9.]+)"),
        "nlin_rms": g(r"RMS Δŷ test = ([0-9.]+)"),
        "nlin_dsse": g(r"SSE train tăng đúng ‖r‖²/s = ([0-9.]+)"),
    }


# ---------------------------------------------------------------------------
# SGD: B1, B4, C.3 (lr gốc) và bản tinh chỉnh
# ---------------------------------------------------------------------------

def sgd_summary():
    return read_json("sgd_b1_b4/summary.json")


def nv7():
    return read_json("nv7/summary.json")


def nv9():
    return read_json("nv9/summary.json")


def b1_rows():
    return {(r["dataset"], r["H"], r["model"]): r for r in sgd_summary()["B1"]["rows"]}


def b4_rows():
    return {(r["dataset"], r["H"], r["model"]): r for r in sgd_summary()["B4"]["rows"]}


def b4t_rows():
    return {(r["dataset"], r["H"], r["model"]): r for r in nv9()["A_B4"]["rows"]}


def c3_rows():
    return {(r["dataset"], r["H"]): r for r in sgd_summary()["C3"]["rows"]}


def c3t_rows():
    return {(r["dataset"], r["H"]): r for r in nv9()["A_C3"]["rows"]}


@lru_cache(None)
def _merged_sel():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_nv9 import merged_selection
    return merged_selection()[0]


def lr_cells():
    """lr đã chọn sau khi mở rộng lưới (NHIEM_VU_9 A); mean_val_by_mult, sel_files gồm cả các mức mới."""
    return {(c["dataset"], int(c["H"]), c["model"]): c for c in _merged_sel().values()}


def lr_noisy_count():
    """Số ô mà trung bình val của mức lr chọn hơn mức thứ hai chưa tới một sai số chuẩn (5 seed của mức chọn)."""
    n, ratios = 0, []
    for c in lr_cells().values():
        means = {k: float(v) for k, v in c["mean_val_by_mult"].items()}
        ch = f"{float(c['chosen_mult']):g}"
        second = sorted(v for k, v in means.items() if k != ch)[0]
        vals = [json.loads((ROOT / Path(p.replace("\\", "/"))).read_text(encoding="utf-8"))["val_mse"]
                for p in c["sel_files"][ch]]
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ratios.append((second - means[ch]) / se)
        n += (second - means[ch]) < se
    return n, float(np.median(ratios))


# ---------------------------------------------------------------------------
# ETTh2
# ---------------------------------------------------------------------------

def etth2():
    return read_json("etth2_report_2/summary.json")


def constant_windows():
    return read_json("constant_windows/summary.json")
