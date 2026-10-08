"""Lập bảng và hình của docs/BAO_CAO_NV7.md (docs/NHIEM_VU_7.md).

    python scripts/report_nv7.py --part B        # B: đổi tiêu chí chọn λ* (không chạy lại gì)
    python scripts/report_nv7.py --part all

Ghi results/nv7/tables_<phần>.md và results/nv7/summary.json (gộp các phần đã chạy). Không chọn gì trên test.

Tiêu chí (NHIEM_VU_7 B.1): MSE test báo ở λ*_MSE = argmin val_mse, MAE test ở λ*_MAE = argmin val_mae, cùng một
quy tắc cho mọi mô hình và mục tiêu. "Cũ" là val_objective (λ* theo metric val của chính mục tiêu), cách chọn của
bảng chính trong các báo cáo trước. Đọc results/results_v2.csv và results/outliers/results_v2.csv (đã gồm cả
val_objective, trùng results.csv).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_5060ti import git_commit                       # noqa: E402

OUT = ROOT / "results" / "nv7"
DATASETS = ["ETTh1", "ETTh2", "ETTm1"]
HORIZONS = [96, 192, 336, 720]
MODELS = ["Linear", "DLinear", "NLinear"]
OBJECTIVES = ["MSE", "MAE", "Huber"]
RATES = [0.0, 0.01, 0.05, 0.1]


def num(x, p=4):
    return f"{x:.{p}f}".replace(".", ",") if x >= 0 else f"−{abs(x):.{p}f}".replace(".", ",")


def lam_s(x):
    x = float(x)
    if x == 0:
        return "0"
    s = f"{x:.4g}"
    return s.replace(".", ",")


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def okey(r):
    """Khóa outlier của một dòng ("-", 0, 0, 10) cho dữ liệu gốc."""
    if not r.get("outlier_mode"):
        return ("-", 0.0, 0, 10.0)
    return (r["outlier_mode"], float(r["outlier_rate"]), int(r["outlier_seed"]),
            float(r.get("outlier_magnitude") or 10))


def index(rows):
    """(okey, dataset, H, model, objective, selection, level) → dòng."""
    return {(okey(r), r["dataset"], int(r["H"]), r["model"], r["objective"], r["selection"], r["level"]): r
            for r in rows}


class Sel:
    """Truy xuất MSE/MAE test theo tiêu chí cũ (val_objective) hoặc mới (val_mse cho MSE test, val_mae cho MAE test)."""

    def __init__(self, idx):
        self.idx = idx

    def row(self, ok, ds, H, m, o, sel, level="lam_star"):
        return self.idx.get((ok, ds, H, m, o, sel, level))

    def test(self, ok, ds, H, m, o, metric, rule):
        sel = "val_objective" if rule == "old" else ("val_mse" if metric == "mse" else "val_mae")
        r = self.row(ok, ds, H, m, o, sel)
        return None if r is None else float(r[f"test_{metric}"])

    def lam(self, ok, ds, H, m, o, sel):
        r = self.row(ok, ds, H, m, o, sel)
        return None if r is None else float(r["lam"])

    def ranking(self, ok, ds, H, m, metric, rule):
        v = {o: self.test(ok, ds, H, m, o, metric, rule) for o in OBJECTIVES}
        if any(x is None for x in v.values()):
            return None, v
        return sorted(OBJECTIVES, key=lambda o: v[o]), v


GRID = ("-", 0.0, 0, 10.0)


# ---------------------------------------------------------------------------
# Phần B
# ---------------------------------------------------------------------------

def part_b(summ):
    grid = read(ROOT / "results" / "results_v2.csv")
    old = read(ROOT / "results" / "results.csv")
    S = Sel(index(grid))
    md = ["# Phần B: đổi tiêu chí chọn λ* (Huber δ = 1, chờ Phần A)", ""]

    # kiểm: val_objective của results_v2 trùng results.csv
    oi = index(old)
    n_same = sum(1 for k, r in oi.items() if k[5] in ("val_objective", "val_mse")
                 and S.idx.get(k) is not None and all(S.idx[k][c] == r[c] for c in ("lam", "test_mse", "test_mae")))
    md += [f"Kiểm: {n_same}/{len(oi)} dòng (val_objective, val_mse) của results.csv có trong results_v2.csv với cùng "
           "λ, MSE test, MAE test.", ""]
    summ["B_check_results_v2_vs_results"] = [n_same, len(oi)]

    # B4 không đổi: mục tiêu MSE, λ* theo val_mse (B4 đọc val_objective của MSE, tức val_mse)
    n_b4 = sum(1 for ds in DATASETS for H in HORIZONS for m in MODELS
               if S.lam(GRID, ds, H, m, "MSE", "val_mse") == S.lam(GRID, ds, H, m, "MSE", "val_objective"))
    md += [f"Kiểm B4: λ* của mục tiêu MSE theo val_mse trùng λ* mà report_sgd_b1_b4.py đọc (val_objective của MSE) "
           f"ở {n_b4}/36 ô, nên B4 không đổi.", ""]
    summ["B_check_B4_unchanged"] = n_b4

    # mục tiêu nào đổi cột nào
    md += ["Theo định nghĩa: với mục tiêu MSE, MSE test giữ nguyên (val_objective = val_mse), MAE test đổi sang λ*_MAE; "
           "với MAE, MAE test giữ nguyên, MSE test đổi sang λ*_MSE; với Huber, cả hai đổi (val_huber không còn dùng).", ""]
    n_lam_diff = {}
    for o in OBJECTIVES:
        for metric, sel in (("mse", "val_mse"), ("mae", "val_mae")):
            d = [(ds, H, m) for ds in DATASETS for H in HORIZONS for m in MODELS
                 if S.lam(GRID, ds, H, m, o, sel) != S.lam(GRID, ds, H, m, o, "val_objective")]
            n_lam_diff[f"{o}_{metric}"] = len(d)
    md += ["Số ô (trên 36) có λ* mới khác λ* cũ: " + ", ".join(
        f"{o} ({metric.upper()} test) {n_lam_diff[f'{o}_{metric}']}" for o in OBJECTIVES for metric in ("mse", "mae")), ""]
    summ["B_n_lam_changed"] = n_lam_diff

    # ---- B3 theo tiêu chí mới ----
    md += ["## B3 theo tiêu chí mới (Huber δ = 1)", "",
           "MSE test ở λ*_MSE (argmin val MSE), MAE test ở λ*_MAE (argmin val MAE). λ = 0: MSE / MAE test. "
           "**Đậm**: thấp nhất trong ô (dataset, H, mô hình) giữa ba mục tiêu.", "",
           "| Dataset | H | Mô hình | Mục tiêu | λ = 0: MSE / MAE test | λ*_MSE | MSE test | λ*_MAE | MAE test |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for H in HORIZONS:
            for m in MODELS:
                best = {met: min(OBJECTIVES, key=lambda o: S.test(GRID, ds, H, m, o, met, "new")) for met in ("mse", "mae")}
                for o in OBJECTIVES:
                    z = S.row(GRID, ds, H, m, o, "val_mse", "lam0")
                    mse, mae = S.test(GRID, ds, H, m, o, "mse", "new"), S.test(GRID, ds, H, m, o, "mae", "new")
                    f = lambda v, met: f"**{num(v)}**" if best[met] == o else num(v)
                    md.append(f"| {ds} | {H} | {m} | {o} | {num(float(z['test_mse']))} / {num(float(z['test_mae']))} | "
                              f"{lam_s(S.lam(GRID, ds, H, m, o, 'val_mse'))} | {f(mse, 'mse')} | "
                              f"{lam_s(S.lam(GRID, ds, H, m, o, 'val_mae'))} | {f(mae, 'mae')} |")

    # ---- thứ hạng mục tiêu trong từng ô ----
    md += ["", "## Thứ hạng mục tiêu trong từng ô B3: cũ (val_objective) so với mới", "",
           "Thứ hạng từ tốt đến kém theo MSE test và theo MAE test. Chỉ liệt kê ô có thứ hạng đổi.", "",
           "| Dataset | H | Mô hình | metric | cũ | mới | giá trị mới (MSE / MAE / Huber) |", "|---|---|---|---|---|---|---|"]
    changes = {"mse": [], "mae": []}
    best_changes = {"mse": 0, "mae": 0}
    for ds in DATASETS:
        for H in HORIZONS:
            for m in MODELS:
                for met in ("mse", "mae"):
                    ro, _ = S.ranking(GRID, ds, H, m, met, "old")
                    rn, vn = S.ranking(GRID, ds, H, m, met, "new")
                    if ro != rn:
                        changes[met].append((ds, H, m, ro, rn))
                        best_changes[met] += ro[0] != rn[0]
                        md.append(f"| {ds} | {H} | {m} | {met.upper()} test | {' < '.join(ro)} | {' < '.join(rn)} | "
                                  f"{' / '.join(num(vn[o]) for o in OBJECTIVES)} |")
    md += ["", f"Thứ hạng đổi ở {len(changes['mse'])}/36 ô theo MSE test (mục tiêu tốt nhất đổi ở {best_changes['mse']}) "
           f"và {len(changes['mae'])}/36 ô theo MAE test (mục tiêu tốt nhất đổi ở {best_changes['mae']}). "
           "Vị trí của Huber trong các thứ hạng này **chờ Phần A** (δ_MAD).", ""]
    summ["B_ranking_changes"] = {met: [list(map(str, c[:3])) + [" < ".join(c[3]), " < ".join(c[4])] for c in v]
                                 for met, v in changes.items()}

    # đếm mục tiêu tốt nhất
    md += ["## Mục tiêu tốt nhất trong ô (số ô trên 12 mỗi dataset × mô hình)", "",
           "| Dataset | Mô hình | MSE test, cũ: MSE / MAE / Huber | MSE test, mới | MAE test, cũ | MAE test, mới |",
           "|---|---|---|---|---|---|"]
    best_counts = {}
    for ds in DATASETS:
        for m in MODELS:
            cells = []
            for met in ("mse", "mae"):
                for rule in ("old", "new"):
                    c = {o: 0 for o in OBJECTIVES}
                    for H in HORIZONS:
                        c[S.ranking(GRID, ds, H, m, met, rule)[0][0]] += 1
                    cells.append(" / ".join(str(c[o]) for o in OBJECTIVES))
                    best_counts[f"{ds}_{m}_{met}_{rule}"] = c
            md.append(f"| {ds} | {m} | " + " | ".join(cells) + " |")
    summ["B_best_counts"] = best_counts

    # ---- MAE cho MSE test tốt hơn MSE ----
    md += ["", "## \"Huấn luyện bằng MAE cho MSE test tốt hơn huấn luyện bằng MSE\": đếm lại", "",
           "Số H (trên 4) có MSE test(MAE) < MSE test(MSE). Cũ: mỗi mục tiêu ở λ* theo metric val của chính nó "
           "(MAE theo val MAE). Mới: cả hai ở λ*_MSE. Chênh = MSE test(MSE) − MSE test(MAE) theo từng H (dương là MAE tốt hơn).", "",
           "| Dataset | Mô hình | cũ: số H | cũ: chênh theo H = 96/192/336/720 | mới: số H | mới: chênh |", "|---|---|---|---|---|---|"]
    mae_wins = {}
    for ds in DATASETS:
        for m in MODELS:
            out = []
            for rule in ("old", "new"):
                d = [S.test(GRID, ds, H, m, "MSE", "mse", rule) - S.test(GRID, ds, H, m, "MAE", "mse", rule) for H in HORIZONS]
                out += [str(sum(x > 0 for x in d)), " / ".join(num(x) for x in d)]
                mae_wins[f"{ds}_{m}_{rule}"] = d
            md.append(f"| {ds} | {m} | " + " | ".join(out) + " |")
    summ["B_mae_beats_mse_on_mse_test"] = mae_wins

    # ---- outlier ----
    part_b_outliers(md, summ)

    # ---- λ* DLinear ETTh1 H = 96 ----
    md += ["", "## λ* của DLinear, ETTh1, H = 96", "",
           "| Mục tiêu | val_objective (cũ) | λ*_MSE (mới, cho MSE test) | λ*_MAE (mới, cho MAE test) | MSE test cũ → mới | MAE test cũ → mới |",
           "|---|---|---|---|---|---|"]
    dl = {}
    for o in OBJECTIVES:
        a, b, c = (S.lam(GRID, "ETTh1", 96, "DLinear", o, s) for s in ("val_objective", "val_mse", "val_mae"))
        dl[o] = [a, b, c]
        md.append(f"| {o} | {lam_s(a)} | {lam_s(b)} | {lam_s(c)} | "
                  f"{num(S.test(GRID, 'ETTh1', 96, 'DLinear', o, 'mse', 'old'))} → {num(S.test(GRID, 'ETTh1', 96, 'DLinear', o, 'mse', 'new'))} | "
                  f"{num(S.test(GRID, 'ETTh1', 96, 'DLinear', o, 'mae', 'old'))} → {num(S.test(GRID, 'ETTh1', 96, 'DLinear', o, 'mae', 'new'))} |")
    summ["B_dlinear_etth1_h96_lams"] = dl
    (OUT / "tables_B.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {OUT / 'tables_B.md'}")


def part_b_outliers(md, summ):
    rows = read(ROOT / "results" / "outliers" / "results_v2.csv")
    S = Sel(index(rows + read(ROOT / "results" / "results_v2.csv")))
    md += ["", "## Thí nghiệm outlier: mục tiêu tốt nhất theo tiêu chí cũ và mới (Huber δ = 1)", "",
           "Chèn vào chuỗi (seed 2026; rate 0 là grid gốc) và chèn chỉ vào đích (H = 96, 720). "
           "Ô ghi: tốt nhất theo MSE test / theo MAE test, cũ → mới. In đậm khi đổi.", "",
           "| Mô hình | H | chế độ | " + " | ".join(f"rate {r:.0%}" for r in RATES) + " |",
           "|---|---|---|" + "---|" * len(RATES)]
    counts = {}
    n_changed = 0
    for m in ("Linear", "NLinear"):
        for H in HORIZONS:
            for mode in ("series", "target"):
                cells = []
                for rate in RATES:
                    if mode == "target" and (H not in (96, 720) or rate == 0.05):
                        cells.append("—")
                        continue
                    ok = GRID if rate == 0 else (mode, rate, 2026, 10.0)
                    parts = []
                    for met in ("mse", "mae"):
                        bo = S.ranking(ok, "ETTh1", H, m, met, "old")[0][0]
                        bn = S.ranking(ok, "ETTh1", H, m, met, "new")[0][0]
                        counts.setdefault(f"{mode}_{rate:g}_{met}_old", {o: 0 for o in OBJECTIVES})[bo] += 1
                        counts.setdefault(f"{mode}_{rate:g}_{met}_new", {o: 0 for o in OBJECTIVES})[bn] += 1
                        parts.append(bo if bo == bn else f"**{bo} → {bn}**")
                        n_changed += (bo != bn) and rate > 0
                    cells.append(" / ".join(parts))
                md.append(f"| {m} | {H} | {mode} | " + " | ".join(cells) + " |")
    md += ["", f"Số ô có mục tiêu tốt nhất đổi (rate > 0, cả hai metric): {n_changed}.", ""]
    md += ["### Đếm mục tiêu tốt nhất theo MSE test, chèn vào chuỗi (8 ô mô hình × H): MSE / MAE / Huber", "",
           "| rate | cũ | mới |", "|---|---|---|"]
    for rate in RATES:
        c_old, c_new = counts[f"series_{rate:g}_mse_old"], counts[f"series_{rate:g}_mse_new"]
        md.append(f"| {rate:.0%} | {' / '.join(str(c_old[o]) for o in OBJECTIVES)} | {' / '.join(str(c_new[o]) for o in OBJECTIVES)} |")
    md += ["", "Theo MAE test:", "", "| rate | cũ | mới |", "|---|---|---|"]
    for rate in RATES:
        c_old, c_new = counts[f"series_{rate:g}_mae_old"], counts[f"series_{rate:g}_mae_new"]
        md.append(f"| {rate:.0%} | {' / '.join(str(c_old[o]) for o in OBJECTIVES)} | {' / '.join(str(c_new[o]) for o in OBJECTIVES)} |")

    # thứ tự suy giảm 0 → 10% và chênh mục tiêu
    md += ["", "### Suy giảm MSE test 0% → 10% (chuỗi, seed 2026): khoảng trên 8 ô, cũ → mới", "",
           "| Mục tiêu | cũ | mới |", "|---|---|---|"]
    deg = {}
    for o in OBJECTIVES:
        for rule in ("old", "new"):
            d = [S.test(("series", 0.1, 2026, 10.0), "ETTh1", H, m, o, "mse", rule) - S.test(GRID, "ETTh1", H, m, o, "mse", rule)
                 for m in ("Linear", "NLinear") for H in HORIZONS]
            deg[f"{o}_{rule}"] = d
        md.append(f"| {o} | {num(min(deg[o + '_old']))} đến {num(max(deg[o + '_old']))} | "
                  f"{num(min(deg[o + '_new']))} đến {num(max(deg[o + '_new']))} |")
    order_same = sum(1 for i in range(8) if sorted(OBJECTIVES, key=lambda o: deg[o + "_new"][i]) == ["MSE", "Huber", "MAE"])
    md += ["", f"Thứ tự suy giảm MSE < Huber < MAE (mới) ở {order_same}/8 ô.", ""]
    summ["B_outlier_best_counts"] = counts
    summ["B_outlier_degradation_mse"] = deg


# ---------------------------------------------------------------------------
# Phần A
# ---------------------------------------------------------------------------

HM = ROOT / "results" / "huber_mad"
SENS = [0.5, 1.0, 2.0, 4.0]
SENS_CELLS = [("ETTh1", 96), ("ETTh1", 720), ("ETTh2", 720)]


def huber_paths(path):
    """Đường λ của Huber trong một lambda_path: khóa (okey, ds, H, model) → dòng theo λ tăng."""
    out = {}
    for r in read(path):
        if r["objective"] == "Huber":
            out.setdefault((okey(r), r["dataset"], int(r["H"]), r["model"]), []).append(r)
    return {k: sorted(v, key=lambda r: float(r["lam"])) for k, v in out.items()}


def edge_report(paths, base_n=10):
    """Số đường phải mở rộng, và các đường có λ* (val_mse hoặc val_mae) vẫn ở đầu mút trên sau khi mở rộng."""
    ext, top, zero, nonconv, fallback = [], [], [], 0, 0
    for k, p in paths.items():
        lams = [float(r["lam"]) for r in p]
        if len(lams) > base_n:
            ext.append((k, len(lams) - base_n))
        for met in ("val_mse", "val_mae"):
            b = float(min(p, key=lambda r: float(r[met]))["lam"])
            if b == lams[-1]:
                top.append((k, met, b))
            if b == 0:
                zero.append((k, met))
        nonconv += sum(r["converged"] != "True" for r in p)
        fallback += sum(int(r["n_fallback"] or 0) for r in p)
    return ext, top, zero, nonconv, fallback


def lam_star(path, met):
    return min(path, key=lambda r: float(r[met]))


def part_a(summ):
    md = ["# Phần A: δ của Huber theo thang phần dư", ""]
    # ---- A.1 δ ----
    deltas = read(HM / "deltas.csv")
    g = [r for r in deltas if not r["outlier_mode"]]
    o = [r for r in deltas if r["outlier_mode"]]
    dg = np.array([float(r["delta"]) for r in g])
    md += ["## A.1. δ_MAD", "",
           f"Grid gốc (36 ô dataset × H × mô hình): δ_MAD từ {num(dg.min())} đến {num(dg.max())} (trung vị "
           f"{num(float(np.median(dg)))}), tức {num(dg.min())}–{num(dg.max())} lần δ = 1 cũ. Các tập train đã chèn "
           f"outlier ({len(o)} dòng): {num(min(float(r['delta']) for r in o))}–{num(max(float(r['delta']) for r in o))}. "
           f"Mọi δ nằm trong [0,05; 5]: {all(r['in_range'] == 'True' for r in deltas)}. "
           f"δ của DLinear trùng Linear ở {sum(1 for r in g if r['model'] == 'DLinear' and r['delta'] == next(x['delta'] for x in g if x['model'] == 'Linear' and x['dataset'] == r['dataset'] and x['H'] == r['H']))}/12 ô (trùng từng bit).", "",
           "| Dataset | H | Mô hình | δ_MAD | σ̂ | σ̂ theo h: min / trung vị / max | max/min theo h |", "|---|---|---|---|---|---|---|"]
    for r in g:
        if r["model"] == "DLinear":
            continue
        md.append(f"| {r['dataset']} | {r['H']} | {r['model']} | {num(float(r['delta']))} | {num(float(r['sigma_hat']))} | "
                  f"{num(float(r['sigma_h_min']))} / {num(float(r['sigma_h_median']))} / {num(float(r['sigma_h_max']))} | "
                  f"{num(float(r['sigma_h_max']) / float(r['sigma_h_min']), 1)} |")
    md += ["", "DLinear: δ bằng Linear (cùng nghiệm MSE λ = 0), không lặp lại.", "",
           "Tập train đã chèn outlier (ETTh1, seed 2026 trừ khi ghi khác):", "",
           "| chế độ | rate | seed | độ lớn | H | Mô hình | δ_MAD | δ_MAD tập gốc | mask |", "|---|---|---|---|---|---|---|---|---|"]
    base = {(r["dataset"], r["H"], r["model"]): float(r["delta"]) for r in g}
    for r in o:
        md.append(f"| {r['outlier_mode']} | {float(r['outlier_rate']):.0%} | {r['outlier_seed']} | {float(r['outlier_magnitude']):g} | "
                  f"{r['H']} | {r['model']} | {num(float(r['delta']))} | {num(base[(r['dataset'], r['H'], r['model'])])} | {r['mask_sha1']} |")
    summ["A_delta_grid_range"] = [float(dg.min()), float(dg.max())]
    summ["A_delta_outlier_range"] = [min(float(r["delta"]) for r in o), max(float(r["delta"]) for r in o)]

    # ---- đường λ, mở rộng đầu mút ----
    for name, path in (("grid", HM / "lambda_path.csv"), ("outlier", HM / "outliers" / "lambda_path.csv")):
        if not path.exists():
            continue
        paths = huber_paths(path)
        ext, top, zero, nonconv, fb = edge_report(paths)
        md += ["", f"## Đường λ của Huber δ_MAD ({name}): {len(paths)} đường", "",
               f"- Phải mở rộng đầu mút: {len(ext)} đường" + (": " + "; ".join(
                   f"{k[1]} H={k[2]} {k[3]}{'' if k[0] == GRID else ' ' + str(k[0][:3])} (+{n})" for k, n in ext) if ext else "") + ".",
               f"- λ*_MSE hoặc λ*_MAE vẫn ở đầu mút trên sau khi mở rộng: {len({t[0] for t in top})} đường"
               + (": " + "; ".join(f"{k[1]} H={k[2]} {k[3]} {met} λ = {lam_s(b)}" for k, met, b in top) if top else "") + ".",
               f"- λ* = 0 (đầu mút dưới, không mở rộng được): {len(zero)} (đường, metric).",
               f"- Dòng converged = False: {nonconv}; số bước h fallback float64: {fb}."]
        summ[f"A_edges_{name}"] = {"n_paths": len(paths), "extended": len(ext), "top_edge": [list(map(str, t)) for t in top],
                                   "lam0": len(zero), "nonconverged": nonconv, "fallback": fb}

    # ---- B3 hoàn chỉnh ----
    fin = read(HM / "results_v2.csv")
    b_old = read(ROOT / "results" / "results_v2.csv")
    SF, SB = Sel(index(fin)), Sel(index(b_old))
    md += ["", "## B3 hoàn chỉnh (tiêu chí mới, Huber δ_MAD)", "",
           "MSE test ở λ*_MSE, MAE test ở λ*_MAE. MSE, MAE như Phần B; Huber chạy lại với δ_MAD. "
           "**Đậm**: thấp nhất trong ô giữa ba mục tiêu. Cột cuối: Huber δ = 1 (Phần B) để so.", "",
           "| Dataset | H | Mô hình | Mục tiêu | δ | λ = 0: MSE / MAE test | λ*_MSE | MSE test | λ*_MAE | MAE test | Huber δ = 1: MSE / MAE test |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for H in HORIZONS:
            for m in MODELS:
                best = {met: min(OBJECTIVES, key=lambda ob: SF.test(GRID, ds, H, m, ob, met, "new")) for met in ("mse", "mae")}
                for ob in OBJECTIVES:
                    z = SF.row(GRID, ds, H, m, ob, "val_mse", "lam0")
                    mse, mae = SF.test(GRID, ds, H, m, ob, "mse", "new"), SF.test(GRID, ds, H, m, ob, "mae", "new")
                    f = lambda v, met: f"**{num(v)}**" if best[met] == ob else num(v)
                    old = (f"{num(SB.test(GRID, ds, H, m, ob, 'mse', 'new'))} / {num(SB.test(GRID, ds, H, m, ob, 'mae', 'new'))}"
                           if ob == "Huber" else "")
                    dl = num(float(z["delta"])) if ob == "Huber" else ("1e-6" if ob == "MAE" else "—")
                    md.append(f"| {ds} | {H} | {m} | {ob} | {dl} | {num(float(z['test_mse']))} / {num(float(z['test_mae']))} | "
                              f"{lam_s(SF.lam(GRID, ds, H, m, ob, 'val_mse'))} | {f(mse, 'mse')} | "
                              f"{lam_s(SF.lam(GRID, ds, H, m, ob, 'val_mae'))} | {f(mae, 'mae')} | {old} |")

    # ---- vị trí Huber ----
    md += ["", "## Vị trí của Huber so với MSE và MAE trong B3", "",
           "Hạng của Huber (1 = tốt nhất) giữa ba mục tiêu. Ba cột: cũ (δ = 1, val_objective), Phần B (δ = 1, tiêu chí mới), "
           "cuối (δ_MAD, tiêu chí mới). Chỉ liệt kê ô có hạng của Huber đổi giữa Phần B và cuối, hoặc giữa cũ và cuối.", "",
           "| Dataset | H | Mô hình | metric | hạng cũ | hạng Phần B | hạng cuối | Huber cuối − Huber δ=1 (Phần B) | thứ hạng cuối |",
           "|---|---|---|---|---|---|---|---|---|"]
    pos = {"mse": {"old_final": 0, "b_final": 0}, "mae": {"old_final": 0, "b_final": 0}}
    huber_vs = {}
    for ds in DATASETS:
        for H in HORIZONS:
            for m in MODELS:
                for met in ("mse", "mae"):
                    ro = SB.ranking(GRID, ds, H, m, met, "old")[0]
                    rb = SB.ranking(GRID, ds, H, m, met, "new")[0]
                    rf, vf = SF.ranking(GRID, ds, H, m, met, "new")
                    po, pb, pf = ro.index("Huber") + 1, rb.index("Huber") + 1, rf.index("Huber") + 1
                    pos[met]["old_final"] += po != pf
                    pos[met]["b_final"] += pb != pf
                    d = vf["Huber"] - SB.test(GRID, ds, H, m, "Huber", met, "new")
                    huber_vs[f"{ds}_{H}_{m}_{met}"] = {"rank_old": po, "rank_B": pb, "rank_final": pf, "d_huber": d,
                                                       "order_final": rf}
                    if po != pf or pb != pf:
                        md.append(f"| {ds} | {H} | {m} | {met.upper()} test | {po} | {pb} | {pf} | {num(d)} | {' < '.join(rf)} |")
    md += ["", f"Hạng của Huber đổi (cuối so với Phần B): {pos['mse']['b_final']}/36 ô theo MSE test, {pos['mae']['b_final']}/36 theo MAE test. "
           f"Cuối so với cũ: {pos['mse']['old_final']}/36 và {pos['mae']['old_final']}/36.", ""]
    dh = {met: [v["d_huber"] for k, v in huber_vs.items() if k.endswith(met)] for met in ("mse", "mae")}
    md += [f"Huber δ_MAD − Huber δ = 1 (tiêu chí mới): MSE test {num(min(dh['mse']))} đến {num(max(dh['mse']))} "
           f"(tốt hơn ở {sum(x < 0 for x in dh['mse'])}/36 ô); MAE test {num(min(dh['mae']))} đến {num(max(dh['mae']))} "
           f"(tốt hơn ở {sum(x < 0 for x in dh['mae'])}/36 ô).", ""]
    rank_counts = {met: {p: sum(1 for k, v in huber_vs.items() if k.endswith(met) and v["rank_final"] == p) for p in (1, 2, 3)}
                   for met in ("mse", "mae")}
    md += [f"Hạng của Huber (cuối): theo MSE test hạng 1/2/3 ở {rank_counts['mse'][1]}/{rank_counts['mse'][2]}/{rank_counts['mse'][3]} ô; "
           f"theo MAE test {rank_counts['mae'][1]}/{rank_counts['mae'][2]}/{rank_counts['mae'][3]} ô.", ""]
    md += ["Mục tiêu tốt nhất trong ô (cuối), số ô trên 4 H: MSE / MAE / Huber", "",
           "| Dataset | Mô hình | MSE test | MAE test |", "|---|---|---|---|"]
    best_final = {}
    for ds in DATASETS:
        for m in MODELS:
            cells = []
            for met in ("mse", "mae"):
                c = {ob: 0 for ob in OBJECTIVES}
                for H in HORIZONS:
                    c[SF.ranking(GRID, ds, H, m, met, "new")[0][0]] += 1
                best_final[f"{ds}_{m}_{met}"] = c
                cells.append(" / ".join(str(c[ob]) for ob in OBJECTIVES))
            md.append(f"| {ds} | {m} | " + " | ".join(cells) + " |")
    summ["A_huber_position"] = {"changes": pos, "rank_counts": rank_counts, "cells": huber_vs, "best_final": best_final}

    part_a_outliers(md, summ)
    part_a_sensitivity(md, summ)
    (OUT / "tables_A.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {OUT / 'tables_A.md'}")


def outlier_sel(final=True):
    """Sel trên grid + outlier; final=True: Huber δ_MAD (results/huber_mad/...), ngược lại Huber δ = 1."""
    if final:
        rows = read(HM / "outliers" / "results_v2.csv") + read(HM / "results_v2.csv")
    else:
        rows = read(ROOT / "results" / "outliers" / "results_v2.csv") + read(ROOT / "results" / "results_v2.csv")
    return Sel(index(rows))


def part_a_outliers(md, summ):
    if not (HM / "outliers" / "results_v2.csv").exists():
        return
    SF, SB = outlier_sel(True), outlier_sel(False)
    md += ["", "## Thí nghiệm outlier với Huber δ_MAD (tiêu chí mới)", "",
           "δ tính trên chính tập train đã chèn. Rate 0 là grid gốc (Huber δ_MAD). MSE, MAE không đổi so với Phần B.", "",
           "### Bảng chính: Huber, chèn vào chuỗi (seed 2026)", "",
           "| Mô hình | H | rate | δ | λ*_MSE | MSE test | λ*_MAE | MAE test | Δ MSE so với 0% | Δ MAE | Huber δ = 1: MSE / MAE test |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("Linear", "NLinear"):
        for H in HORIZONS:
            for rate in RATES:
                ok = GRID if rate == 0 else ("series", rate, 2026, 10.0)
                r = SF.row(ok, "ETTh1", H, m, "Huber", "val_mse")
                mse, mae = SF.test(ok, "ETTh1", H, m, "Huber", "mse", "new"), SF.test(ok, "ETTh1", H, m, "Huber", "mae", "new")
                b_mse, b_mae = SF.test(GRID, "ETTh1", H, m, "Huber", "mse", "new"), SF.test(GRID, "ETTh1", H, m, "Huber", "mae", "new")
                dd = "—" if rate == 0 else f"{num(mse - b_mse)} | {num(mae - b_mae)}"
                md.append(f"| {m} | {H} | {rate:.0%} | {num(float(r['delta']))} | {lam_s(SF.lam(ok, 'ETTh1', H, m, 'Huber', 'val_mse'))} | "
                          f"{num(mse)} | {lam_s(SF.lam(ok, 'ETTh1', H, m, 'Huber', 'val_mae'))} | {num(mae)} | "
                          f"{dd if rate else '— | —'} | {num(SB.test(ok, 'ETTh1', H, m, 'Huber', 'mse', 'new'))} / "
                          f"{num(SB.test(ok, 'ETTh1', H, m, 'Huber', 'mae', 'new'))} |")
    # mục tiêu tốt nhất
    md += ["", "### Mục tiêu tốt nhất theo MSE test / MAE test (tiêu chí mới): Huber δ = 1 → δ_MAD", "",
           "| Mô hình | H | chế độ | " + " | ".join(f"rate {r:.0%}" for r in RATES) + " |", "|---|---|---|" + "---|" * len(RATES)]
    counts, n_changed = {}, 0
    for m in ("Linear", "NLinear"):
        for H in HORIZONS:
            for mode in ("series", "target"):
                if mode == "target" and H not in (96, 720):
                    continue
                cells = []
                for rate in RATES:
                    if mode == "target" and rate == 0.05:
                        cells.append("—")
                        continue
                    ok = GRID if rate == 0 else (mode, rate, 2026, 10.0)
                    parts = []
                    for met in ("mse", "mae"):
                        bo, bn = SB.ranking(ok, "ETTh1", H, m, met, "new")[0][0], SF.ranking(ok, "ETTh1", H, m, met, "new")[0][0]
                        if mode == "series" or rate > 0:
                            counts.setdefault(f"{mode}_{rate:g}_{met}_d1", {x: 0 for x in OBJECTIVES})[bo] += 1
                            counts.setdefault(f"{mode}_{rate:g}_{met}_mad", {x: 0 for x in OBJECTIVES})[bn] += 1
                        n_changed += bo != bn
                        parts.append(bo if bo == bn else f"**{bo} → {bn}**")
                    cells.append(" / ".join(parts))
                md.append(f"| {m} | {H} | {mode} | " + " | ".join(cells) + " |")
    md += ["", f"Số (ô, metric) có mục tiêu tốt nhất đổi khi thay δ = 1 bằng δ_MAD: {n_changed}.", "",
           "Đếm mục tiêu tốt nhất, chèn vào chuỗi (8 ô): MSE / MAE / Huber", "",
           "| rate | MSE test, δ = 1 | MSE test, δ_MAD | MAE test, δ = 1 | MAE test, δ_MAD |", "|---|---|---|---|---|"]
    for rate in RATES:
        md.append(f"| {rate:.0%} | " + " | ".join(" / ".join(str(counts[f"series_{rate:g}_{met}_{v}"][x]) for x in OBJECTIVES)
                                               for met in ("mse", "mae") for v in ("d1", "mad")) + " |")
    md += ["", "Chèn chỉ vào đích (4 ô: 2 mô hình × H = 96, 720): MSE / MAE / Huber", "",
           "| rate | MSE test, δ = 1 | MSE test, δ_MAD | MAE test, δ = 1 | MAE test, δ_MAD |", "|---|---|---|---|---|"]
    for rate in (0.01, 0.1):
        md.append(f"| {rate:.0%} | " + " | ".join(" / ".join(str(counts[f"target_{rate:g}_{met}_{v}"][x]) for x in OBJECTIVES)
                                               for met in ("mse", "mae") for v in ("d1", "mad")) + " |")
    # suy giảm 0 → 10%
    md += ["", "### Suy giảm 0% → 10% (chuỗi, 8 ô), tiêu chí mới", "",
           "| Mục tiêu | Δ MSE test | Δ MAE test |", "|---|---|---|"]
    deg = {}
    for ob, S, tag in (("MSE", SF, "MSE"), ("MAE", SF, "MAE"), ("Huber", SB, "Huber δ = 1"), ("Huber", SF, "Huber δ_MAD")):
        d = {met: [S.test(("series", 0.1, 2026, 10.0), "ETTh1", H, m, ob, met, "new") - S.test(GRID, "ETTh1", H, m, ob, met, "new")
                   for m in ("Linear", "NLinear") for H in HORIZONS] for met in ("mse", "mae")}
        deg[tag] = d
        md.append(f"| {tag} | {num(min(d['mse']))} đến {num(max(d['mse']))} | {num(min(d['mae']))} đến {num(max(d['mae']))} |")
    order = sum(1 for i in range(8) if sorted(["MSE", "MAE", "Huber δ_MAD"], key=lambda t: deg[t]["mse"][i]) == ["MSE", "Huber δ_MAD", "MAE"])
    md += ["", f"Thứ tự suy giảm MSE test MSE < Huber δ_MAD < MAE ở {order}/8 ô.", ""]
    # target: thay đổi
    md += ["### Chèn chỉ vào đích: Δ MSE test so với 0% (tiêu chí mới)", "",
           "| Mô hình | H | rate | MSE | MAE | Huber δ = 1 | Huber δ_MAD |", "|---|---|---|---|---|---|---|"]
    tgt = {}
    for m in ("Linear", "NLinear"):
        for H in (96, 720):
            for rate in (0.01, 0.1):
                ok = ("target", rate, 2026, 10.0)
                v = [SF.test(ok, "ETTh1", H, m, "MSE", "mse", "new") - SF.test(GRID, "ETTh1", H, m, "MSE", "mse", "new"),
                     SF.test(ok, "ETTh1", H, m, "MAE", "mse", "new") - SF.test(GRID, "ETTh1", H, m, "MAE", "mse", "new"),
                     SB.test(ok, "ETTh1", H, m, "Huber", "mse", "new") - SB.test(GRID, "ETTh1", H, m, "Huber", "mse", "new"),
                     SF.test(ok, "ETTh1", H, m, "Huber", "mse", "new") - SF.test(GRID, "ETTh1", H, m, "Huber", "mse", "new")]
                tgt[f"{m}_{H}_{rate:g}"] = v
                md.append(f"| {m} | {H} | {rate:.0%} | " + " | ".join(num(x) for x in v) + " |")
    # B.5
    md += ["", "### B.5 với Huber δ_MAD (Linear, H = 96, chuỗi, seed 2026–2028), MSE test ở λ*_MSE", "",
           "| rate | Mục tiêu | seed 2026 / 2027 / 2028 | TB | max − min |", "|---|---|---|---|---|"]
    b5 = {}
    for rate in (0.01, 0.1):
        means = {}
        for ob in OBJECTIVES:
            v = [SF.test(("series", rate, s, 10.0), "ETTh1", 96, "Linear", ob, "mse", "new") for s in (2026, 2027, 2028)]
            means[ob] = float(np.mean(v))
            b5[f"{rate:g}_{ob}"] = v
            md.append(f"| {rate:.0%} | {ob} | {' / '.join(num(x) for x in v)} | {num(means[ob])} | {num(max(v) - min(v))} |")
        order_seed = [sorted(OBJECTIVES, key=lambda ob: b5[f"{rate:g}_{ob}"][i]) for i in range(3)]
        md.append(f"| {rate:.0%} | thứ tự theo seed | {' ; '.join(' < '.join(x) for x in order_seed)} | khác biệt TB giữa mục tiêu "
                  f"{num(max(means.values()) - min(means.values()))} | |")
    summ["A_outlier"] = {"best_counts": counts, "n_changed": n_changed, "degradation": deg, "target_dmse": tgt, "B5": b5}
    figure_h2_v2()


def part_a_sensitivity(md, summ):
    S0 = Sel(index(read(HM / "results_v2.csv")))
    sens = {}
    for s in SENS:
        f = HM / "results_v2.csv" if s == 1.0 else HM / f"sens_x{s:g}" / "results_v2.csv"
        if f.exists():
            sens[s] = Sel(index(read(f)))
    if len(sens) < len(SENS):
        return
    SD1 = Sel(index(read(ROOT / "results" / "results_v2.csv")))          # δ = 1 (grid cũ)
    md += ["", "## A.3. Độ nhạy theo δ", "",
           "Huber với δ ∈ {0,5; 1; 2; 4} × δ_MAD và δ = 1 cố định; λ* theo tiêu chí B.1. Hạng = vị trí của Huber "
           "giữa ba mục tiêu (MSE, MAE cố định như B3).", "",
           "| Dataset | H | Mô hình | δ | λ*_MSE | MSE test | hạng (MSE test) | λ*_MAE | MAE test | hạng (MAE test) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for ds, H in SENS_CELLS:
        for m in ("Linear", "NLinear"):
            ref = {met: {ob: S0.test(GRID, ds, H, m, ob, met, "new") for ob in ("MSE", "MAE")} for met in ("mse", "mae")}
            rows = [(f"{s:g} × δ_MAD", sens[s]) for s in SENS] + [("1 (cố định)", SD1)]
            for tag, S in rows:
                r = S.row(GRID, ds, H, m, "Huber", "val_mse")
                v = {met: S.test(GRID, ds, H, m, "Huber", met, "new") for met in ("mse", "mae")}
                rank = {met: 1 + sum(ref[met][ob] < v[met] for ob in ("MSE", "MAE")) for met in ("mse", "mae")}
                out[f"{ds}_{H}_{m}_{tag}"] = {"delta": float(r["delta"]), **{f"test_{k}": x for k, x in v.items()},
                                              **{f"rank_{k}": x for k, x in rank.items()},
                                              "lam_mse": S.lam(GRID, ds, H, m, "Huber", "val_mse"),
                                              "lam_mae": S.lam(GRID, ds, H, m, "Huber", "val_mae")}
                md.append(f"| {ds} | {H} | {m} | {tag.replace('.', ',')} = {num(float(r['delta']))} | {lam_s(out[f'{ds}_{H}_{m}_{tag}']['lam_mse'])} | "
                          f"{num(v['mse'])} | {rank['mse']} | {lam_s(out[f'{ds}_{H}_{m}_{tag}']['lam_mae'])} | {num(v['mae'])} | {rank['mae']} |")
            md.append(f"| {ds} | {H} | {m} | tham chiếu: MSE / MAE | | {num(ref['mse']['MSE'])} / {num(ref['mse']['MAE'])} | | | "
                      f"{num(ref['mae']['MSE'])} / {num(ref['mae']['MAE'])} | |")
    for sc in (0.5, 2.0, 4.0):
        ext, top, zero, nonconv, fb = edge_report(huber_paths(HM / f"sens_x{sc:g}" / "lambda_path.csv"))
        md.append(f"\n×{sc:g}: {len(ext)} đường mở rộng, {len(top)} (đường, metric) ở đầu mút trên sau khi mở rộng, "
                  f"converged = False {nonconv}, fallback {fb}.")
    summ["A_sensitivity"] = out
    figure_sensitivity(out)


def _style(ax):
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


COLORS = {"MSE": "#2a78d6", "MAE": "#eb6834", "Huber": "#1baf7a", "Huber δ=1": "#8a8a8a"}


def figure_sensitivity(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S0 = Sel(index(read(HM / "results_v2.csv")))
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.6))
    for j, (ds, H, m) in enumerate((ds, H, m) for ds, H in SENS_CELLS for m in ("Linear", "NLinear")):
        for i, met in enumerate(("mse", "mae")):
            ax = axes[i, j]
            dm = out[f"{ds}_{H}_{m}_1 × δ_MAD"]["delta"]
            xs = SENS
            ys = [out[f"{ds}_{H}_{m}_{s:g} × δ_MAD"][f"test_{met}"] for s in SENS]
            ax.plot(xs, ys, "-^", color=COLORS["Huber"], ms=5, lw=1.6, label="Huber, δ = {0,5; 1; 2; 4} × δ_MAD")
            d1 = out[f"{ds}_{H}_{m}_1 (cố định)"]
            ax.plot([1.0 / dm], [d1[f"test_{met}"]], "D", color=COLORS["Huber δ=1"], ms=5, label="Huber, δ = 1 (cũ)")
            for ob in ("MSE", "MAE"):
                ax.axhline(S0.test(GRID, ds, H, m, ob, met, "new"), color=COLORS[ob], lw=1.2, ls="--",
                           label=f"mục tiêu {ob}")
            ax.set_xscale("log")
            ax.set_xticks(SENS)
            ax.set_xticklabels(["0,5", "1", "2", "4"])
            ax.minorticks_off()
            ax.set_title(f"{ds} H = {H}, {m}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{met.upper()} test ở λ*_{met.upper()}")
            if i == 1:
                ax.set_xlabel("δ / δ_MAD (log)")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Độ nhạy của Huber theo δ (λ* theo val MSE cho MSE test, theo val MAE cho MAE test)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    f = HM / "A3_delta_sensitivity.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    print(f"→ {f.relative_to(ROOT)}")


def figure_h2_v2():
    """H2 với Huber δ_MAD, tiêu chí mới: MSE test ở λ*_MSE, MAE test ở λ*_MAE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = outlier_sel(True)
    markers = {"MSE": "o", "MAE": "s", "Huber": "^"}
    for met, fname in (("mse", "H2_v2_outlier_robustness.png"), ("mae", "H2_v2_outlier_robustness_mae.png")):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
        for i, m in enumerate(("Linear", "NLinear")):
            for j, H in enumerate(HORIZONS):
                ax = axes[i, j]
                for ob in OBJECTIVES:
                    for mode, ls, rates in (("series", "-", RATES), ("target", "--", [0.0, 0.01, 0.1])):
                        pts = [(r * 100, v) for r in rates
                               if (v := S.test(GRID if r == 0 else (mode, r, 2026, 10.0), "ETTh1", H, m, ob, met, "new")) is not None]
                        if len(pts) < 2:
                            continue
                        x, y = zip(*pts)
                        name = "Huber (δ_MAD)" if ob == "Huber" else ob
                        ax.plot(x, y, ls=ls, color=COLORS[ob], marker=markers[ob], ms=4 if mode == "series" else 3,
                                lw=1.6 if mode == "series" else 1.2, alpha=1 if mode == "series" else 0.8,
                                label=f"{name}, chèn vào chuỗi" if mode == "series" else f"{name}, chèn chỉ vào đích")
                ax.set_title(f"{m}, H = {H}", fontsize=10)
                ax.set_xticks([0, 1, 5, 10])
                _style(ax)
                if j == 0:
                    ax.set_ylabel(f"{met.upper()} test (dữ liệu chuẩn hóa)")
                if i == 1:
                    ax.set_xlabel("tỷ lệ outlier trong train (%)")
        h, l = [], []
        for ax in axes.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh)
                    l.append(ll)
        fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"{met.upper()} test theo tỷ lệ outlier, ETTh1, λ* theo val {met.upper()} sạch; Huber δ_MAD; "
                     "outlier ±10 sd, seed 2026", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        f = ROOT / "results" / "outliers" / fname
        fig.savefig(f, dpi=150)
        plt.close(fig)
        print(f"→ {f.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Phần C
# ---------------------------------------------------------------------------

MAGS = [3.0, 5.0, 10.0]


def part_c(summ):
    rows = (read(ROOT / "results" / "outliers_mag" / "results_v2.csv") + read(HM / "outliers" / "results_v2.csv")
            + read(HM / "results_v2.csv"))
    S = Sel(index(rows))
    md = ["# Phần C: độ lớn outlier 3σ, 5σ (và 10σ)", "",
          "Linear, ETTh1, seed 2026; λ* theo tiêu chí B.1 (MSE test ở λ*_MSE, MAE test ở λ*_MAE); Huber δ_MAD tính trên "
          "chính tập train đã chèn. Độ lớn 10: MSE, MAE từ results/outliers/, Huber từ results/huber_mad/outliers/. "
          "Rate 0 là grid gốc (Huber δ_MAD). Cùng seed nên vị trí outlier trùng nhau ở mọi độ lớn (mask_sha1 trong deltas.csv).", "",
          "| H | chế độ | rate | độ lớn | MSE: MSE / MAE test | MAE: MSE / MAE test | Huber: MSE / MAE test | tốt nhất (MSE test / MAE test) |",
          "|---|---|---|---|---|---|---|---|"]
    best, vals = {}, {}
    for H in (96, 720):
        z = {ob: (S.test(GRID, "ETTh1", H, "Linear", ob, "mse", "new"), S.test(GRID, "ETTh1", H, "Linear", ob, "mae", "new"))
             for ob in OBJECTIVES}
        bz = [min(OBJECTIVES, key=lambda ob: z[ob][k]) for k in (0, 1)]
        md.append(f"| {H} | — | 0% | — | " + " | ".join(f"{num(z[ob][0])} / {num(z[ob][1])}" for ob in OBJECTIVES)
                  + f" | {bz[0]} / {bz[1]} |")
        for mode in ("series", "target"):
            for rate in (0.01, 0.1):
                for mag in MAGS:
                    ok = (mode, rate, 2026, mag)
                    v = {ob: (S.test(ok, "ETTh1", H, "Linear", ob, "mse", "new"), S.test(ok, "ETTh1", H, "Linear", ob, "mae", "new"))
                         for ob in OBJECTIVES}
                    if any(x is None for t in v.values() for x in t):
                        md.append(f"| {H} | {mode} | {rate:.0%} | {mag:g} | (thiếu) | | | |")
                        continue
                    b = [min(OBJECTIVES, key=lambda ob: v[ob][k]) for k in (0, 1)]
                    best[f"{H}_{mode}_{rate:g}_{mag:g}"] = b
                    vals[f"{H}_{mode}_{rate:g}_{mag:g}"] = v
                    md.append(f"| {H} | {mode} | {rate:.0%} | {mag:g} | " + " | ".join(f"{num(v[ob][0])} / {num(v[ob][1])}" for ob in OBJECTIVES)
                              + f" | {b[0]} / {b[1]} |")
    md += ["", "## Δ MSE test so với rate 0 theo độ lớn", "",
           "| H | chế độ | rate | độ lớn | MSE | MAE | Huber |", "|---|---|---|---|---|---|---|"]
    for H in (96, 720):
        for mode in ("series", "target"):
            for rate in (0.01, 0.1):
                for mag in MAGS:
                    k = f"{H}_{mode}_{rate:g}_{mag:g}"
                    if k not in vals:
                        continue
                    md.append(f"| {H} | {mode} | {rate:.0%} | {mag:g} | " + " | ".join(
                        num(vals[k][ob][0] - S.test(GRID, "ETTh1", H, "Linear", ob, "mse", "new")) for ob in OBJECTIVES) + " |")
    summ["C"] = {"best": best, "values": vals}
    figure_c(S, vals)
    (OUT / "tables_C.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {OUT / 'tables_C.md'}")


def figure_c(S, vals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    markers = {"MSE": "o", "MAE": "s", "Huber": "^"}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for i, met in enumerate(("mse", "mae")):
        for j, (H, mode) in enumerate((H, mode) for H in (96, 720) for mode in ("series", "target")):
            ax = axes[i, j]
            for ob in OBJECTIVES:
                base = S.test(GRID, "ETTh1", H, "Linear", ob, met, "new")
                for rate, ls in ((0.01, "--"), (0.1, "-")):
                    ys = [vals[f"{H}_{mode}_{rate:g}_{mag:g}"][ob][0 if met == "mse" else 1] for mag in MAGS]
                    ax.plot([0] + MAGS, [base] + ys, ls=ls, color=COLORS[ob], marker=markers[ob], ms=4,
                            label=f"{ob}{' (δ_MAD)' if ob == 'Huber' else ''}, rate {rate:.0%}")
            ax.set_xticks([0, 3, 5, 10])
            ax.set_title(f"H = {H}, {'chèn vào chuỗi' if mode == 'series' else 'chèn chỉ vào đích'}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{met.upper()} test ở λ*_{met.upper()}")
            if i == 1:
                ax.set_xlabel("độ lớn outlier (× sd; 0 = gốc)")
            _style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle("Linear, ETTh1: sai số test theo độ lớn outlier (seed 2026, cùng vị trí ở mọi độ lớn)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    f = ROOT / "results" / "outliers_mag" / "C_outlier_magnitude.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    print(f"→ {f.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Phần D
# ---------------------------------------------------------------------------

def part_d(summ):
    from scipy import stats
    from optim_etth2 import label, side
    sel = json.loads((ROOT / "results" / "sgd_lr" / "selection.json").read_text(encoding="utf-8"))["cells"]
    old = json.loads((ROOT / "results" / "sgd_b1_b4" / "summary.json").read_text(encoding="utf-8"))
    old_b4 = {(r["dataset"], r["H"], r["model"]): r for r in old["B4"]["rows"]}
    old_c3 = {(r["dataset"], r["H"]): r for r in old["C3"]["rows"]}
    sys.path.insert(0, str(ROOT / "scripts"))
    import sgd_lr_tune as T

    def ci(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

    def welch(a, b):
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        return float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * np.sqrt(va + vb))

    def g(x):
        return f"{x:g}".replace(".", ",")

    def fci(m, h):
        return f"{num(m)} [{num(m - h)}; {num(m + h)}]"

    def runs(ds, m, H, mult, seeds, allow_missing=False):
        rs = [T.find(ds, m, H, mult, s) for s in seeds]
        if not allow_missing:
            assert all(r is not None for r in rs), (ds, m, H, mult)
        return [r[1] for r in rs if r is not None]

    md = ["# Phần D: SGD đã tinh chỉnh learning rate", "",
          "Chọn lr ∈ {0,5; 1; 2; 4} × lr gốc (thêm ×8 nếu ×4 thắng) theo trung bình val MSE của seed 2021–2025; chạy cuối "
          "seed 2026–2040 (15 seed mới). Dạng đóng: MSE λ* theo val MSE (results/results.csv), như B4.", "",
          "## Bảng lr đã chọn", "",
          "| Dataset | Mô hình | H | lr gốc | val MSE TB (seed 2021–2025): ×0,5 / ×1 / ×2 / ×4 / ×8 | chọn | lr chọn |",
          "|---|---|---|---|---|---|---|"]
    chosen = {}
    for ds in T.DATASETS:
        for m in T.MODELS:
            for H in T.HORIZONS:
                c = sel[f"{ds} {m} H={H}"]
                mv = c["mean_val_by_mult"]
                chosen[(ds, m, H)] = c["chosen_mult"]
                md.append(f"| {ds} | {m} | {H} | {g(c['base_lr'])} | " + " / ".join(
                    (f"**{num(mv[k], 5)}**" if float(k) == c["chosen_mult"] else num(mv[k], 5)) if k in mv else "—"
                    for k in ("0.5", "1", "2", "4", "8")) + f" | ×{g(c['chosen_mult'])} | {g(c['chosen_lr'])} |")
    cnt = {}
    for v in chosen.values():
        cnt[v] = cnt.get(v, 0) + 1
    md += ["", "Số ô theo mức chọn: " + ", ".join(f"×{g(k)}: {v}" for k, v in sorted(cnt.items())) + ".", ""]

    # B4-tinh chỉnh
    star = {}
    for r in read(ROOT / "results" / "results.csv"):
        if r["objective"] == "MSE" and r["selection"] == "val_objective" and r["level"] == "lam_star":
            star[(r["dataset"], int(r["H"]), r["model"])] = r
    md += ["## B4-tinh chỉnh: SGD đã tinh chỉnh lr (15 seed 2026–2040) so với dạng đóng λ*", "",
           "Δ = TB SGD − dạng đóng, CI 95% theo t (14 bậc tự do). Cột \"×1, 2026–2040\": lr gốc trên đúng 15 seed đó "
           "(không chạy mới), để tách phần nhãn đổi do lr với phần do bớt seed. B4 cũ: lr gốc, 20 seed.", "",
           "| Dataset | H | Mô hình | lr | Δ val [CI] | Δ test [CI] | nhãn (tinh chỉnh) | nhãn ×1, 2026–2040 | nhãn B4 cũ |",
           "|---|---|---|---|---|---|---|---|---|"]
    b4 = []
    for ds in T.DATASETS:
        for H in T.HORIZONS:
            for m in T.MODELS:
                cf = star[(ds, H, m)]
                cv, ct = float(cf["val_mse"]), float(cf["test_mse"])
                labs = {}
                for tag, mult in (("tuned", chosen[(ds, m, H)]), ("x1", 1.0)):
                    rs = runs(ds, m, H, mult, T.FINAL_SEEDS, allow_missing=(tag == "x1"))
                    mv, hv = ci([r["val_mse"] - cv for r in rs])
                    mt, ht = ci([r["test_mse"] - ct for r in rs])
                    labs[tag] = {"d_val": mv, "d_val_ci": hv, "d_test": mt, "d_test_ci": ht,
                                 "label": label(side(mv, hv), side(mt, ht)), "n": len(rs)}
                o = old_b4[(ds, H, m)]
                row = {"dataset": ds, "H": H, "model": m, "mult": chosen[(ds, m, H)], **{f"{k}_{t}": v for t, d in labs.items() for k, v in d.items()},
                       "label_old": o["label"], "d_test_old": o["d_test"]}
                b4.append(row)
                t = labs["tuned"]
                md.append(f"| {ds} | {H} | {m} | ×{g(chosen[(ds, m, H)])} | {fci(t['d_val'], t['d_val_ci'])} | {fci(t['d_test'], t['d_test_ci'])} | "
                          f"{t['label']} | {labs['x1']['label']}{'' if labs['x1']['n'] == 15 else ' (n = ' + str(labs['x1']['n']) + ')'} | {o['label']} |")
    labels = ["SGD tốt hơn thật", "dạng đóng tốt hơn thật", "khác biệt do val khác test", "SGD có dấu hiệu khớp val quá mức",
              "dạng đóng có dấu hiệu khớp val quá mức", "không phân biệt được"]
    md += ["", "| nhãn | tinh chỉnh (15 seed) | ×1, 2026–2040 | B4 cũ (×1, 20 seed) |", "|---|---|---|---|"]
    counts = {}
    for lb in labels:
        c = [sum(r[k] == lb for r in b4) for k in ("label_tuned", "label_x1", "label_old")]
        counts[lb] = c
        md.append(f"| {lb} | {c[0]} | {c[1]} | {c[2]} |")
    worse_test = sum(r["d_test_tuned"] - r["d_test_ci_tuned"] > 0 for r in b4)
    worse_test_mean = sum(r["d_test_tuned"] > 0 for r in b4)
    changed = [r for r in b4 if r["label_tuned"] != r["label_old"]]
    by_lr = [r for r in changed if r["label_tuned"] != r["label_x1"]]
    md += ["", f"SGD đã tinh chỉnh vẫn kém dạng đóng trên test: {worse_test_mean}/36 ô theo trung bình, {worse_test}/36 ô có ý nghĩa "
           f"(CI nằm trên 0). \"SGD tốt hơn thật\": {counts['SGD tốt hơn thật'][0]} ô "
           f"({', '.join(f'{r['dataset']} H={r['H']} {r['model']}' for r in b4 if r['label_tuned'] == 'SGD tốt hơn thật')}).", "",
           f"Nhãn đổi so với B4 cũ: {len(changed)}/36 ô; trong đó {len(by_lr)} ô nhãn tinh chỉnh khác nhãn ×1 trên cùng 15 seed "
           f"(đổi do lr), {len(changed) - len(by_lr)} ô nhãn ×1 trên 15 seed đã khác B4 cũ (đổi do bớt seed).", "",
           "| Dataset | H | Mô hình | lr | B4 cũ | ×1, 2026–2040 | tinh chỉnh |", "|---|---|---|---|---|---|---|"]
    for r in changed:
        md.append(f"| {r['dataset']} | {r['H']} | {r['model']} | ×{g(r['mult'])} | {r['label_old']} | {r['label_x1']} | {r['label_tuned']} |")
    # Δ test: tinh chỉnh so với ×1 trên cùng seed
    imp = [r["d_test_tuned"] - r["d_test_x1"] for r in b4 if r["mult"] != 1.0]
    md += ["", f"Ở {len(imp)} ô có lr chọn khác ×1: TB test (tinh chỉnh) − TB test (×1), cùng 15 seed: "
           f"{num(min(imp))} đến {num(max(imp))} (âm ở {sum(x < 0 for x in imp)} ô)." if imp else "", ""]
    summ["D_B4"] = {"rows": b4, "counts": counts, "worse_test_sig": worse_test, "worse_test_mean": worse_test_mean,
                    "n_changed": len(changed), "n_changed_by_lr": len(by_lr), "chosen_counts": {f"{k:g}": v for k, v in cnt.items()}}

    # C.3-tinh chỉnh
    md += ["## C.3-tinh chỉnh: DLinear − Linear khi cả hai được tinh chỉnh lr (15 seed mỗi mô hình, CI Welch)", "",
           "| Dataset | H | lr Linear / DLinear | Δ val [CI] | Δ test [CI] | DLinear tốt hơn có ý nghĩa (val / test) | C.3 cũ: Δ test [CI] |",
           "|---|---|---|---|---|---|---|"]
    c3 = []
    for ds in T.DATASETS:
        for H in T.HORIZONS:
            a = runs(ds, "DLinear", H, chosen[(ds, "DLinear", H)], T.FINAL_SEEDS)
            b = runs(ds, "Linear", H, chosen[(ds, "Linear", H)], T.FINAL_SEEDS)
            dv = welch(np.array([r["val_mse"] for r in a]), np.array([r["val_mse"] for r in b]))
            dt = welch(np.array([r["test_mse"] for r in a]), np.array([r["test_mse"] for r in b]))
            o = old_c3[(ds, H)]
            bv, bt = dv[0] + dv[1] < 0, dt[0] + dt[1] < 0
            lv, lt = dv[0] - dv[1] > 0, dt[0] - dt[1] > 0
            c3.append({"dataset": ds, "H": H, "d_val": dv[0], "d_val_ci": dv[1], "d_test": dt[0], "d_test_ci": dt[1],
                       "dlinear_better_val": bv, "dlinear_better_test": bt, "linear_better_val": lv, "linear_better_test": lt,
                       "old_dlinear_better_test": o["dlinear_better_test"]})
            yn = lambda x: "có" if x else "không"
            md.append(f"| {ds} | {H} | ×{g(chosen[(ds, 'Linear', H)])} / ×{g(chosen[(ds, 'DLinear', H)])} | {fci(*dv)} | {fci(*dt)} | "
                      f"{yn(bv)} / {yn(bt)}{' (Linear tốt hơn: ' + yn(lv) + ' / ' + yn(lt) + ')' if lv or lt else ''} | "
                      f"{fci(o['d_test'], o['d_test_ci'])} |")
    n_b = sum(r["dlinear_better_val"] and r["dlinear_better_test"] for r in c3)
    m4 = [r for r in c3 if r["old_dlinear_better_test"]]
    md += ["", f"DLinear tốt hơn có ý nghĩa ở cả val lẫn test: {n_b}/12 ô (cũ: 4/12, đều ETTm1). Bốn ô ETTm1 cũ: "
           + "; ".join(f"H={r['H']}: test {fci(r['d_test'], r['d_test_ci'])} ({'còn' if r['dlinear_better_test'] else 'không còn'} có ý nghĩa)"
                       for r in m4) + ".", ""]
    summ["D_C3"] = {"rows": c3, "n_dlinear_better": n_b}
    (OUT / "tables_D.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {OUT / 'tables_D.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["all"], choices=["B", "A", "C", "D", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "summary.json"
    summ = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    parts = ["B", "A", "C", "D"] if "all" in args.part else args.part
    for p in parts:
        globals()[f"part_{p.lower()}"](summ)
    summ["git_commit"] = git_commit()
    f.write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
