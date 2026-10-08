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
           "Mọi thay đổi đều liên quan Huber hoặc cột không phải metric của chính mục tiêu; phần Huber **chờ Phần A**.", ""]
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
