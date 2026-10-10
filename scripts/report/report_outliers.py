"""Lập bảng và hình của docs/reports/BAO_CAO_OUTLIER.md (docs/tasks/NHIEM_VU_6.md, Phần E).

    python scripts/grid/select_lambda.py --in results/outliers/lambda_path.csv --out results/outliers/results.csv
    python scripts/report/report_outliers.py

Đọc:
    results/outliers/results.csv      λ* của train có outlier (select_lambda trên results/outliers/lambda_path.csv)
    results/outliers/lambda_path.csv  số vòng, converged, số điểm chèn, tỷ lệ cửa sổ bị ảnh hưởng
    results/results.csv               rate 0 (grid trên dữ liệu gốc)
    results/outliers/d1_official/     D.1 (code gốc, DLinear ETTm1 H = 96)
    results/outliers/d2_lr/           D.2 (Linear ETTm1, lr gấp đôi); so với results/sgd_b1_b4/ETTm1/
Ghi results/outliers/{tables.md, summary.json, H2_outlier_robustness.png, H2_outlier_robustness_mae.png}.

λ* chọn trên val sạch (val không bị chèn). Bảng chính dùng cách chọn val_objective; val_mse ở bảng phụ.
Δ = metric ở rate r − metric ở rate 0 (cùng mô hình, H, mục tiêu, cách chọn λ*). Test không dùng để chọn gì.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / d) for d in ("grid", "sgd", "checks", "report")]
from bench_5060ti import save_json                          # noqa: E402

OUT = ROOT / "results" / "outliers"
MODELS = ["Linear", "NLinear"]
OBJECTIVES = ["MSE", "MAE", "Huber"]
HORIZONS = [96, 192, 336, 720]
RATES = [0.01, 0.05, 0.1]
SEED = 2026
B5_SEEDS = [2026, 2027, 2028]
PUB_D1 = 0.299                                    # Bảng 2 của Zeng et al., DLinear ETTm1 H = 96


def num(x, p=4):
    return f"{x:.{p}f}".replace(".", ",") if x >= 0 else f"−{abs(x):.{p}f}".replace(".", ",")


def sgn(x, p=4):
    return ("+" if x >= 0 else "") + num(x, p)


def pct(r):
    return f"{r * 100:g}%"


def lam_fmt(x):
    return "0" if x == 0 else f"{x:.4g}".replace(".", ",")


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * np.sqrt(va + vb))


def fci(m, h):
    return f"{sgn(m)} [{num(m - h)}; {num(m + h)}]"


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load():
    """star[(mode, rate, seed, sel, model, H, obj)] = dòng λ* (rate 0: mode '-', seed '-')."""
    star = {}
    for r in read_csv(ROOT / "results" / "results.csv"):
        if r["dataset"] == "ETTh1" and r["model"] in MODELS and r["level"] == "lam_star":
            star[("-", 0.0, "-", r["selection"], r["model"], int(r["H"]), r["objective"])] = r
    for r in read_csv(OUT / "results.csv"):
        if r["level"] == "lam_star":
            star[(r["outlier_mode"], float(r["outlier_rate"]), int(r["outlier_seed"]), r["selection"], r["model"],
                  int(r["H"]), r["objective"])] = r
    path = read_csv(OUT / "lambda_path.csv")
    return star, path


def get(star, mode, rate, seed, sel, m, H, o):
    if rate == 0:
        return star.get(("-", 0.0, "-", sel, m, H, o))
    return star.get((mode, rate, seed, sel, m, H, o))


def main():
    star, path = load()
    md, summ = [], {}

    # ---------------- thiết kế chèn: số điểm, tỷ lệ cửa sổ ----------------
    md += ["## Thiết kế chèn: số điểm và tỷ lệ cửa sổ train bị ảnh hưởng", "",
           "Tỷ lệ cửa sổ = tỷ lệ cửa sổ train (một cửa sổ cho mỗi kênh, như make_dataset) có ít nhất một outlier "
           "trong x, trong y.", "",
           "| Chế độ | rate | seed | H | số điểm chèn | cửa sổ có outlier trong x | trong y |", "|---|---|---|---|---|---|---|"]
    design = {}
    for r in path:
        k = (r["outlier_mode"], float(r["outlier_rate"]), int(r["outlier_seed"]), int(r["H"]))
        design.setdefault(k, (int(r["n_contaminated"]), float(r["frac_windows_x"]), float(r["frac_windows_y"])))
    for k in sorted(design, key=lambda k: (k[0] != "series", k[2], k[1], k[3])):
        n, fx, fy = design[k]
        md.append(f"| {k[0]} | {pct(k[1])} | {k[2]} | {k[3]} | {n} | {num(fx, 3)} | {num(fy, 3)} |")
    summ["design"] = {f"{k[0]} r{k[1]:g} s{k[2]} H{k[3]}": v for k, v in design.items()}

    # ---------------- hội tụ ----------------
    conv = {}
    for r in path:
        k = (r["outlier_mode"], float(r["outlier_rate"]), int(r["outlier_seed"]), r["model"], int(r["H"]),
             r["objective"])
        c = conv.setdefault(k, [0, 0, 0, 0])
        c[0] += 1
        c[1] += r["converged"] != "True"
        c[2] += int(r["n_iter"] or 0)
        c[3] += int(r["n_fallback"] or 0) if r.get("n_fallback") else 0
    bad = {k: v for k, v in conv.items() if v[1]}
    md += ["", "## Hội tụ", "",
           f"{len(conv)} đường λ. Số dòng converged = False: {sum(v[1] for v in conv.values())}"
           + (" (" + "; ".join(f"{k[0]} {pct(k[1])} s{k[2]} {k[3]} H={k[4]} {k[5]}: {v[1]}/{v[0]}"
                               for k, v in bad.items()) + ")" if bad else "")
           + f". Đường có quá 2 λ không hội tụ: {sum(v[1] > 2 for v in conv.values())}. "
           f"Bước h cần fallback float64 (tổng): {sum(v[3] for v in conv.values())}."]
    summ["convergence"] = {f"{k[0]} r{k[1]:g} s{k[2]} {k[3]} H{k[4]} {k[5]}": v for k, v in conv.items()}

    # ---------------- bảng chính (series, seed 2026, val_objective) ----------------
    for sel in ("val_objective", "val_mse"):
        title = ("## Bảng chính: chèn vào chuỗi gốc (seed 2026), λ* theo val_objective" if sel == "val_objective"
                 else "## Bảng phụ: như bảng chính nhưng λ* theo val_mse")
        md += ["", title, "",
               "Δ so với rate 0 (dữ liệu gốc, cùng cách chọn λ*). λ* chọn trên val sạch.", "",
               "| Mô hình | H | Mục tiêu | rate | λ* | MSE test | MAE test | Δ MSE | Δ MAE |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            for H in HORIZONS:
                for o in OBJECTIVES:
                    base = get(star, "series", 0, SEED, sel, m, H, o)
                    for rate in [0.0] + RATES:
                        r = get(star, "series", rate, SEED, sel, m, H, o)
                        if r is None or base is None:
                            continue
                        t, a = float(r["test_mse"]), float(r["test_mae"])
                        dt, da = t - float(base["test_mse"]), a - float(base["test_mae"])
                        md.append(f"| {m} | {H} | {o} | {pct(rate)} | {lam_fmt(float(r['lam']))}"
                                  f"{' (đầu mút)' if r['lam_at_edge'] == 'True' else ''} | {num(t)} | {num(a)} | "
                                  f"{'—' if rate == 0 else sgn(dt)} | {'—' if rate == 0 else sgn(da)} |")
                        summ.setdefault(sel, {})[f"{m} H{H} {o} r{rate:g}"] = {
                            "lam": float(r["lam"]), "test_mse": t, "test_mae": a, "d_mse": dt, "d_mae": da}

    # ---------------- tóm tắt: mục tiêu tốt nhất và suy giảm 0 → 10% ----------------
    md += ["", "## Tóm tắt theo mục tiêu (series, seed 2026, val_objective)", "",
           "Suy giảm = MSE (MAE) test ở rate 10% − ở rate 0. \"Tốt nhất\" = mục tiêu có MSE test thấp nhất trong ô.", "",
           "| Mô hình | H | suy giảm MSE: MSE / MAE / Huber | suy giảm MAE: MSE / MAE / Huber | tốt nhất (MSE test) ở 0% / 1% / 5% / 10% |",
           "|---|---|---|---|---|"]
    best_count = {}
    for m in MODELS:
        for H in HORIZONS:
            dm, da, bests = [], [], []
            for o in OBJECTIVES:
                b = get(star, "series", 0, SEED, "val_objective", m, H, o)
                r = get(star, "series", 0.1, SEED, "val_objective", m, H, o)
                dm.append(sgn(float(r["test_mse"]) - float(b["test_mse"])) if r and b else "—")
                da.append(sgn(float(r["test_mae"]) - float(b["test_mae"])) if r and b else "—")
            for rate in [0.0] + RATES:
                vals = {o: get(star, "series", rate, SEED, "val_objective", m, H, o) for o in OBJECTIVES}
                if all(vals.values()):
                    w = min(OBJECTIVES, key=lambda o: float(vals[o]["test_mse"]))
                    bests.append(w)
                    best_count[(m, rate, w)] = best_count.get((m, rate, w), 0) + 1
                else:
                    bests.append("—")
            md.append(f"| {m} | {H} | {' / '.join(dm)} | {' / '.join(da)} | {' / '.join(bests)} |")
    summ["best_count"] = {f"{k[0]} r{k[1]:g} {k[2]}": v for k, v in best_count.items()}

    # ---------------- C: series so với target ----------------
    md += ["", "## C. Chèn vào chuỗi gốc so với chèn chỉ vào đích (seed 2026, val_objective)", "",
           "| Mô hình | H | Mục tiêu | rate | series: MSE / MAE test | target: MSE / MAE test | Δ MSE so với 0%: series / target |",
           "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for H in (96, 720):
            for o in OBJECTIVES:
                b = get(star, "series", 0, SEED, "val_objective", m, H, o)
                for rate in (0.01, 0.1):
                    s = get(star, "series", rate, SEED, "val_objective", m, H, o)
                    t = get(star, "target", rate, SEED, "val_objective", m, H, o)
                    if not (s and t and b):
                        continue
                    md.append(f"| {m} | {H} | {o} | {pct(rate)} | {num(float(s['test_mse']))} / "
                              f"{num(float(s['test_mae']))} | {num(float(t['test_mse']))} / {num(float(t['test_mae']))} | "
                              f"{sgn(float(s['test_mse']) - float(b['test_mse']))} / "
                              f"{sgn(float(t['test_mse']) - float(b['test_mse']))} |")
                    summ.setdefault("target", {})[f"{m} H{H} {o} r{rate:g}"] = {
                        "series_test_mse": float(s["test_mse"]), "target_test_mse": float(t["test_mse"]),
                        "series_test_mae": float(s["test_mae"]), "target_test_mae": float(t["test_mae"])}

    # ---------------- B.5: độ nhạy theo seed chèn ----------------
    md += ["", "## B.5. Độ nhạy theo seed chèn (Linear, H = 96, series, val_objective)", "",
           "Độ lệch giữa seed = max − min của MSE test trên 3 seed (2026–2028). Khác biệt giữa mục tiêu = max − min "
           "của trung bình 3 seed giữa ba mục tiêu.", "",
           "| rate | Mục tiêu | MSE test theo seed 2026 / 2027 / 2028 | TB ± sd | max − min | MAE test: TB ± sd | max − min |",
           "|---|---|---|---|---|---|---|"]
    b5 = {}
    for rate in (0.01, 0.1):
        means = {}
        for o in OBJECTIVES:
            rs = [get(star, "series", rate, s, "val_objective", "Linear", 96, o) for s in B5_SEEDS]
            if not all(rs):
                continue
            t = np.array([float(r["test_mse"]) for r in rs])
            a = np.array([float(r["test_mae"]) for r in rs])
            means[o] = (t.mean(), a.mean())
            b5[f"r{rate:g} {o}"] = {"test_mse": t.tolist(), "test_mae": a.tolist()}
            md.append(f"| {pct(rate)} | {o} | {' / '.join(num(x) for x in t)} | {num(t.mean())} ± {num(t.std(ddof=1))} | "
                      f"{num(np.ptp(t))} | {num(a.mean())} ± {num(a.std(ddof=1))} | {num(np.ptp(a))} |")
        if len(means) == 3:
            bt = max(v[0] for v in means.values()) - min(v[0] for v in means.values())
            ba = max(v[1] for v in means.values()) - min(v[1] for v in means.values())
            spread = max(np.ptp(b5[f"r{rate:g} {o}"]["test_mse"]) for o in OBJECTIVES)
            md.append(f"| {pct(rate)} | khác biệt giữa mục tiêu | | | MSE {num(bt)} (độ lệch seed lớn nhất {num(spread)}) | "
                      f"| MAE {num(ba)} |")
            b5[f"r{rate:g} between_objectives"] = {"mse": bt, "mae": ba, "max_seed_range_mse": spread}
    summ["b5"] = b5

    # ---------------- D.1 ----------------
    d1 = sorted(glob.glob(str(OUT / "d1_official" / "*.json")))
    repo = [json.load(open(f, encoding="utf-8")) for f in
            sorted(glob.glob(str(ROOT / "results" / "sgd_b1_b4" / "ETTm1" / "DLinear_H96_seed*.json")))]
    rt = np.array([r["test_mse"] for r in repo])
    if d1:
        md += ["", "## D.1. Code gốc LTSF-Linear, DLinear ETTm1 H = 96 (CPU)", "",
               f"Repo (SGD, 20 seed): {num(rt.mean())} ± {num(rt.std(ddof=1))}, [{num(rt.min())}; {num(rt.max())}]. "
               f"Công bố: {num(PUB_D1, 3)}.", "",
               "| seed | code gốc: MSE / MAE test | chấm lại bằng pipeline repo: MSE test | epoch tốt nhất | thời gian (s) |",
               "|---|---|---|---|---|"]
        offs = []
        for f in d1:
            r = json.load(open(f, encoding="utf-8"))
            offs.append(r["official_test_mse"])
            md.append(f"| {r['seed']} | {num(r['official_test_mse'])} / {num(r['official_test_mae'])} | "
                      f"{num(r['repo_test_mse'])} | {r['best_epoch']}/{r['n_epochs_run']} | {r['seconds']:.0f} |")
        offs = np.array(offs)
        d, h = welch(offs, rt) if len(offs) > 1 else (float("nan"), float("nan"))
        md += ["", f"Code gốc: {num(offs.mean())} ± {num(offs.std(ddof=1)) if len(offs) > 1 else '—'} (n = {len(offs)}); "
               f"số seed dưới 0,300: {int((offs < 0.300).sum())}/{len(offs)}; code gốc − repo: {fci(d, h)} (Welch)."]
        summ["d1"] = {"official": offs.tolist(), "repo_mean": float(rt.mean()), "welch": [d, h]}

    # ---------------- D.2 ----------------
    d2dirs = sorted(glob.glob(str(OUT / "d2_lr" / "ETTm1_lr*")))
    if d2dirs:
        md += ["", "## D.2. Linear ETTm1 với lr lớn hơn (SGD, seed 2021–2040)", "",
               "Δ = TB(Linear, lr mới) − TB(nhóm so sánh), CI 95% Welch. Linear và DLinear lr gốc 0,0001 lấy từ "
               "results/sgd_b1_b4/ETTm1/.", "",
               "| lr | H | n | Linear lr mới: val / test (TB) | − DLinear lr gốc: Δ val | Δ test | − Linear lr gốc: Δ val | Δ test | còn kém DLinear có ý nghĩa? (val / test) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for dd in d2dirs:
            lr = dd.split("_lr")[-1]
            for H in HORIZONS:
                new = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{dd}/Linear_H{H}_seed*.json"))]
                if len(new) < 2:
                    continue
                ref = {m: [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(
                    str(ROOT / "results" / "sgd_b1_b4" / "ETTm1" / f"{m}_H{H}_seed*.json")))] for m in ("DLinear", "Linear")}
                row = {}
                cells = []
                for m in ("DLinear", "Linear"):
                    for k in ("val_mse", "test_mse"):
                        d, h = welch([r[k] for r in new], [r[k] for r in ref[m]])
                        row[f"{m} {k}"] = (d, h)
                        cells.append(fci(d, h))
                worse = ["có" if row[f"DLinear {k}"][0] - row[f"DLinear {k}"][1] > 0 else "không"
                         for k in ("val_mse", "test_mse")]
                md.append(f"| {lr.replace('.', ',')} | {H} | {len(new)} | {num(np.mean([r['val_mse'] for r in new]))} / "
                          f"{num(np.mean([r['test_mse'] for r in new]))} | {' | '.join(cells)} | {' / '.join(worse)} |")
                summ.setdefault("d2", {})[f"lr{lr} H{H}"] = {k: list(v) for k, v in row.items()} | {"n": len(new)}

    figure(star, "test_mse", "H2_outlier_robustness.png", "MSE test")
    figure(star, "test_mae", "H2_outlier_robustness_mae.png", "MAE test")
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_json(OUT / "summary.json", summ)
    print("\n".join(md))


def figure(star, metric, fname, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"MSE": "#2a78d6", "MAE": "#eb6834", "Huber": "#1baf7a"}
    markers = {"MSE": "o", "MAE": "s", "Huber": "^"}
    fig, axes = plt.subplots(len(MODELS), len(HORIZONS), figsize=(14, 6.8))
    for i, m in enumerate(MODELS):
        for j, H in enumerate(HORIZONS):
            ax = axes[i, j]
            for o in OBJECTIVES:
                for mode, ls, rates in (("series", "-", [0.0] + RATES), ("target", "--", [0.0, 0.01, 0.1])):
                    pts = [(r * 100, float(x[metric])) for r in rates
                           if (x := get(star, mode, r, SEED, "val_objective", m, H, o)) is not None]
                    if len(pts) < 2:
                        continue
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, ls=ls, color=colors[o], marker=markers[o], ms=4 if mode == "series" else 3,
                            lw=1.6 if mode == "series" else 1.2, alpha=1 if mode == "series" else 0.8,
                            label=f"{o}, chèn vào chuỗi" if mode == "series" else f"{o}, chèn chỉ vào đích")
            ax.set_title(f"{m}, H = {H}", fontsize=10)
            ax.set_xticks([0, 1, 5, 10])
            ax.grid(color="#e5e5e5", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{label} (dữ liệu chuẩn hóa)")
            if i == len(MODELS) - 1:
                ax.set_xlabel("tỷ lệ outlier trong train (%)")
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{label} theo tỷ lệ outlier, ETTh1, λ* theo val sạch (val_objective); outlier ±10 sd, seed 2026",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"→ {(OUT / fname).relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
