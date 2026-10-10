"""Sinh lại hình H1–H5 của báo cáo LaTeX thành PDF vector trong report/figures/ (NHIEM_VU_8, B.3).

    python scripts/report/make_figures.py

Không vẽ lại từ đầu: gọi đúng các hàm vẽ đã sinh các hình PNG trong results/ (report_nv7.py, report_sgd_b1_b4.py;
H1 chép từ notebooks/01_three_predictions.ipynb), với cùng dữ liệu đầu vào đọc từ results/. Chỉ khác:
- Figure.savefig được chặn để ghi PDF vào report/figures/ thay vì PNG vào results/ (results/ không bị ghi);
- bỏ tiêu đề chung (suptitle), vì chú thích hình nằm trong LaTeX;
- cùng phông (DejaVu Sans, có dấu tiếng Việt) và cỡ chữ cho mọi hình.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.figure import Figure                             # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)] + [str(ROOT / "scripts" / d) for d in ("grid", "sgd", "checks", "report")]
FIG = ROOT / "report" / "figures"

# tên PNG cũ → tên PDF trong báo cáo
NAMES = {
    "H1_three_predictions.png": "H1_three_predictions.pdf",
    "H2_v2_outlier_robustness.png": "H2_outlier_mse.pdf",
    "H2_v2_outlier_robustness_mae.png": "H2_outlier_mae.pdf",
    "H3_sgd_vs_optimum.png": "H3_sgd_vs_optimum.pdf",
    "A3_delta_sensitivity.png": "H4_delta_sensitivity.pdf",
    "C_outlier_magnitude.png": "H5_outlier_magnitude.pdf",
}

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}

_orig_savefig = Figure.savefig
written = []


def _redirect(self, fname, *a, **kw):
    name = Path(str(fname)).name
    if name not in NAMES:
        raise RuntimeError(f"make_figures: hình lạ {name}")
    if self._suptitle is not None:
        self._suptitle.set_text("")
    out = FIG / NAMES[name]
    kw.pop("dpi", None)
    kw.setdefault("bbox_inches", "tight")
    kw.setdefault("metadata", {"CreationDate": None})           # PDF giống hệt nhau giữa các lần sinh
    _orig_savefig(self, out, *a, format="pdf", **kw)
    written.append(out)


def h1():
    """Như ô vẽ H1 của notebooks/01_three_predictions.ipynb; dữ liệu: results/three_predictions_sweep.csv."""
    import numpy as np
    import pandas as pd
    sweep = pd.read_csv(ROOT / "results" / "three_predictions_sweep.csv")
    KERNELS = [5, 15, 25, 49]
    PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    MARK = ["o", "s", "^", "D"]
    lin0 = sweep[(sweep.model == "Linear") & (sweep.lam == 0)].iloc[0]
    rmse_lin = float(np.sqrt(lin0.mse_test))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    def decorate(ax, title):
        ax.set_xscale("symlog", linthresh=1, linscale=0.6)
        ax.set_yscale("log")
        ax.set_xlim(-0.35, 3e8)
        ax.set_ylim(1e-15, 3)
        ax.set_xticks([0, 1, 1e2, 1e4, 1e6, 1e8])
        ax.set_xticklabels(["0", "1", "10²", "10⁴", "10⁶", "10⁸"])
        ax.axhspan(1e-15, 1e-11, color="0.93", zorder=0)
        ax.text(3e7, 3e-15, "sai số máy float64", ha="right", va="bottom", fontsize=8.5, color="0.4")
        ax.axhline(rmse_lin, ls="--", lw=1, color="0.45")
        ax.text(3e7, rmse_lin * 1.25, f"RMSE test của Linear = {rmse_lin:.3f}".replace(".", ","), ha="right",
                va="bottom", fontsize=8.5, color="0.35")
        ax.set_xlabel("λ (phạt trên tổng bình phương, như hàm ridge)")
        ax.set_title(title)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    ax = axes[0]
    for i, k in enumerate(KERNELS):
        g = sweep[(sweep.model == "DLinear") & (sweep.k == k)]
        ax.plot(g.lam, g.rms_gap_same, color=PAL[i], marker=MARK[i], ms=3.5, lw=1.6, label=f"k = {k}")
        ax.plot(g.lam, g.rms_gap_best, color=PAL[i], ls=":", lw=1.3)
    ax.plot([], [], color="0.3", lw=1.6, label="so với Linear cùng λ")
    ax.plot([], [], color="0.3", ls=":", lw=1.3, label="so với Linear ở λ' khớp nhất")
    ax.set_ylabel("RMS chênh lệch dự báo với Linear (test)")
    ax.legend(loc="center right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.42))
    decorate(ax, "DLinear: trùng Linear tại λ = 0, tách khi λ > 0")

    ax = axes[1]
    g = sweep[sweep.model == "NLinear"]
    ax.plot(g.lam, g.rms_gap_same, color="#8172B3", marker="o", ms=3.5, lw=1.6, label="so với Linear cùng λ")
    ax.plot(g.lam, g.rms_gap_best, color="#8172B3", ls=":", lw=1.3, label="so với Linear ở λ' khớp nhất")
    gap0 = g[g.lam == 0].rms_gap_same.iloc[0]
    ax.axhline(gap0, color="#8172B3", lw=0.8, ls="-.")
    ax.text(1.5, gap0 * 0.55, f"mức tách tại λ = 0: {gap0:.4f}".replace(".", ","), fontsize=8.5, color="#5b4d8a",
            va="top")
    ax.legend(loc="center right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 0.42))
    decorate(ax, "NLinear: tách khỏi Linear ở mọi λ")
    fig.tight_layout()
    fig.savefig("H1_three_predictions.png")
    plt.close(fig)


def _nv7_summary():
    return json.loads((ROOT / "results" / "nv7" / "summary.json").read_text(encoding="utf-8"))


def h2():
    import report_nv7 as N
    N.figure_h2_v2()                                              # MSE test và MAE test


def h4():
    import report_nv7 as N
    N.figure_sensitivity(_nv7_summary()["A_sensitivity"])


def h5():
    import report_nv7 as N
    rows = (N.read(ROOT / "results" / "outliers_mag" / "results_v2.csv")
            + N.read(N.HM / "outliers" / "results_v2.csv") + N.read(N.HM / "results_v2.csv"))
    N.figure_c(N.Sel(N.index(rows)), _nv7_summary()["C"]["values"])


def h2_h4_h5():
    h2()
    h4()
    h5()


def capture(fn):
    """Gọi một hàm vẽ (h1, h2, h3, h4, h5), giữ lại các Figure thay vì ghi file. Dùng trong notebook 02, 03."""
    figs = []

    def keep(self, fname, *a, **kw):
        figs.append(self)

    plt.rcParams.update(STYLE)
    Figure.savefig = keep
    try:
        with contextlib.redirect_stdout(io.StringIO()):     # các hàm vẽ cũ in đường dẫn results/ không còn đúng
            fn()
    finally:
        Figure.savefig = _orig_savefig
    return figs


def capture_png(fn, dpi=100):
    """Như capture nhưng trả ảnh PNG (bytes), hiển thị được trong notebook dù backend là Agg."""

    out = []
    for fig in capture(fn):
        buf = io.BytesIO()
        _orig_savefig(fig, buf, format="png", dpi=dpi, bbox_inches="tight")
        out.append(buf.getvalue())
        plt.close(fig)
    return out


def h3():
    import report_sgd_b1_b4 as B
    star = B.results_star()
    runs = {(ds, H, m): B.load_runs(ds, m, H) for ds in B.DATASETS for H in B.HORIZONS for m in B.MODELS}
    runs = {k: v for k, v in runs.items() if len(v) >= 2}
    B.figure(runs, star)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    Figure.savefig = _redirect
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            h1()
        plt.rcParams.update(STYLE)
        with contextlib.redirect_stdout(io.StringIO()):
            h2_h4_h5()
        plt.rcParams.update(STYLE)
        with contextlib.redirect_stdout(io.StringIO()):
            h3()
    finally:
        Figure.savefig = _orig_savefig
    for f in written:
        print(f"→ {f.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
