"""Sinh report/generated/numbers.tex: mọi con số đo được mà phần chữ của báo cáo LaTeX dùng (NHIEM_VU_8, B.3).

    python scripts/report/make_numbers.py

Đọc results/ qua report_data.py, không ghi gì vào results/. Quy ước đặt tên macro ghi ở đầu file sinh ra.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as D                                          # noqa: E402

OUT = D.ROOT / "report" / "generated" / "numbers.tex"

WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
HEADER = r"""% Tự động sinh bởi scripts/report/make_numbers.py từ results/. KHÔNG SỬA TAY.
%
% Quy ước tên macro: \nm<Nhóm><Tên>, chỉ gồm chữ cái (LaTeX không cho chữ số trong tên lệnh):
%   - mỗi chữ số viết bằng chữ tiếng Anh viết hoa đầu: 1 → One, 96 → NineSix, 720 → SevenTwoZero;
%   - dataset: ETThOne, ETThTwo, ETTmOne; horizon: H + số (HNineSix); mô hình: Linear, DLinear, NLinear;
%   - mục tiêu: MSE, MAE, Huber; metric báo cáo: Mse (MSE test ở λ*_MSE), Mae (MAE test ở λ*_MAE);
%   - nhóm: BTwo (RQ1, bảng B2), BThree (B3, nghiệm tối ưu), Out (outlier), Mag (độ lớn outlier), Delta (δ của Huber),
%     BOne, BFour, CThree (SGD với lr gốc), Tuned (SGD đã tinh chỉnh lr, lưới đã mở rộng), EThTwo (đoạn cảm biến bị kẹt),
%     Drop (code gốc với drop_last), Ridge (họ ridge trên ETTh2), Stuck (vị trí đoạn hằng), DeltaH (δ theo bước h);
%   - hậu tố Min/Max/Range/Count/Pct là khoảng hoặc số đếm; số đếm không có đơn vị.
% Số thập phân dùng dấu phẩy, bọc trong \ensuremath nên dùng được cả trong văn bản lẫn công thức.
"""


def spell(s):
    return "".join(WORDS[int(c)] if c.isdigit() else c for c in str(s) if c.isalnum())


def name(*parts):
    n = "nm" + "".join(spell(p) for p in parts)
    assert n.isalpha(), n
    return n


def fnum(x, p=4, sign=False):
    s = f"{abs(x):.{p}f}".replace(".", "{,}")
    if x < 0 and float(f"{abs(x):.{p}f}") != 0:
        s = "-" + s
    elif sign:
        s = "+" + s
    return rf"\ensuremath{{{s}}}"


def fsci(x, p=0):
    """2e-13 → 2·10^{-13}."""
    m, e = f"{float(x):.{p}e}".split("e")
    return rf"\ensuremath{{{m.replace('.', '{,}')}\cdot 10^{{{int(e)}}}}}"


def flam(x):
    r"""λ: 0, 300, 1{,}778\cdot 10^{4}."""
    x = float(x)
    if x == 0:
        return r"\ensuremath{0}"
    if x < 1e4:
        return rf"\ensuremath{{{f'{x:.4g}'.replace('.', '{,}')}}}"
    m, e = f"{x:.4g}".split("e+") if "e+" in f"{x:.4g}" else (f"{x / 10 ** int(np.log10(x)):.4g}", int(np.log10(x)))
    m = m.replace(".", "{,}")
    return rf"\ensuremath{{{m}\cdot 10^{{{int(e)}}}}}" if m != "1" else rf"\ensuremath{{10^{{{int(e)}}}}}"


def frange(a, b, p=4, sign=False):
    return fnum(a, p, sign) + "--" + fnum(b, p, sign)


class Macros:
    def __init__(self):
        self.d = {}

    def put(self, n, v):
        assert n not in self.d, n
        self.d[n] = v

    def tex(self):
        lines = [HEADER]
        for n, v in self.d.items():
            lines.append(rf"\newcommand{{\{n}}}{{{v}}}")
        return "\n".join(lines) + "\n"


def rq1(M):
    s = D.sweep()
    b2 = D.b2_md_numbers()
    M.put(name("BTwo", "RankZ"), b2["rankZ"])
    M.put(name("BTwo", "NcolZ"), b2["ncolZ"])
    dl0 = [r for r in s if r["model"] == "DLinear" and r["lam"] == 0]
    M.put(name("BTwo", "DLinZeroRelDWMax"), fsci(max(r["rel_dW"] for r in dl0)))
    M.put(name("BTwo", "DLinZeroDyhatMax"), fsci(b2["dlin_dyhat"]))
    M.put(name("BTwo", "DLinZeroGrad"), fsci(b2["dlin_grad"]))
    for k in (5, 15, 25, 49):
        r = D.sweep_at("DLinear", 1e4, float(k))
        M.put(name("BTwo", "DLinRelDWLamFourK", k), fnum(r["rel_dW"], 2))
        M.put(name("BTwo", "DLinRmsLamFourK", k), fnum(r["rms_gap_same"], 3))
        M.put(name("BTwo", "DLinRmsBestLamFourK", k), fnum(r["rms_gap_best"], 3))
    # tỷ lệ rms_gap_best / rms_gap_same của DLinear ở λ > 0 (trung vị)
    ratio = [r["rms_gap_best"] / r["rms_gap_same"] for r in s if r["model"] == "DLinear" and r["lam"] > 0
             and r["rms_gap_same"] > 0]
    M.put(name("BTwo", "DLinBestOverSameMedianPct"), f"{100 * float(np.median(ratio)):.0f}")
    M.put(name("BTwo", "NLinDWF"), fnum(float(b2["nlin_dWF"]), 4))
    M.put(name("BTwo", "NLinRmsZero"), fnum(float(b2["nlin_rms"]), 4))
    M.put(name("BTwo", "NLinDSSE"), fnum(float(b2["nlin_dsse"]), 1))
    M.put(name("BTwo", "NLinRmsLamFour"), fnum(D.sweep_at("NLinear", 1e4)["rms_gap_same"], 3))
    M.put(name("BTwo", "NLinRmsLamEight"), fnum(D.sweep_at("NLinear", 1e8)["rms_gap_same"], 3))
    lin0 = D.sweep_at("Linear", 0.0)
    M.put(name("BTwo", "LinRmseTest"), fnum(np.sqrt(lin0["mse_test"]), 3))
    # DLinear − Linear tại nghiệm tối ưu trên toàn B3 (mục tiêu MSE, MSE test ở λ*_MSE)
    d = [abs(D.b3(ds, H, "DLinear", "MSE", "mse") - D.b3(ds, H, "Linear", "MSE", "mse"))
         for ds in D.DATASETS for H in D.HORIZONS]
    M.put(name("BThree", "DLinMinusLinAbsMax"), fnum(max(d), 4))
    dall = [abs(D.b3(ds, H, "DLinear", ob, met) - D.b3(ds, H, "Linear", ob, met))
            for ds in D.DATASETS for H in D.HORIZONS for ob in D.OBJECTIVES for met in D.METRICS]
    M.put(name("BThree", "DLinMinusLinAllAbsMax"), fnum(max(dall), 4))
    M.put(name("BThree", "DLinMinusLinAllCount"), str(len(dall)))
    # NLinear − Linear tại nghiệm tối ưu (mục tiêu MSE, MSE test ở λ*_MSE), theo dataset
    for ds in D.DATASETS:
        d = [D.b3(ds, H, "NLinear", "MSE", "mse") - D.b3(ds, H, "Linear", "MSE", "mse") for H in D.HORIZONS]
        M.put(name("BThree", "NLinMinusLin", ds, "Range"), frange(min(d, key=abs), max(d, key=abs), 4))
    # closed-form chênh trong C.3 (cf_d_test)
    cf = [abs(r["cf_d_test"]) for r in D.c3_rows().values()]
    M.put(name("CThree", "CfAbsMax"), fnum(max(cf), 4))


def b3_cells(M):
    for ds, H, m in D.cells():
        for ob in D.OBJECTIVES:
            for met in D.METRICS:
                M.put(name("BThree", ds, "H", H, m, ob, met.capitalize()), fnum(D.b3(ds, H, m, ob, met)))
        M.put(name("BThree", ds, "H", H, m, "MSELam"), flam(D.b3_lam(ds, H, m, "MSE", "mse")))
        M.put(name("Delta", ds, "H", H, m), fnum(D.b3_delta(ds, H, m), 3))


def rq2(M):
    for met in D.METRICS:
        bc = D.best_counts(met)
        for ob in D.OBJECTIVES:
            M.put(name("BThree", "Best", ob, met.capitalize(), "Count"), str(bc[ob]))
        hr = D.huber_rank_counts(met)
        for k in (1, 2, 3):
            M.put(name("BThree", "HuberRank", k, met.capitalize(), "Count"), str(hr[k]))
    M.put(name("BThree", "CellCount"), str(len(D.cells())))
    # MAE so với MSE trên MSE test
    for ds in D.DATASETS:
        for m in D.MODELS:
            g = D.mae_minus_mse_gain(ds, m)
            M.put(name("BThree", "MaeBeatsMse", ds, m, "Count"), str(sum(x > 0 for x in g)))
        g = D.mae_minus_mse_gain(ds, "Linear") + D.mae_minus_mse_gain(ds, "DLinear")
        M.put(name("BThree", "MaeGain", ds, "LinDLin", "Range"), frange(min(g), max(g), 3))
    M.put(name("BThree", "MaeBeatsMseTotal", "Count"),
          str(sum(x > 0 for ds in D.DATASETS for m in D.MODELS for x in D.mae_minus_mse_gain(ds, m))))
    g = D.mae_minus_mse_gain("ETTh2", "NLinear")
    M.put(name("BThree", "MaeGainETThTwoNLinear", "Range"), frange(min(g), max(g), 4))
    # δ_MAD
    deltas = [D.b3_delta(*k) for k in D.cells()]
    M.put(name("Delta", "Min"), fnum(min(deltas), 2))
    M.put(name("Delta", "Max"), fnum(max(deltas), 2))
    M.put(name("Delta", "Median"), fnum(float(np.median(deltas)), 2))
    M.put(name("Delta", "RatioMin"), fnum(1 / max(deltas), 1))
    M.put(name("Delta", "RatioMax"), fnum(1 / min(deltas), 1))
    lo, hi = D.nv7()["A_delta_outlier_range"]
    M.put(name("Delta", "Outlier", "Range"), frange(lo, hi, 2))
    # Huber δ_MAD so với δ = 1
    for met in D.METRICS:
        better = sum(D.b3(*k, "Huber", met) < D.b3(*k, "Huber", met, huber="one") for k in D.cells())
        M.put(name("Delta", "MadBetter", met.capitalize(), "Count"), str(better))
    M.put(name("Delta", "ETThTwoLinearHSevenTwoZero", "One"), fnum(D.b3("ETTh2", 720, "Linear", "Huber", "mse", huber="one")))
    M.put(name("Delta", "ETThTwoLinearHSevenTwoZero", "Mad"), fnum(D.b3("ETTh2", 720, "Linear", "Huber", "mse")))
    # độ nhạy theo δ (A.3)
    sens = D.nv7()["A_sensitivity"]
    e2 = [sens[f"ETTh2_720_Linear_{s:g} × δ_MAD"]["test_mse"] for s in (0.5, 1, 2, 4)]
    M.put(name("Delta", "SensETThTwoLinMse", "Range"), frange(min(e2), max(e2), 3))
    rk = [v[f"rank_{met}"] for kk, v in sens.items() if "δ_MAD" in kk for met in D.METRICS]
    rk += [v[f"rank_{met}"] for kk, v in sens.items() if "cố định" in kk for met in D.METRICS]
    M.put(name("Delta", "SensRankTwo", "Count"), str(sum(r == 2 for r in rk)))
    M.put(name("Delta", "Sens", "Count"), str(len(rk)))
    amp = []
    for H in (96, 720):
        for m in ("Linear", "NLinear"):
            v = [sens[f"ETTh1_{H}_{m}_{s:g} × δ_MAD"]["test_mse"] for s in (0.5, 1, 2, 4)]
            amp.append(max(v) - min(v))
    M.put(name("Delta", "SensETThOneAmp", "Range"), frange(min(amp), max(amp), 3))
    mono = 0
    for kk in {"_".join(k.split("_")[:3]) for k in sens}:
        v = [sens[f"{kk}_{s:g} × δ_MAD"]["test_mae"] for s in (0.5, 1, 2, 4)]
        mono += all(a < b for a, b in zip(v, v[1:]))
    M.put(name("Delta", "SensMaeMonotone", "Count"), str(mono))


def outliers(M):
    # chèn chỉ vào đích: thay đổi MSE test so với 0%
    tgt = {ob: [] for ob in D.OBJECTIVES}
    for H in (96, 720):
        for m in ("Linear", "NLinear"):
            for rate in (0.01, 0.1):
                for ob in D.OBJECTIVES:
                    tgt[ob].append(D.outlier("target", rate, H, m, ob, "mse") - D.outlier("target", 0, H, m, ob, "mse"))
    M.put(name("Out", "TargetMSE", "Max"), fnum(max(tgt["MSE"]), 3))
    rob = tgt["MAE"] + tgt["Huber"]
    M.put(name("Out", "TargetRobustAbsMax"), fnum(max(abs(x) for x in rob), 3))
    # chèn vào chuỗi: mục tiêu tốt nhất theo MSE test, 8 ô (Linear, NLinear × 4 H)
    for rate in (0.01, 0.05, 0.1):
        c = {ob: 0 for ob in D.OBJECTIVES}
        for H in D.HORIZONS:
            for m in ("Linear", "NLinear"):
                v = {ob: D.outlier("series", rate, H, m, ob, "mse") for ob in D.OBJECTIVES}
                c[min(v, key=v.get)] += 1
        for ob in D.OBJECTIVES:
            M.put(name("Out", "SeriesBest", ob, "R", f"{rate * 100:g}", "Count"), str(c[ob]))
    deg = {ob: [] for ob in D.OBJECTIVES}
    for H in D.HORIZONS:
        for m in ("Linear", "NLinear"):
            for ob in D.OBJECTIVES:
                deg[ob].append(D.outlier("series", 0.1, H, m, ob, "mse") - D.outlier("series", 0, H, m, ob, "mse"))
    for ob in D.OBJECTIVES:
        M.put(name("Out", "SeriesDeg", ob, "Range"), frange(min(deg[ob]), max(deg[ob]), 3))
    order = sum(deg["MSE"][i] < deg["Huber"][i] < deg["MAE"][i] for i in range(8))
    M.put(name("Out", "SeriesDegOrder", "Count"), str(order))
    # độ lệch theo seed (B.5): Linear H = 96, rate 10%, seed 2026–2028
    sd = {ob: float(np.ptp([D.outlier("series", 0.1, 96, "Linear", ob, "mse", seed=s) for s in (2026, 2027, 2028)]))
          for ob in D.OBJECTIVES}
    M.put(name("Out", "SeedSpread", "Range"), frange(min(sd.values()), max(sd.values()), 3))
    # λ* = 0 trên các đường λ của Huber δ_MAD ở rate 5%, 10% (chuỗi), BAO_CAO_NV7 4.2
    M.put(name("Out", "LamZero", "Count"), str(D.nv7()["A_edges_outlier"]["lam0"]))
    lz = [D._index("outliers/results_v2.csv").get((("series", r, 10.0), 2026, "ETTh1", H, m, ob, D.SEL[met], "lam_star"))
          for r in (0.05, 0.1) for H in D.HORIZONS for m in ("Linear", "NLinear") for ob in ("MSE", "MAE")
          for met in D.METRICS]
    M.put(name("Out", "LamZeroMseMae", "Count"), str(sum(float(x["lam"]) == 0 for x in lz)))
    M.put(name("Out", "LamZeroMseMaeTotal", "Count"), str(len(lz)))
    # độ lớn (Phần C): Linear, ETTh1
    for H in (96, 720):
        for mag in (3.0, 5.0):
            for ob in D.OBJECTIVES:
                M.put(name("Mag", "H", H, "Series", "R", "Ten", "M", int(mag), ob),
                      fnum(D.outlier("series", 0.1, H, "Linear", ob, "mse", mag=mag)))
    for ob in D.OBJECTIVES:
        M.put(name("Mag", "HNineSixSeriesROneM", "Three", ob, "Deg"),
              fnum(D.outlier("series", 0.1, 96, "Linear", ob, "mse", mag=3.0)
                   - D.outlier("series", 0, 96, "Linear", ob, "mse"), 4, sign=True))
    v5 = [D.outlier("series", 0.1, 96, "Linear", ob, met, mag=5.0) for ob in D.OBJECTIVES for met in D.METRICS]
    spread = max(max(v5[0::2]) - min(v5[0::2]), max(v5[1::2]) - min(v5[1::2]))
    M.put(name("Mag", "HNineSixFiveSpread"), fnum(spread, 4))
    tg = [D.outlier("target", 0.1, 96, "Linear", "MSE", "mse", mag=g) - D.outlier("target", 0, 96, "Linear", "MSE", "mse")
          for g in (3.0, 5.0, 10.0)]
    for g, x in zip((3, 5, 10), tg):
        M.put(name("Mag", "TargetMSEDeg", "M", g), fnum(x, 4, sign=True))


def sgd(M):
    S = D.sgd_summary()
    M.put(name("BOne", "Inside", "Count"), str(S["B1"]["n_inside"]))
    rows = D.b1_rows()
    outside = [k for k, r in rows.items() if not r["pub_inside"]]
    M.put(name("BOne", "Outside", "Count"), str(len(outside)))
    M.put(name("BOne", "OutsideDLinear", "Count"), str(sum(k[2] == "DLinear" for k in outside)))
    M.put(name("BOne", "OutsideLinear", "Count"), str(sum(k[2] == "Linear" for k in outside)))
    M.put(name("BOne", "OutsideBelow", "Count"), str(sum(rows[k]["pub"] < rows[k]["min"] for k in outside)))
    for (ds, H, m), r in rows.items():
        M.put(name("BOne", ds, "H", H, m, "Mean"), fnum(r["mean"]))
        M.put(name("BOne", ds, "H", H, m, "Sd"), fnum(r["sd"]))
        M.put(name("BOne", ds, "H", H, m, "Pub"), fnum(r["pub"], 3))
        M.put(name("BOne", ds, "H", H, m, "N"), str(r["n"]))
    # B4, lr gốc
    lab = {"SGD tốt hơn thật": "SgdBetter", "dạng đóng tốt hơn thật": "CfBetter",
           "khác biệt do val khác test": "ValTest", "SGD có dấu hiệu khớp val quá mức": "SgdOverfit",
           "dạng đóng có dấu hiệu khớp val quá mức": "CfOverfit", "không phân biệt được": "Tie"}
    for k, v in S["B4"]["counts"].items():
        M.put(name("BFour", lab[k], "Count"), str(v["total"]))
    b4 = D.b4_rows()
    for ds in ("ETTh1", "ETTm1"):
        d = [r["d_test"] for k, r in b4.items() if k[0] == ds]
        M.put(name("BFour", ds, "DTest", "Range"), frange(min(d), max(d), 4, sign=True))
        M.put(name("BFour", ds, "Worse", "Count"), str(sum(x > 0 for x in d)))
    # B4 tinh chỉnh (lưới lr đã mở rộng, NHIEM_VU_9 A)
    T = D.nv9()["A_B4"]
    for k, v in T["counts"].items():
        M.put(name("Tuned", lab[k], "Count"), str(v[0]))
        M.put(name("Tuned", lab[k], "XOne", "Count"), str(v[1]))
    M.put(name("Tuned", "WorseSig", "Count"), str(T["worse_test_sig"]))
    M.put(name("Tuned", "WorseMean", "Count"), str(T["worse_test_mean"]))
    M.put(name("Tuned", "Changed", "Count"), str(T["n_changed"]))
    M.put(name("Tuned", "ChangedByLr", "Count"), str(T["n_changed_by_lr"]))
    for mult, n in T["chosen_counts"].items():
        M.put(name("Tuned", "Chosen", spell(mult.replace(".", "p")).replace("p", "Point"), "Count"), str(n))
    tb = D.b4t_rows()
    hm = [r for k, r in tb.items() if k[0] in ("ETTh1", "ETTm1")]
    M.put(name("Tuned", "HMOne", "WorseSig", "Count"),
          str(sum(r["d_test_tuned"] - r["d_test_ci_tuned"] > 0 for r in hm)))
    M.put(name("Tuned", "HMOne", "WorseMean", "Count"), str(sum(r["d_test_tuned"] > 0 for r in hm)))
    M.put(name("Tuned", "HMOne", "Cell", "Count"), str(len(hm)))
    d = [r["d_test_tuned"] for r in hm]
    M.put(name("Tuned", "HMOne", "DTest", "Range"), frange(min(d), max(d), 4, sign=True))
    lr = D.lr_cells()
    A = D.nv9()["A"]
    M.put(name("Tuned", "LrEdgeOld", "Count"), str(len(A["changes"])))
    M.put(name("Tuned", "LrEdge", "Count"), str(A["n_still_edge"]))
    M.put(name("Tuned", "LrExtChanged", "Count"), str(A["n_changed"]))
    M.put(name("Tuned", "LrExtWallHours"), fnum(A["wall_hours"], 1))
    # danh sách (văn bản do dữ liệu sinh ra): số ô theo mức chọn; các ô còn ở biên sau khi mở rộng
    g = lambda x: f"{float(x):g}".replace(".", "{,}")                     # noqa: E731
    cc = sorted(T["chosen_counts"].items(), key=lambda kv: float(kv[0]))
    M.put(name("Tuned", "LrChosenList"),
          "; ".join(rf"$\times {g(k)}$: {v} ô" if i == 0 else rf"$\times {g(k)}$: {v}" for i, (k, v) in enumerate(cc)))
    edge = [c for c in A["changes"] if c["still_at_edge"]]
    M.put(name("Tuned", "LrStillEdgeList"),
          "; ".join(rf"{c['cell'].split()[0]} {c['cell'].split()[1]} $H = {c['cell'].split('=')[1]}$ ở $\times {g(c['new_mult'])}$"
                    for c in edge) or "không có")
    n, med = D.lr_noisy_count()
    M.put(name("Tuned", "LrNoisy", "Count"), str(n))
    M.put(name("Tuned", "LrNoisyMedianRatio"), fnum(med, 2))
    M.put(name("Tuned", "LrCell", "Count"), str(len(lr)))
    for m in ("Linear", "DLinear"):
        M.put(name("Tuned", "ETTmOne", m, "ChoseGeEight", "Count"),
              str(sum(float(c["chosen_mult"]) >= 8 for k, c in lr.items() if k[0] == "ETTm1" and k[2] == m)))
        M.put(name("Tuned", "ETTmOne", m, "ChoseLeTwo", "Count"),
              str(sum(float(c["chosen_mult"]) <= 2 for k, c in lr.items() if k[0] == "ETTm1" and k[2] == m)))
    # C.3
    c3 = S["C3"]
    M.put(name("CThree", "DLinBetterBoth", "Count"), str(c3["n_both"]))
    M.put(name("CThree", "LinBetterTest", "Count"), str(c3["n_linear_better_test"]))
    rows = D.c3_rows()
    sig = [r["d_test"] for r in rows.values() if r["dlinear_better_val"] and r["dlinear_better_test"]]
    M.put(name("CThree", "SigDTest", "Range"), frange(max(sig), min(sig), 4))
    for (ds, H), r in rows.items():
        if ds != "ETTm1" and H == 720:
            M.put(name("CThree", ds, "HSevenTwoZero", "DTest"), fnum(r["d_test"], 3, sign=True))
            M.put(name("CThree", ds, "HSevenTwoZero", "DTestLo"), fnum(r["d_test"] - r["d_test_ci"], 3))
            M.put(name("CThree", ds, "HSevenTwoZero", "DTestHi"), fnum(r["d_test"] + r["d_test_ci"], 3))
            pub = D.b1_rows()
            M.put(name("CThree", ds, "HSevenTwoZero", "PubD"),
                  fnum(pub[(ds, 720, "DLinear")]["pub"] - pub[(ds, 720, "Linear")]["pub"], 3))
    t = D.nv9()["A_C3"]
    M.put(name("Tuned", "CThreeDLinBetter", "Count"), str(t["n_dlinear_better"]))
    tr = D.c3t_rows()
    x1 = sum(r["x1_dlinear_better_test"] for k, r in tr.items() if k[0] == "ETTm1")
    M.put(name("Tuned", "CThreeXOneETTmOneTest", "Count"), str(x1))
    em = [r for k, r in tr.items() if k[0] == "ETTm1"]
    M.put(name("Tuned", "CThreeETTmOne", "DTest", "Range"),
          frange(min(r["d_test"] for r in em), max(r["d_test"] for r in em), 4))
    M.put(name("Tuned", "CThreeETTmOneCIHalfMax"),
          fnum(max(max(abs(r["d_test"] - r["d_test_ci"]), abs(r["d_test"] + r["d_test_ci"])) for r in em), 4))


def etth2(M):
    E = D.etth2()
    B = {(r["model"], r["H"], r["metric"]): r for r in E["B"]}
    r = B[("Linear", 720, "test")]
    M.put(name("EThTwo", "CfFullHSevenTwoZero"), fnum(r["cf_full"], 3))
    M.put(name("EThTwo", "CfFiltHSevenTwoZero"), fnum(r["cf_filt"], 3))
    M.put(name("EThTwo", "SgdFullHSevenTwoZero"), fnum(r["sgd_full"], 3))
    M.put(name("EThTwo", "SgdFiltHSevenTwoZero"), fnum(r["sgd_filt"], 3))
    d720 = [B[(m, 720, "test")]["d_cf"] for m in ("Linear", "DLinear")] + \
           [B[(m, 720, "test")]["d_sgd"] for m in ("Linear", "DLinear")]
    M.put(name("EThTwo", "FiltGainHSevenTwoZero", "Range"), frange(-max(d720), -min(d720), 3))
    d96 = [B[(m, 96, "test")][k] for m in ("Linear", "DLinear") for k in ("d_cf", "d_sgd")]
    M.put(name("EThTwo", "FiltLossHNineSix", "Range"), frange(min(d96), max(d96), 2))
    C = {(r["model"], r["H"]): r for r in E["C"]}
    for H in (336, 720):
        M.put(name("EThTwo", "MaeGainRemainPct", "H", H), f"{100 * C[('Linear', H)]['ratio']:.0f}")
    cw = D.constant_windows()
    fr = [c["n_constant"] / c["n_windows"] for c in cw["counts"] if c["split"] == "train"]
    M.put(name("EThTwo", "ConstTrainPct", "Range"), frange(100 * min(fr), 100 * max(fr), 1))
    # số ô H ≥ 336 có SGD cải thiện khi lọc và vẫn tốt hơn dạng đóng
    k = [B[(m, H, "test")] for m in ("Linear", "DLinear") for H in (336, 720)]
    M.put(name("EThTwo", "SgdStillBetter", "Count"), str(sum(not r["mechanism"] for r in k)))


def repro(M):
    """Thời gian chạy đo được (cột seconds của lambda_path.csv và JSON của SGD), cho Phụ lục D."""
    def hours(rel, **flt):
        rows = D.read_csv(D.RES / rel)
        return sum(float(r["seconds"] or 0) for r in rows if all(r[k] == v for k, v in flt.items())) / 3600
    M.put(name("Repro", "GridHours"), fnum(hours("lambda_path.csv"), 1))
    for ds in D.DATASETS:
        M.put(name("Repro", "GridHours", ds), fnum(hours("lambda_path.csv", dataset=ds), 1))
    M.put(name("Repro", "CheckCellMinutes"), f"{60 * hours('lambda_path.csv', dataset='ETTh1', H='96'):.0f}")
    M.put(name("Repro", "HuberMadHours"), fnum(hours("huber_mad/lambda_path.csv"), 1))
    import json
    sec = 0.0
    for d in (D.RES / "sgd_b1_b4" / "ETTh1", D.RES / "sgd_b1_b4" / "ETTm1", D.RES / "sgd_etth2_20seed"):
        for f in d.glob("*_seed*.json"):
            sec += json.loads(f.read_text(encoding="utf-8")).get("seconds", 0.0)
    M.put(name("Repro", "SgdCpuHours"), fnum(sec / 3600, 1))
    sel = D.read_json("sgd_lr/selection.json")
    M.put(name("Repro", "SgdLrWallHours"), fnum(sel["wall_seconds_last_call"] / 3600, 1))
    import subprocess
    # commit cuối cùng đổi results/ (không dùng HEAD, để sinh lại không tạo khác biệt ở mỗi commit)
    head = subprocess.run(["git", "log", "-1", "--format=%h", "--abbrev=7", "--", "results"], cwd=D.ROOT,
                          capture_output=True, text=True).stdout.strip() or "?"
    M.put(name("Repro", "Head"), rf"\texttt{{{head}}}")


def nv9(M):
    """NHIEM_VU_9: code gốc với drop_last (B), họ ridge ETTh2 (C1), vị trí đoạn hằng (C2), δ theo bước h (C3)."""
    import json
    N = D.nv9()
    B = N["B"]
    M.put(name("Drop", "PubInTrue", "Count"), str(B["n_pub_in_dl1"]))
    M.put(name("Drop", "PubInFalse", "Count"), str(B["n_pub_in_dl0"]))
    M.put(name("Drop", "Cell", "Count"), str(B["n_cells"]))
    M.put(name("Drop", "DiffAbsMax"), fsci(B["max_abs_diff"]))
    main = [r for r in B["rows"] if not r["control"]]
    M.put(name("Drop", "Dropped", "Range"), f"{min(r['n_dropped'] for r in main)}--{max(r['n_dropped'] for r in main)}")
    M.put(name("Drop", "DroppedPctMax"), fnum(100 * max(r["n_dropped"] / r["n_test"] for r in main), 2))
    below = [r for r in main if r["pub"] < r["dl1_min"]]
    M.put(name("Drop", "PubBelowTrue", "Count"), str(len(below)))
    gap = [r["dl1_min"] - r["pub"] for r in below]
    if gap:
        M.put(name("Drop", "PubGap", "Range"), frange(min(gap), max(gap), 4))
    M.put(name("Drop", "Seed", "Count"), str(len(main[0]["dl1"])))
    # code gốc DLinear ETTm1 H = 96 trên CPU (NHIEM_VU_6 D.1)
    d1 = [json.loads(f.read_text(encoding="utf-8"))["official_test_mse"]
          for f in sorted((D.RES / "outliers" / "d1_official").glob("DLinear_H96_seed*.json"))]
    M.put(name("Off", "ETTmOneHNineSix", "Mean"), fnum(float(np.mean(d1))))
    M.put(name("Off", "ETTmOneHNineSix", "Sd"), fnum(float(np.std(d1, ddof=1))))
    M.put(name("Off", "ETTmOneHNineSix", "N"), str(len(d1)))
    # C1
    C1 = N["C1"]
    M.put(name("Ridge", "BestWorseOrig", "Count"), str(C1["n_ridge_best_worse_than_sgd_orig"]))
    M.put(name("Ridge", "BestWorseTuned", "Count"), str(C1["n_ridge_best_worse_than_sgd_tuned"]))
    M.put(name("Ridge", "Cell", "Count"), str(C1["n_cells"]))
    g = [r["best_minus_sgd_orig"] for r in C1["rows"]]
    M.put(name("Ridge", "GapOrig", "Range"), frange(min(g), max(g), 4))
    g = [r["best_minus_sgd_tuned"] for r in C1["rows"]]
    M.put(name("Ridge", "GapTuned", "Range"), frange(min(g), max(g), 4))
    g = [r["star_test"] - r["best_test"] for r in C1["rows"]]
    M.put(name("Ridge", "StarMinusBest", "Max"), fnum(max(g), 4))
    # C2
    C2 = N["C2"]
    runs = C2["runs"]
    M.put(name("Stuck", "Run", "Count"), str(len(runs)))
    M.put(name("Stuck", "RunLong", "Count"), str(sum(r["hours"] >= 96 for r in runs)))
    M.put(name("Stuck", "DayThirtyOne", "Count"),
          str(sum(r["hours"] == 24 and r["start"][8:13] == "31 00" for r in runs)))
    M.put(name("Stuck", "RunMaxHours"), str(max(r["hours"] for r in runs)))
    for sp in ("train", "val", "test"):
        M.put(name("Stuck", "RunLong", sp.capitalize(), "Count"),
              str(sum(r["hours"] >= 96 and sp in r["split"].split("/") for r in runs)))
    # C3
    C3 = N["C3"]
    M.put(name("DeltaH", "RankChanged", "Count"), str(C3["n_rank_changed"]))
    M.put(name("DeltaH", "Cell", "Count"), str(len(C3["rows"])))
    for r in C3["rows"]:
        k = (r["dataset"], "H", r["H"], r["model"])
        M.put(name("DeltaH", *k, "Min"), fnum(r["delta_h_min"], 3))
        M.put(name("DeltaH", *k, "Max"), fnum(r["delta_h_max"], 3))
        M.put(name("DeltaH", *k, "Mad"), fnum(r["delta_mad"], 3))
        for hub, tag in (("Huber δ_MAD", "HubMad"), ("Huber δ_h", "HubH")):
            for met in D.METRICS:
                M.put(name("DeltaH", *k, tag, met.capitalize()), fnum(r["test"][hub][met]))
    M.put(name("Repro", "DeltaHMinutes"), f"{C3['meta']['wall_seconds'] / 60:.0f}")
    sec = sum(json.loads(f.read_text(encoding="utf-8"))["seconds"]
              for f in (D.RES / "official_ltsf_droplast").glob("*/*_dl[01].json"))
    M.put(name("Repro", "DropGpuHours"), fnum(sec / 3600, 1))


def main():
    M = Macros()
    repro(M)
    rq1(M)
    b3_cells(M)
    rq2(M)
    outliers(M)
    sgd(M)
    etth2(M)
    nv9(M)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(M.tex(), encoding="utf-8")
    print(f"→ {OUT.relative_to(D.ROOT).as_posix()} ({len(M.d)} macro)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
