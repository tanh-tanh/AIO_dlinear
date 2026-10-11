"""Sinh mọi bảng của báo cáo LaTeX thành report/generated/tab_*.tex (NHIEM_VU_8, B.3).

    python scripts/report/make_tables.py

booktabs, số thập phân dấu phẩy, 4 chữ số sau dấu phẩy cho MSE/MAE. Đánh đậm theo đúng quy tắc của các báo cáo kỹ
thuật: trong B3 và các bảng outlier, giá trị thấp nhất giữa ba mục tiêu của cùng (dataset, H, mô hình, metric).
Mỗi file chỉ chứa phần tabular/longtable; chú thích và nhãn nằm trong sections/*.tex.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_data as D                                          # noqa: E402

OUT = D.ROOT / "report" / "generated"
HEAD = "% Tự động sinh bởi scripts/report/make_tables.py từ results/. KHÔNG SỬA TAY.\n"


def n(x, p=4, sign=False):
    s = f"{abs(x):.{p}f}".replace(".", ",")
    if x < 0 and float(f"{abs(x):.{p}f}") != 0:
        return "$-$" + s
    return ("+" if sign else "") + s


def ci(m, h, p=4):
    return f"{n(m, p, True)} [{n(m - h, p)}; {n(m + h, p)}]"


def lam(x):
    x = float(x)
    if x == 0:
        return "0"
    if x < 1e4:
        return f"{x:.4g}".replace(".", ",")
    m, e = f"{x:.3e}".split("e+")
    m = f"{float(m):.4g}".replace(".", ",")
    return f"${m}\\cdot 10^{{{int(e)}}}$" if m != "1" else f"$10^{{{int(e)}}}$"


def bold_min(vals, p=4):
    lo = min(vals)
    return [f"\\textbf{{{n(v, p)}}}" if v == lo else n(v, p) for v in vals]


def write(name, body):
    body = re.sub(r"(\d),(\d)", r"\1{,}\2", body)       # dấu phẩy thập phân dùng được cả trong công thức
    f = OUT / f"tab_{name}.tex"
    f.write_text(HEAD + body, encoding="utf-8")
    print(f"→ {f.relative_to(D.ROOT).as_posix()}")


LABEL = {"SGD tốt hơn thật": "SGD tốt hơn thật", "dạng đóng tốt hơn thật": "dạng đóng tốt hơn thật",
         "khác biệt do val khác test": "val khác test", "SGD có dấu hiệu khớp val quá mức": "SGD khớp val quá mức",
         "dạng đóng có dấu hiệu khớp val quá mức": "dạng đóng khớp val quá mức",
         "không phân biệt được": "không phân biệt được"}


# ---------------------------------------------------------------------------

def tab_b2():
    s = D.sweep()
    b2 = D.b2_md_numbers()
    dl0 = max(r["rel_dW"] for r in s if r["model"] == "DLinear" and r["lam"] == 0)
    r4 = [D.sweep_at("DLinear", 1e4, float(k)) for k in (5, 15, 25, 49)]
    rel = "/".join(n(r["rel_dW"], 2) for r in r4)
    body = r"""\begin{tabular}{@{}p{0.2\linewidth}p{0.36\linewidth}p{0.36\linewidth}@{}}
\toprule
 & $\lambda = 0$ & $\lambda > 0$ \\
\midrule
DLinear so với Linear & \textbf{Trùng} với cả 4 kernel $k$: $\operatorname{rank} Z = %s/%s$; $\|\Delta W\|/\|W\| \le %s$; gradient qua module $\approx %s$ lần lúc khởi tạo. & \textbf{Tách}, mức tách tăng theo $k$. Tại $\lambda = 10^4$, $k = 5/15/25/49$: $\|\Delta W\|/\|W\| = %s$; không $\lambda'$ nào của Linear xóa được phần tách. \\
\addlinespace
NLinear so với Linear & \textbf{Tách}: $\|\Delta W\|_F = %s$; RMS chênh dự báo test $%s$. & \textbf{Tách}, tăng theo $\lambda$: RMS chênh $%s$ ($\lambda = 10^4$) đến $%s$ ($\lambda = 10^8$). \\
\bottomrule
\end{tabular}
""" % (b2["rankZ"], b2["ncolZ"], sci(dl0), sci(b2["dlin_grad"]), rel, n(float(b2["nlin_dWF"]), 4),
       n(float(b2["nlin_rms"]), 4), n(D.sweep_at("NLinear", 1e4)["rms_gap_same"], 3),
       n(D.sweep_at("NLinear", 1e8)["rms_gap_same"], 3))
    write("b2", body)


def sci(x):
    m, e = f"{float(x):.0e}".split("e")
    return f"{m}\\cdot 10^{{{int(e)}}}"


def tab_b3_main():
    rows = [r"\begin{longtable}{@{}llr rrr rrr@{}}", r"\toprule",
            r" & & & \multicolumn{3}{c}{MSE test ở $\lambda^*_{\mathrm{MSE}}$} & \multicolumn{3}{c}{MAE test ở $\lambda^*_{\mathrm{MAE}}$} \\",
            r"\cmidrule(lr){4-6}\cmidrule(l){7-9}",
            r"Dataset & Mô hình & $H$ & MSE & MAE & Huber & MSE & MAE & Huber \\", r"\midrule", r"\endhead"]
    for ds in D.DATASETS:
        for m in D.MODELS:
            for H in D.HORIZONS:
                v = [D.b3(ds, H, m, ob, met) for met in D.METRICS for ob in D.OBJECTIVES]
                c = bold_min(v[:3]) + bold_min(v[3:])
                rows.append(f"{ds} & {m} & {H} & " + " & ".join(c) + r" \\")
            rows.append(r"\addlinespace[2pt]")
        rows.append(r"\midrule" if ds != D.DATASETS[-1] else "")
    rows += [r"\bottomrule", r"\end{longtable}"]
    write("b3_main", "\n".join(x for x in rows if x) + "\n")


def tab_b3_full():
    rows = [r"{\footnotesize", r"\begin{longtable}{@{}lrll rr rr rr r@{}}", r"\toprule",
            r" & & & & \multicolumn{2}{c}{$\lambda = 0$} & \multicolumn{2}{c}{MSE test} & \multicolumn{2}{c}{MAE test} & \\",
            r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
            r"Dataset & $H$ & Mô hình & Mục tiêu & MSE & MAE & $\lambda^*_{\mathrm{MSE}}$ & giá trị & $\lambda^*_{\mathrm{MAE}}$ & giá trị & $\delta$ \\",
            r"\midrule", r"\endhead"]
    for ds in D.DATASETS:
        for H in D.HORIZONS:
            for m in D.MODELS:
                best = {met: min(D.b3(ds, H, m, ob, met) for ob in D.OBJECTIVES) for met in D.METRICS}
                for ob in D.OBJECTIVES:
                    z = [n(D.b3(ds, H, m, ob, met, level="lam0")) for met in D.METRICS]
                    cell = []
                    for met in D.METRICS:
                        v = D.b3(ds, H, m, ob, met)
                        cell += [lam(D.b3_lam(ds, H, m, ob, met)), f"\\textbf{{{n(v)}}}" if v == best[met] else n(v)]
                    dl = "--" if ob == "MSE" else ("$10^{-6}$" if ob == "MAE" else n(D.b3_delta(ds, H, m), 3))
                    rows.append(f"{ds} & {H} & {m} & {ob} & " + " & ".join(z + cell) + f" & {dl} \\\\")
                rows.append(r"\addlinespace[2pt]")
    rows += [r"\bottomrule", r"\end{longtable}", "}"]
    write("b3_full", "\n".join(rows) + "\n")


def tab_b1():
    B = D.b1_rows()
    rows = [r"\begin{tabular}{@{}lr ll ll ll@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{DLinear} & \multicolumn{2}{c}{NLinear} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
            r"Dataset & $H$ & SGD: TB $\pm$ sd & công bố & SGD: TB $\pm$ sd & công bố & SGD: TB $\pm$ sd & công bố \\",
            r"\midrule"]
    for ds in D.DATASETS:
        for H in D.HORIZONS:
            c = []
            for m in D.MODELS:
                r = B[(ds, H, m)]
                pub = n(r["pub"], 3)
                if not r["pub_inside"]:
                    pub = f"\\textbf{{{pub}}}$^\\dagger$"
                sd = f"{n(r['mean'])} $\\pm$ {n(r['sd'])}" + ("" if r["n"] == 20 else f" ({r['n']})")
                c += [sd, pub]
            rows.append(f"{ds} & {H} & " + " & ".join(c) + r" \\")
        if ds != D.DATASETS[-1]:
            rows.append(r"\addlinespace[3pt]")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("b1", "\n".join(rows) + "\n")


def tab_b4_counts():
    T = D.nv7()["D_B4"]["counts"]
    rows = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
            r"Nhãn & lr gốc, 20 seed & lr gốc, 15 seed mới & lr tinh chỉnh, 15 seed mới \\", r"\midrule"]
    for k, v in T.items():
        rows.append(f"{LABEL[k]} & {v[2]} & {v[1]} & {v[0]} \\\\")
    rows += [r"\midrule", f"Tổng & {sum(v[2] for v in T.values())} & {sum(v[1] for v in T.values())} & "
             f"{sum(v[0] for v in T.values())} \\\\", r"\bottomrule", r"\end{tabular}"]
    write("b4_counts", "\n".join(rows) + "\n")


def tab_b4_full():
    old, tuned = D.b4_rows(), D.b4t_rows()
    lr = D.lr_cells()
    rows = [r"{\footnotesize", r"\begin{longtable}{@{}lrl l l l l@{}}", r"\toprule",
            r"Dataset & $H$ & Mô hình & $\Delta$ test, lr gốc [CI 95\%] & nhãn (lr gốc) & lr & $\Delta$ test, tinh chỉnh [CI 95\%]; nhãn \\",
            r"\midrule", r"\endhead"]
    for ds in D.DATASETS:
        for H in D.HORIZONS:
            for m in D.MODELS:
                o, t = old[(ds, H, m)], tuned[(ds, H, m)]
                mult = f"$\\times${float(lr[(ds, H, m)]['chosen_mult']):g}".replace(".", ",")
                rows.append(f"{ds} & {H} & {m} & {ci(o['d_test'], o['d_test_ci'])} & {LABEL[o['label']]} & {mult} & "
                            f"{ci(t['d_test_tuned'], t['d_test_ci_tuned'])}; {LABEL[t['label_tuned']]} \\\\")
        rows.append(r"\addlinespace[2pt]")
    rows += [r"\bottomrule", r"\end{longtable}", "}"]
    write("b4_full", "\n".join(rows) + "\n")


def tab_lr():
    lr = D.lr_cells()
    mults = ["0.5", "1", "2", "4", "8"]
    rows = [r"{\footnotesize", r"\begin{longtable}{@{}lrlr rrrrr r@{}}", r"\toprule",
            r" & & & & \multicolumn{5}{c}{MSE val trung bình (seed 2021--2025), lr $= \times$ lr gốc} & \\",
            r"\cmidrule(lr){5-9}",
            r"Dataset & $H$ & Mô hình & lr gốc & $\times 0{,}5$ & $\times 1$ & $\times 2$ & $\times 4$ & $\times 8$ & chọn \\",
            r"\midrule", r"\endhead"]
    for ds in ("ETTh2", "ETTh1", "ETTm1"):
        for m in D.MODELS:
            for H in D.HORIZONS:
                c = lr[(ds, H, m)]
                mv = {k: float(v) for k, v in c["mean_val_by_mult"].items()}
                ch = f"{float(c['chosen_mult']):g}"
                vals = [("\\textbf{%s}" % n(mv[k], 5) if k == ch else n(mv[k], 5)) if k in mv else "--" for k in mults]
                rows.append(f"{ds} & {H} & {m} & {n(float(c['base_lr']), 4).rstrip('0').rstrip(',')} & " + " & ".join(vals)
                            + f" & $\\times${ch.replace('.', ',')} \\\\")
            rows.append(r"\addlinespace[2pt]")
    rows += [r"\bottomrule", r"\end{longtable}", "}"]
    write("lr", "\n".join(rows) + "\n")


def tab_c3():
    old, tuned = D.c3_rows(), D.c3t_rows()
    lr = D.lr_cells()
    rows = [r"\begin{tabular}{@{}lr l l l r@{}}", r"\toprule",
            r"Dataset & $H$ & lr gốc, 20 seed & lr tinh chỉnh, 15 seed & lr (Linear / DLinear) & dạng đóng \\",
            r"\midrule"]
    for ds in D.DATASETS:
        for H in D.HORIZONS:
            o, t = old[(ds, H)], tuned[(ds, H)]
            a = ci(o["d_test"], o["d_test_ci"])
            if o["dlinear_better_test"]:
                a = f"\\textbf{{{a}}}"
            b = ci(t["d_test"], t["d_test_ci"])
            if t["dlinear_better_test"]:
                b = f"\\textbf{{{b}}}"
            mults = " / ".join(f"$\\times${float(lr[(ds, H, m)]['chosen_mult']):g}".replace(".", ",")
                               for m in ("Linear", "DLinear"))
            rows.append(f"{ds} & {H} & {a} & {b} & {mults} & {n(o['cf_d_test'], 4, True)} \\\\")
        if ds != D.DATASETS[-1]:
            rows.append(r"\addlinespace[3pt]")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("c3", "\n".join(rows) + "\n")


def tab_outliers():
    rows = [r"{\footnotesize", r"\begin{longtable}{@{}lr l rrr rrr@{}}", r"\toprule",
            r" & & & \multicolumn{3}{c}{MSE test ở $\lambda^*_{\mathrm{MSE}}$} & \multicolumn{3}{c}{MAE test ở $\lambda^*_{\mathrm{MAE}}$} \\",
            r"\cmidrule(lr){4-6}\cmidrule(l){7-9}",
            r"Mô hình & $H$ & chèn, tỷ lệ & MSE & MAE & Huber & MSE & MAE & Huber \\", r"\midrule", r"\endhead"]
    for m in ("Linear", "NLinear"):
        for H in D.HORIZONS:
            cases = [("--", 0)] + [("series", r) for r in (0.01, 0.05, 0.1)]
            if H in (96, 720):
                cases += [("target", r) for r in (0.01, 0.1)]
            for mode, rate in cases:
                v = [D.outlier(mode if rate else "series", rate, H, m, ob, met) for met in D.METRICS
                     for ob in D.OBJECTIVES]
                lab = "gốc" if rate == 0 else f"{'chuỗi' if mode == 'series' else 'đích'}, {rate * 100:g}\\%"
                rows.append(f"{m} & {H} & {lab} & " + " & ".join(bold_min(v[:3]) + bold_min(v[3:])) + r" \\")
            rows.append(r"\addlinespace[2pt]")
    rows += [r"\bottomrule", r"\end{longtable}", "}"]
    write("outliers", "\n".join(rows) + "\n")


def tab_magnitude():
    rows = [r"\begin{tabular}{@{}r l r r rrr rrr@{}}", r"\toprule",
            r" & & & & \multicolumn{3}{c}{MSE test ở $\lambda^*_{\mathrm{MSE}}$} & \multicolumn{3}{c}{MAE test ở $\lambda^*_{\mathrm{MAE}}$} \\",
            r"\cmidrule(lr){5-7}\cmidrule(l){8-10}",
            r"$H$ & chèn & tỷ lệ & độ lớn & MSE & MAE & Huber & MSE & MAE & Huber \\", r"\midrule"]
    for H in (96, 720):
        v = [D.outlier("series", 0, H, "Linear", ob, met) for met in D.METRICS for ob in D.OBJECTIVES]
        rows.append(f"{H} & gốc & -- & -- & " + " & ".join(bold_min(v[:3]) + bold_min(v[3:])) + r" \\")
        for mode in ("series", "target"):
            for rate in (0.01, 0.1):
                for mag in (3.0, 5.0, 10.0):
                    v = [D.outlier(mode, rate, H, "Linear", ob, met, mag=mag) for met in D.METRICS for ob in D.OBJECTIVES]
                    rows.append(f"{H} & {'chuỗi' if mode == 'series' else 'đích'} & {rate * 100:g}\\% & "
                                f"{mag:g}$\\sigma$ & " + " & ".join(bold_min(v[:3]) + bold_min(v[3:])) + r" \\")
        if H == 96:
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("magnitude", "\n".join(rows) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab_b2()
    tab_b3_main()
    tab_b3_full()
    tab_b1()
    tab_b4_counts()
    tab_b4_full()
    tab_lr()
    tab_c3()
    tab_outliers()
    tab_magnitude()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
