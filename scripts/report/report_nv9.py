"""Bảng và tóm tắt của nhiệm vụ 9 (docs/tasks/NHIEM_VU_9.md) → results/nv9/summary.json, results/nv9/tables_*.md.

    python scripts/report/report_nv9.py                 # mọi phần có dữ liệu
    python scripts/report/report_nv9.py --part C1 C2    # chỉ đọc dữ liệu cũ, không cần các lần chạy mới

Phần:
    A   lưới lr mở rộng (results/sgd_lr_ext/selection.json gộp với results/sgd_lr/selection.json): bảng lr, B4 và C.3
        tinh chỉnh tính lại bằng report_nv7.part_d (cùng code, lựa chọn đã gộp), danh sách thay đổi so với nv7.
    B   code gốc với drop_last của test loader (results/official_ltsf_droplast/): số công bố, 5 seed True, 5 seed False.
    C1  họ ridge trên ETTh2 (results/lambda_path.csv, 142 λ): λ = 0, λ*_MSE, MSE test nhỏ nhất trên lưới (mức tham chiếu
        chọn theo test, KHÔNG hợp lệ để chọn mô hình), so với SGD lr gốc (20 seed) và tinh chỉnh (15 seed).
    C2  đoạn cảm biến bị kẹt trên ETTh2: các đoạn hằng ≥ 24 giờ theo kênh, bảng gốc / đã lọc (kết quả đã có).
    C3  Huber δ theo bước h (results/huber_delta_h/) so với δ_MAD chung (results/huber_mad/), MSE, MAE.
Không ghi gì vào các thư mục kết quả khác.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / d) for d in ("grid", "sgd", "checks", "report")]
RES = ROOT / "results"
OUT = RES / "nv9"
HORIZONS = [96, 192, 336, 720]
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
L = 336


def git_commit():
    from bench_5060ti import git_commit as g
    return g()


def num(x, p=4):
    return "—" if x is None else f"{x:.{p}f}".replace(".", ",")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# lựa chọn lr đã gộp (dùng ở A và C1)
# ---------------------------------------------------------------------------

def merged_selection():
    """Ô của results/sgd_lr/selection.json, với ô biên thay bằng kết quả của lưới mở rộng (nếu có)."""
    old = jload(RES / "sgd_lr" / "selection.json")["cells"]
    f = RES / "sgd_lr_ext" / "selection.json"
    ext = jload(f)["cells"] if f.exists() else {}
    sel = {}
    for k, c in old.items():
        c = dict(c, mean_val_by_mult=dict(c["mean_val_by_mult"]))
        if k in ext:
            e = ext[k]
            c["sel_files"] = dict(c["sel_files"])
            for m, lv in e["new_levels"].items():
                if lv["mean_val"] is not None:
                    c["mean_val_by_mult"][m] = lv["mean_val"]
                    c["sel_files"][m] = e["sel_files"][m]
            c.update(chosen_mult=e["chosen_mult"], chosen_lr=e["chosen_lr"], final_files=e["final_files"],
                     final_mean_val=e["final_mean_val"], ext=e)
        sel[k] = c
    return sel, ext


def tuned_test(sel, ds, m, H):
    return [jload(ROOT / Path(p.replace("\\", "/")))["test_mse"] for p in sel[f"{ds} {m} H={H}"]["final_files"]]


# ---------------------------------------------------------------------------
# A
# ---------------------------------------------------------------------------

def part_a(summ):
    import report_nv7
    from sgd_lr_extend import find
    sel, ext = merged_selection()
    assert ext, "chưa có results/sgd_lr_ext/selection.json"
    cols = ("0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32")
    tmp = {}
    report_nv7.part_d(tmp, sel=sel, find=find, out=OUT / "tables_A.md", mult_cols=cols)
    summ["A_B4"], summ["A_C3"] = tmp["D_B4"], tmp["D_C3"]
    old = jload(RES / "nv7" / "summary.json")
    ob4 = {(r["dataset"], r["H"], r["model"]): r for r in old["D_B4"]["rows"]}
    oc3 = {(r["dataset"], r["H"]): r for r in old["D_C3"]["rows"]}
    changes = []
    for k, e in ext.items():
        key = (e["dataset"], e["H"], e["model"])
        n = next(r for r in tmp["D_B4"]["rows"] if (r["dataset"], r["H"], r["model"]) == key)
        o = ob4[key]
        changes.append({"cell": k, "edge": e["edge"], "old_mult": e["old_chosen_mult"], "new_mult": e["chosen_mult"],
                        "changed": e["changed"], "still_at_edge": e["still_at_edge"],
                        "new_levels": {m: lv["mean_val"] for m, lv in e["new_levels"].items()},
                        "failed": {m: lv["failed"] for m, lv in e["new_levels"].items() if lv["failed"]},
                        "old_mean_val_edge": e["old_mean_val_by_mult"][f"{e['old_chosen_mult']:g}"],
                        "d_test_old": o["d_test_tuned"], "d_test_new": n["d_test_tuned"],
                        "d_test_ci_old": o["d_test_ci_tuned"], "d_test_ci_new": n["d_test_ci_tuned"],
                        "label_old": o["label_tuned"], "label_new": n["label_tuned"]})
    hm = [r for r in tmp["D_B4"]["rows"] if r["dataset"] in ("ETTh1", "ETTm1")]
    hm_old = [r for r in old["D_B4"]["rows"] if r["dataset"] in ("ETTh1", "ETTm1")]
    sig = lambda rs: sum(r["d_test_tuned"] - r["d_test_ci_tuned"] > 0 for r in rs)     # noqa: E731
    stop = {"hm_worse_sig_old": sig(hm_old), "hm_worse_sig_new": sig(hm), "hm_cells": len(hm),
            "sgd_better_outside_etth2": [f"{r['dataset']} H={r['H']} {r['model']}" for r in tmp["D_B4"]["rows"]
                                         if r["label_tuned"] == "SGD tốt hơn thật" and r["dataset"] != "ETTh2"]}
    stop["rq3_changed"] = stop["hm_worse_sig_new"] < 20 or bool(stop["sgd_better_outside_etth2"])
    c3 = {"old_n_dlinear_better": old["D_C3"]["n_dlinear_better"], "new_n_dlinear_better": tmp["D_C3"]["n_dlinear_better"],
          "changed_rows": [{"dataset": r["dataset"], "H": r["H"], "old": [oc3[(r["dataset"], r["H"])][x] for x in ("d_test", "d_test_ci")],
                            "new": [r["d_test"], r["d_test_ci"]]}
                           for r in tmp["D_C3"]["rows"]
                           if abs(r["d_test"] - oc3[(r["dataset"], r["H"])]["d_test"]) > 0]}
    counts_old = {k: v[0] for k, v in old["D_B4"]["counts"].items()}
    counts_new = {k: v[0] for k, v in tmp["D_B4"]["counts"].items()}
    summ["A"] = {"changes": changes, "stop_check": stop, "c3": c3, "counts_old": counts_old, "counts_new": counts_new,
                 "worse_sig_old": old["D_B4"]["worse_test_sig"], "worse_sig_new": tmp["D_B4"]["worse_test_sig"],
                 "worse_mean_old": old["D_B4"]["worse_test_mean"], "worse_mean_new": tmp["D_B4"]["worse_test_mean"],
                 "n_changed": sum(c["changed"] for c in changes), "n_still_edge": sum(c["still_at_edge"] for c in changes),
                 # cộng mọi lần gọi sgd_lr_extend.py (dòng "→ … (x giờ)" của driver.log), không chỉ lần cuối
                 "wall_hours": sum(float(x) for x in re.findall(r"\(([\d.]+) giờ\)", (RES / "sgd_lr_ext" / "driver.log")
                                                                 .read_text(encoding="utf-8")))}
    g = lambda x: f"{x:g}".replace(".", ",")                                            # noqa: E731
    md = ["# Phần A: mở rộng lưới lr ở 15 ô biên", "",
          "| Ô | biên | mức mới: val MSE TB (5 seed) | lr cũ → mới | Δ test cũ → mới [CI] | nhãn cũ → mới | còn ở biên |",
          "|---|---|---|---|---|---|---|"]
    for c in changes:
        lv = "; ".join(f"×{g(float(m))}: {num(v, 5) if v is not None else 'thất bại'}" for m, v in c["new_levels"].items())
        md.append(f"| {c['cell']} | {c['edge']} (×{g(c['old_mult'])}: {num(c['old_mean_val_edge'], 5)}) | {lv} | "
                  f"×{g(c['old_mult'])} → ×{g(c['new_mult'])} | {num(c['d_test_old'])} [±{num(c['d_test_ci_old'])}] → "
                  f"{num(c['d_test_new'])} [±{num(c['d_test_ci_new'])}] | {c['label_old']} → {c['label_new']} | "
                  f"{'có' if c['still_at_edge'] else 'không'} |")
    md += ["", f"SGD kém dạng đóng có ý nghĩa trên test: {summ['A']['worse_sig_old']}/36 → {summ['A']['worse_sig_new']}/36; "
           f"ETTh1 + ETTm1: {stop['hm_worse_sig_old']}/24 → {stop['hm_worse_sig_new']}/24. \"SGD tốt hơn thật\" ngoài ETTh2: "
           f"{stop['sgd_better_outside_etth2'] or 'không có'}. Điểm dừng RQ3: {'CÓ' if stop['rq3_changed'] else 'không'}.",
           f"C.3 (DLinear − Linear, cả hai tinh chỉnh): DLinear tốt hơn có ý nghĩa {c3['old_n_dlinear_better']}/12 → "
           f"{c3['new_n_dlinear_better']}/12.", "", "| nhãn | nv7 | nv9 |", "|---|---|---|"]
    md += [f"| {k} | {counts_old[k]} | {counts_new[k]} |" for k in counts_new]
    (OUT / "tables_A_changes.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# B
# ---------------------------------------------------------------------------

B_CELLS = [("ETTh1", 192, "DLinear"), ("ETTh1", 720, "DLinear"), ("ETTm1", 96, "DLinear"), ("ETTm1", 192, "DLinear"),
           ("ETTm1", 336, "DLinear"), ("ETTh2", 336, "NLinear"), ("ETTh2", 720, "NLinear"), ("ETTm1", 336, "NLinear"),
           ("ETTh2", 720, "DLinear")]
B_SEEDS = list(range(2021, 2026))


def part_b(summ):
    b1 = {(r["dataset"], r["H"], r["model"]): r for r in jload(RES / "sgd_b1_b4" / "summary.json")["B1"]["rows"]}
    rows, md = [], ["# Phần B: code gốc LTSF-Linear với drop_last của test loader", "",
                    "Seed 2021–2025, siêu tham số của script gốc. MSE test do code gốc in ra. \"bỏ\": số cửa sổ test bị bỏ khi "
                    "drop_last = True (N mod batch).", "",
                    "| Ô | công bố | drop_last = True: TB ± sd [min; max] | drop_last = False: TB ± sd [min; max] | "
                    "True − False (cùng seed) | bỏ / N | công bố trong [min; max] (True / False) | repo SGD 20 seed: min |",
                    "|---|---|---|---|---|---|---|---|"]
    for ds, H, m in B_CELLS:
        d = RES / "official_ltsf_droplast" / ds
        r1 = [jload(d / f"{m}_H{H}_seed{s}_dl1.json") for s in B_SEEDS]
        r0 = [jload(d / f"{m}_H{H}_seed{s}_dl0.json") for s in B_SEEDS]
        t1 = np.array([r["official_test_mse"] for r in r1])
        t0 = np.array([r["official_test_mse"] for r in r0])
        same_train = all(abs(a["repo_test_mse"] - b["repo_test_mse"]) < 1e-12 for a, b in zip(r1, r0))
        pub = b1[(ds, H, m)]["pub"]
        row = {"dataset": ds, "H": H, "model": m, "pub": pub, "control": (ds, H, m) == ("ETTh2", 720, "DLinear"),
               "dl1": t1.tolist(), "dl0": t0.tolist(),
               "dl1_mean": float(t1.mean()), "dl1_sd": float(t1.std(ddof=1)), "dl1_min": float(t1.min()), "dl1_max": float(t1.max()),
               "dl0_mean": float(t0.mean()), "dl0_sd": float(t0.std(ddof=1)), "dl0_min": float(t0.min()), "dl0_max": float(t0.max()),
               "diff": (t1 - t0).tolist(), "diff_mean": float((t1 - t0).mean()),
               "diff_min": float((t1 - t0).min()), "diff_max": float((t1 - t0).max()),
               "n_test": r1[0]["n_test_windows"], "batch": r1[0]["batch_size"], "n_dropped": r1[0]["n_test_windows_dropped"],
               "best_epoch_dl1": [r["best_epoch"] for r in r1], "best_epoch_dl0": [r["best_epoch"] for r in r0],
               "same_training": same_train, "device": r1[0]["device"],
               "pub_in_dl1": bool(t1.min() <= pub <= t1.max()), "pub_in_dl0": bool(t0.min() <= pub <= t0.max()),
               "pub_below_dl1_min": bool(pub < t1.min()), "repo_sgd_min": b1[(ds, H, m)]["min"],
               "repo_sgd_mean": b1[(ds, H, m)]["mean"]}
        rows.append(row)
        f = lambda p: f"{num(row[p + '_mean'])} ± {num(row[p + '_sd'])} [{num(row[p + '_min'])}; {num(row[p + '_max'])}]"  # noqa: E731
        md.append(f"| {ds} H={H} {m}{' (đối chứng)' if row['control'] else ''} | {num(pub, 3)} | {f('dl1')} | {f('dl0')} | "
                  f"{num(row['diff_mean'], 5)} [{num(row['diff_min'], 5)}; {num(row['diff_max'], 5)}] | "
                  f"{row['n_dropped']} / {row['n_test']} | {'có' if row['pub_in_dl1'] else 'không'} / "
                  f"{'có' if row['pub_in_dl0'] else 'không'} | {num(row['repo_sgd_min'])} |")
    main8 = [r for r in rows if not r["control"]]
    n_in = sum(r["pub_in_dl1"] for r in main8)
    summ["B"] = {"rows": rows, "n_pub_in_dl1": n_in, "n_pub_in_dl0": sum(r["pub_in_dl0"] for r in main8),
                 "n_cells": len(main8), "all_same_training": all(r["same_training"] for r in rows),
                 "explains": n_in > len(main8) / 2, "stop_5of8": n_in >= 5,
                 "max_abs_diff": max(max(abs(r["diff_min"]), abs(r["diff_max"])) for r in rows)}
    md += ["", f"Số công bố trong [min; max] của 5 seed với drop_last = True: {n_in}/{len(main8)} ô (False: "
           f"{summ['B']['n_pub_in_dl0']}/{len(main8)}). Huấn luyện trùng nhau giữa True và False (chấm lại trên đủ cửa sổ): "
           f"{'mọi ô' if summ['B']['all_same_training'] else 'KHÔNG'}. |True − False| lớn nhất: {summ['B']['max_abs_diff']:.2e}.",
           f"Điểm dừng (≥ 5/8): {'CÓ' if summ['B']['stop_5of8'] else 'không'}."]
    (OUT / "tables_B.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------

def part_c1(summ):
    sel, _ = merged_selection()
    path = [r for r in read_csv(RES / "lambda_path.csv") if r["dataset"] == "ETTh2" and r["objective"] == "MSE"]
    rows, md = [], ["# C.1: họ ridge trên ETTh2 (mục tiêu MSE, 142 λ)", "",
                    "\"ridge tốt nhất theo test\": MSE test nhỏ nhất trên toàn lưới λ — mức tham chiếu chọn theo test, "
                    "KHÔNG hợp lệ để chọn mô hình; chỉ dùng để chỉ ra giới hạn của họ ridge.", "",
                    "| Ô | λ = 0: val / test | λ*_MSE: λ, val / test | ridge tốt nhất theo test: λ, test | SGD lr gốc (20 seed): test TB "
                    "| SGD tinh chỉnh (15 seed): test TB | ridge tốt nhất − SGD gốc / − SGD tinh chỉnh |", "|---|---|---|---|---|---|---|"]
    for m in ("Linear", "DLinear"):
        for H in HORIZONS:
            p = sorted((r for r in path if r["model"] == m and int(r["H"]) == H), key=lambda r: float(r["lam"]))
            assert len(p) == 142, (m, H, len(p))
            z = p[0]
            assert float(z["lam"]) == 0
            st = min(p, key=lambda r: float(r["val_mse"]))
            bt = min(p, key=lambda r: float(r["test_mse"]))
            sgd0 = [jload(RES / "sgd_etth2_20seed" / f"{m}_H{H}_seed{s}.json")["test_mse"] for s in range(2021, 2041)]
            sgdt = tuned_test(sel, "ETTh2", m, H)
            row = {"model": m, "H": H, "lam0_val": float(z["val_mse"]), "lam0_test": float(z["test_mse"]),
                   "lam_star": float(st["lam"]), "star_val": float(st["val_mse"]), "star_test": float(st["test_mse"]),
                   "lam_best_test": float(bt["lam"]), "best_test": float(bt["test_mse"]), "best_test_val": float(bt["val_mse"]),
                   "best_at_edge": bt is p[-1] or bt is p[0],
                   "sgd_orig_mean": float(np.mean(sgd0)), "sgd_orig_n": len(sgd0),
                   "sgd_tuned_mean": float(np.mean(sgdt)), "sgd_tuned_n": len(sgdt),
                   "tuned_mult": sel[f"ETTh2 {m} H={H}"]["chosen_mult"]}
            row["best_minus_sgd_orig"] = row["best_test"] - row["sgd_orig_mean"]
            row["best_minus_sgd_tuned"] = row["best_test"] - row["sgd_tuned_mean"]
            rows.append(row)
            md.append(f"| {m} H={H} | {num(row['lam0_val'])} / {num(row['lam0_test'])} | {row['lam_star']:g}, "
                      f"{num(row['star_val'])} / {num(row['star_test'])} | {row['lam_best_test']:g}, {num(row['best_test'])} | "
                      f"{num(row['sgd_orig_mean'])} | {num(row['sgd_tuned_mean'])} | {num(row['best_minus_sgd_orig'])} / "
                      f"{num(row['best_minus_sgd_tuned'])} |")
    n_o = sum(r["best_minus_sgd_orig"] > 0 for r in rows)
    n_t = sum(r["best_minus_sgd_tuned"] > 0 for r in rows)
    summ["C1"] = {"rows": rows, "n_ridge_best_worse_than_sgd_orig": n_o, "n_ridge_best_worse_than_sgd_tuned": n_t,
                  "n_cells": len(rows), "n_best_at_edge": sum(r["best_at_edge"] for r in rows)}
    md += ["", f"Ridge tốt nhất theo test vẫn kém SGD trung bình: {n_o}/{len(rows)} ô (lr gốc), {n_t}/{len(rows)} ô (tinh chỉnh). "
           f"λ tốt nhất theo test ở đầu mút lưới: {summ['C1']['n_best_at_edge']} ô."]
    (OUT / "tables_C1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------

MIN_RUN = 24


def constant_runs(min_len=MIN_RUN):
    """Các đoạn hằng (giá trị gốc không đổi giữa hai giờ liên tiếp) dài ≥ min_len giờ, theo kênh của ETTh2."""
    from src.data import split_borders
    df = pd.read_csv(ROOT / "data" / "ETTh2.csv")
    tr_end, va_end, te_end = split_borders("ETTh2")
    runs = []
    for c in CHANNELS:
        v = df[c].to_numpy()[:te_end]
        start = 0
        for i in range(1, len(v) + 1):
            if i == len(v) or v[i] != v[start]:
                n = i - start
                if n >= min_len:
                    s, e = start, i - 1
                    parts = [name for name, a, b in (("train", 0, tr_end), ("val", tr_end, va_end), ("test", va_end, te_end))
                             if s < b and e >= a]
                    runs.append({"channel": c, "start_idx": s, "end_idx": e, "start": df["date"][s], "end": df["date"][e],
                                 "hours": n, "value": float(v[s]), "split": "/".join(parts)})
                start = i
    return runs, (tr_end, va_end, te_end)


def part_c2(summ):
    sys.path.insert(0, str(ROOT))
    runs, borders = constant_runs()
    cw = jload(RES / "constant_windows" / "summary.json")
    e2 = jload(RES / "etth2_report_2" / "summary.json")
    cwc = {(c["H"], c["model"]): c for c in cw["cells"]}
    sgd = {(r["model"], r["H"]): r for r in e2["B"] if r["metric"] == "test"}
    mae = {(r["model"], r["H"]): r for r in e2["C"]}
    nl_sgd = {H: float(np.mean([jload(RES / "sgd_etth2_20seed" / f"NLinear_H{H}_seed{s}.json")["test_mse"]
                                for s in range(2021, 2031)])) for H in HORIZONS}
    table = []
    for m in ("Linear", "DLinear", "NLinear"):
        for H in HORIZONS:
            c = cwc[(H, m)]
            s = sgd.get((m, H))
            mm = mae.get(("Linear" if m == "DLinear" else m, H))   # DLinear ở λ = 0 ≡ Linear (mọi mục tiêu)
            table.append({"model": m, "H": H, "cf_full": c["before"]["test_mse"], "cf_filt": c["after"]["test_mse"],
                          "sgd_full": s["sgd_full"] if s else nl_sgd[H], "sgd_filt": s["sgd_filt"] if s else None,
                          "sgd_n": s["n"] if s else 10,
                          "mae0_full": mm["full_mae_obj"], "mae0_filt": mm["filt_mae_obj"],
                          "mse0_full": mm["full_mse_obj"], "mse0_filt": mm["filt_mse_obj"],
                          "gain_full": mm["gain_full"], "gain_filt": mm["gain_filt"], "dlinear_from_linear": m == "DLinear"})
    counts = [c for c in cw["counts"]]
    summ["C2"] = {"criterion": cw["definition"] if "definition" in cw else None, "eps": cw["eps"], "min_run_hours": MIN_RUN,
                  "runs": runs, "borders": list(borders), "table": table, "window_counts": counts,
                  "n_runs": len(runs), "channels_with_runs": sorted({r["channel"] for r in runs}, key=CHANNELS.index)}
    md = ["# C.2: đoạn hằng của ETTh2", "", f"Tiêu chí cửa sổ hằng (constant_windows.py): {cw.get('definition')}", "",
          f"## Đoạn chuỗi hằng ≥ {MIN_RUN} giờ (giá trị gốc lặp lại liên tiếp; ranh giới train/val/test ở hàng "
          f"{borders[0]}, {borders[1]}, {borders[2]})", "", "| Kênh | bắt đầu | kết thúc | giờ | giá trị | thuộc |", "|---|---|---|---|---|---|"]
    md += [f"| {r['channel']} | {r['start']} | {r['end']} | {r['hours']} | {r['value']:g} | {r['split']} |" for r in runs]
    md += ["", "## Gốc / đã lọc (MSE test; lọc chỉ ở train)", "",
           "| Mô hình | H | dạng đóng λ* | SGD TB (n seed) | MAE λ = 0 | lợi thế MAE so với MSE (λ = 0) |", "|---|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['model']} | {t['H']} | {num(t['cf_full'])} / {num(t['cf_filt'])} | {num(t['sgd_full'])} / "
                  f"{num(t['sgd_filt'])} ({t['sgd_n']}) | {num(t['mae0_full'])} / {num(t['mae0_filt'])} | "
                  f"{num(t['gain_full'])} / {num(t['gain_filt'])} |")
    (OUT / "tables_C2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------

def part_c3(summ):
    new = read_csv(RES / "huber_delta_h" / "lambda_path.csv")
    mad = read_csv(RES / "huber_mad" / "lambda_path.csv")
    grid = read_csv(RES / "lambda_path.csv")
    dl = read_csv(RES / "huber_delta_h" / "deltas.csv")
    rows, md = [], ["# C.3: Huber với δ theo bước h", "",
                    "MSE test ở λ*_MSE (argmin val MSE trên đường λ của chính mục tiêu), MAE test ở λ*_MAE.", "",
                    "| Ô | δ_MAD | δ_h: min–max (TB) | MSE | MAE | Huber δ_MAD | Huber δ_h | vị trí Huber δ_MAD / δ_h (MSE test; MAE test) |",
                    "|---|---|---|---|---|---|---|---|"]
    cells = sorted({(r["dataset"], int(r["H"]), r["model"]) for r in new})

    def at(path, met):
        best = min(path, key=lambda r: float(r[f"val_{met}"]))
        return float(best[f"test_{met}"]), float(best["lam"])

    def rank(v, key):
        order = sorted(v, key=lambda k: v[k])
        return order.index(key) + 1
    for ds, H, m in cells:
        k = lambda r: r["dataset"] == ds and int(r["H"]) == H and r["model"] == m      # noqa: E731
        paths = {"MSE": [r for r in grid if k(r) and r["objective"] == "MSE"],
                 "MAE": [r for r in grid if k(r) and r["objective"] == "MAE"],
                 "Huber δ_MAD": [r for r in mad if k(r) and r["objective"] == "Huber"],
                 "Huber δ_h": [r for r in new if k(r)]}
        res = {name: {met: at(p, met) for met in ("mse", "mae")} for name, p in paths.items()}
        dh = np.array([float(r["delta_h"]) for r in dl if r["cell"] == f"{ds}:{H}:{m}"])
        row = {"dataset": ds, "H": H, "model": m, "delta_mad": float(paths["Huber δ_MAD"][0]["delta"]),
               "delta_h_min": float(dh.min()), "delta_h_max": float(dh.max()), "delta_h_mean": float(dh.mean()),
               "test": {n: {met: v[met][0] for met in v} for n, v in res.items()},
               "lam": {n: {met: v[met][1] for met in v} for n, v in res.items()},
               "n_lams_delta_h": len(paths["Huber δ_h"])}
        for met in ("mse", "mae"):
            for hub in ("Huber δ_MAD", "Huber δ_h"):
                v = {"MSE": row["test"]["MSE"][met], "MAE": row["test"]["MAE"][met], "Huber": row["test"][hub][met]}
                row[f"rank_{met}_{hub.split()[1]}"] = rank(v, "Huber")
        rows.append(row)
        t = row["test"]
        md.append(f"| {ds} H={H} {m} | {num(row['delta_mad'])} | {num(row['delta_h_min'])}–{num(row['delta_h_max'])} "
                  f"({num(row['delta_h_mean'])}) | {num(t['MSE']['mse'])} / {num(t['MSE']['mae'])} | "
                  f"{num(t['MAE']['mse'])} / {num(t['MAE']['mae'])} | {num(t['Huber δ_MAD']['mse'])} / {num(t['Huber δ_MAD']['mae'])} | "
                  f"{num(t['Huber δ_h']['mse'])} / {num(t['Huber δ_h']['mae'])} | "
                  f"{row['rank_mse_δ_MAD']} / {row['rank_mse_δ_h']}; {row['rank_mae_δ_MAD']} / {row['rank_mae_δ_h']} |")
    changed = [r for r in rows if r["rank_mse_δ_MAD"] != r["rank_mse_δ_h"] or r["rank_mae_δ_MAD"] != r["rank_mae_δ_h"]]
    summ["C3"] = {"rows": rows, "n_rank_changed": len(changed), "meta": jload(RES / "huber_delta_h" / "summary.json")}
    md += ["", "Ô có dạng \"MSE / MAE test\"; vị trí: hạng của Huber giữa ba mục tiêu (1 = thấp nhất). "
           f"Vị trí đổi khi dùng δ_h: {len(changed)}/{len(rows)} ô."]
    (OUT / "tables_C3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["all"], choices=["A", "B", "C1", "C2", "C3", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "summary.json"
    summ = jload(f) if f.exists() else {}
    parts = ["A", "B", "C1", "C2", "C3"] if "all" in args.part else args.part
    for p in parts:
        globals()[f"part_{p.lower()}"](summ)
    summ["git_commit"] = git_commit()
    f.write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
