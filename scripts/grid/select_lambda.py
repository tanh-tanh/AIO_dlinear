"""Chọn λ* từ results/lambda_path.csv, sinh results/results.csv.

Mỗi (dataset, H, mô hình, mục tiêu) cho hai mức λ: λ = 0 và λ*. λ* có ba cách chọn (cột selection):
    val_objective  λ* tối thiểu metric val của chính mục tiêu (MSE → val_mse, MAE → val_mae, Huber → val_huber)
    val_mse        λ* tối thiểu val_mse với mọi mục tiêu; dùng để báo MSE test (docs/tasks/NHIEM_VU_7.md, B.1)
    val_mae        λ* tối thiểu val_mae với mọi mục tiêu; dùng để báo MAE test (NHIEM_VU_7, B.1)
Mặc định chỉ sinh val_objective và val_mse như trước (results.csv 432 dòng khi grid chạy xong); chọn bằng
--selections. λ* có thể bằng 0. Test không dùng để chọn gì. Các cột val_* của lambda_path.csv không chứa số hạng
phạt λ‖w‖² (kiểm bằng tests/test_val_metrics.py).

    python scripts/grid/select_lambda.py
    python scripts/grid/select_lambda.py --in results/outliers/lambda_path.csv --out results/outliers/results.csv
    python scripts/grid/select_lambda.py --selections val_mse val_mae val_objective --out results/results_v2.csv
    python scripts/grid/select_lambda.py --in results/lambda_path.csv results/huber_mad/lambda_path.csv ...   # NV7 A

Nhiều file --in: một đường λ (cùng khóa nhóm) có ở nhiều file thì lấy trọn đường của file sau cùng, không trộn
dòng giữa các file. Dùng để thay đường Huber δ = 1 của grid bằng đường Huber δ_MAD (results/huber_mad/).

File có các cột outlier (docs/tasks/NHIEM_VU_6.md) thì mỗi (outlier_mode, outlier_rate, outlier_seed) là một nhóm
riêng, và các cột đó được giữ trong file ra; có thêm cột outlier_magnitude (NV7 C) thì nó cũng vào khóa nhóm,
dòng thiếu cột coi là 10. Không có thì chạy y như trước.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH_CSV = ROOT / "results" / "lambda_path.csv"
OUT_CSV = ROOT / "results" / "results.csv"
OWN = {"MSE": "val_mse", "MAE": "val_mae", "Huber": "val_huber"}
SELECTIONS = ["val_objective", "val_mse", "val_mae"]
KEEP = ["dataset", "H", "model", "objective", "lam", "lam_convention", "delta", "k", "train_obj",
        "val_mse", "val_mae", "val_huber", "test_mse", "test_mae", "n_iter", "converged", "init",
        "derived_from", "tol", "patience", "gpu", "git_commit"]
OUTLIER_KEYS = ["outlier_mode", "outlier_rate", "outlier_seed"]
OUTLIER_EXTRA = ["n_contaminated", "frac_windows_x", "frac_windows_y"]
DEFAULT_MAGNITUDE = 10.0


def read_paths(files):
    """Đọc các file lambda_path; trả (nhóm → dòng, có cột outlier?, có cột outlier_magnitude?)."""
    groups, has_outlier, has_mag = {}, False, False
    parsed = []
    for f in files:
        with Path(f).open(encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            has_outlier |= "outlier_mode" in rd.fieldnames
            has_mag |= "outlier_magnitude" in rd.fieldnames
            parsed.append(list(rd))
    for rows in parsed:
        mine = {}
        for r in rows:
            okey = ()
            if has_outlier:
                okey = (r.get("outlier_mode", ""), float(r.get("outlier_rate") or 0), int(r.get("outlier_seed") or 0))
                if has_mag:
                    okey += (float(r.get("outlier_magnitude") or DEFAULT_MAGNITUDE),)
            mine.setdefault((r["dataset"], int(r["H"]), r["model"], r["objective"]) + okey, []).append(r)
        groups.update(mine)                             # file sau thay trọn đường λ của file trước
    return groups, has_outlier, has_mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, nargs="+", default=[PATH_CSV])
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--selections", nargs="+", default=["val_objective", "val_mse"], choices=SELECTIONS)
    args = ap.parse_args()
    groups, has_outlier, has_mag = read_paths(args.inp)
    keep = KEEP + (OUTLIER_KEYS + OUTLIER_EXTRA if has_outlier else []) + (["outlier_magnitude"] if has_mag else [])
    out = []
    for key in sorted(groups):
        path = sorted(groups[key], key=lambda r: float(r["lam"]))
        lams = [float(r["lam"]) for r in path]
        if len(set(lams)) != len(lams):
            sys.exit(f"đường λ {key} có λ lặp")
        if len({r["delta"] for r in path}) > 1:
            sys.exit(f"đường λ {key} trộn nhiều δ")
        zero = next((r for r in path if float(r["lam"]) == 0), None)
        for sel in args.selections:
            metric = OWN[key[3]] if sel == "val_objective" else sel
            best = min(path, key=lambda r: float(r[metric]))
            edge = float(best["lam"]) in (lams[0], lams[-1])
            for level, r in (("lam0", zero), ("lam_star", best)):
                if r is None:
                    continue
                row = {c: r.get(c, "") for c in keep}
                if has_mag and not row["outlier_magnitude"]:
                    row["outlier_magnitude"] = DEFAULT_MAGNITUDE if row.get("outlier_mode") else ""
                out.append({**row, "selection": sel, "selection_metric": metric,
                            "level": level, "lam_at_edge": edge if level == "lam_star" else "",
                            "n_lams": len(path), "lam_min": lams[0], "lam_max": lams[-1]})
    cols = ["selection", "selection_metric", "level", "lam_at_edge", "n_lams", "lam_min", "lam_max"] + keep
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    n_sel = len(args.selections)
    print(f"{len(groups)} đường λ → {len(out)} dòng ({len(out) // n_sel} ô mỗi cách chọn) → {args.out}")
    for r in out:
        if r["level"] == "lam_star":
            print(f"  {r['dataset']} H={r['H']:>3} {r['model']:<7} {r['objective']:<5} {r['selection']:<13} "
                  f"λ* = {float(r['lam']):>10.4g}{' (đầu mút)' if r['lam_at_edge'] else '':<10} "
                  f"val {float(r[r['selection_metric']]):.6f}  test MSE {float(r['test_mse']):.6f} "
                  f"MAE {float(r['test_mae']):.6f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
