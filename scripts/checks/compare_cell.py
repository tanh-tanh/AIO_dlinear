"""So một ô grid chạy lại với results/lambda_path.csv (NHIEM_VU_8, D.1, bước 3 của run_all.sh quick).

    python scripts/checks/compare_cell.py --new <thư mục>/lambda_path.csv [--data ETTh1 --H 96]

Mỗi dòng (dataset, H, mô hình, mục tiêu, λ) của file mới phải có trong results/lambda_path.csv, với chênh tuyệt đối
của các metric (train_obj, val_mse, val_mae, val_huber, test_mse, test_mae) không quá 1e-10 với mục tiêu MSE và
1e-8 với MAE, Huber. Ngược lại, mọi dòng của ô đó trong results/ phải có trong file mới. Thoát mã 1 nếu không khớp.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["train_obj", "val_mse", "val_mae", "val_huber", "test_mse", "test_mae"]
TOL = {"MSE": 1e-10, "MAE": 1e-8, "Huber": 1e-8}


def load(path, data, H):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {(r["dataset"], r["H"], r["model"], r["objective"], f"{float(r['lam']):.6g}"): r
                for r in csv.DictReader(f) if r["dataset"] == data and r["H"] == str(H)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", default=str(ROOT / "results" / "lambda_path.csv"))
    ap.add_argument("--data", default="ETTh1")
    ap.add_argument("--H", type=int, default=96)
    a = ap.parse_args()
    new, ref = load(a.new, a.data, a.H), load(a.ref, a.data, a.H)
    bad, worst = [], {o: 0.0 for o in TOL}
    for k, r in new.items():
        if k not in ref:
            bad.append(f"thiếu trong results/: {k}")
            continue
        for m in METRICS:
            if r[m] == "" and ref[k][m] == "":
                continue
            d = abs(float(r[m]) - float(ref[k][m]))
            worst[k[3]] = max(worst[k[3]], d)
            if d > TOL[k[3]]:
                bad.append(f"{k} {m}: mới {r[m]}, cũ {ref[k][m]}, chênh {d:.3e}")
    for k in ref.keys() - new.keys():
        bad.append(f"thiếu trong file mới: {k}")
    print(f"{a.data} H={a.H}: {len(new)} dòng mới, {len(ref)} dòng trong results/")
    for o, d in worst.items():
        print(f"  {o}: chênh lớn nhất {d:.3e} (ngưỡng {TOL[o]:.0e})")
    if bad:
        print("KHÔNG KHỚP:")
        print("\n".join("  " + b for b in bad[:50]))
        sys.exit(1)
    print("KHỚP")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
