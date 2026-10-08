"""Các cột val_mse, val_mae, val_huber của results/lambda_path.csv không chứa số hạng phạt λ‖w‖² (NHIEM_VU_7, B.2).

Tính lại metric val từ W đã lưu (results/weights/, float64) trên đúng ma trận cửa sổ val của runner, so với cột
trong CSV. Phép phản chứng: cộng số hạng phạt (chuẩn hóa như train_obj, chia n_train·H) vào metric thì lệch
khỏi CSV vượt xa ngưỡng so khớp, nên test phân biệt được hai trường hợp.

W bị .gitignore (results/weights/), nên test bỏ qua khi thiếu file.
"""
import csv
import os
import unittest

import numpy as np

from src.data import load_ett, make_dataset
from src.operators import build_N, build_P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_CSV = os.path.join(ROOT, "results", "lambda_path.csv")
WEIGHTS = os.path.join(ROOT, "results", "weights")
L, K = 336, 25
CELLS = [("ETTh1", 96), ("ETTh2", 336)]
HUBER_DELTA = 1.0
RTOL = 1e-12


def huber(R, delta):
    a = np.abs(R)
    q = np.minimum(a, delta)
    return float((q * (a - q / 2)).mean())


def add_ones(X):
    return np.concatenate([X, np.ones((len(X), 1))], axis=1)


def lam_key(lam):
    return f"{float(lam):.6g}"


@unittest.skipUnless(os.path.exists(PATH_CSV) and os.path.isdir(WEIGHTS), "thiếu lambda_path.csv hoặc results/weights/")
class TestValMetricsHaveNoPenalty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PATH_CSV, encoding="utf-8", newline="") as f:
            cls.rows = list(csv.DictReader(f))
        Ninv = np.linalg.inv(build_N(build_P(L, K)))
        cls.pens = {"Linear": np.eye(L), "DLinear": Ninv, "NLinear": np.diag(np.r_[np.ones(L - 1), 0.0])}

    def check_cell(self, ds, H):
        train, val, _ = load_ett(os.path.join(ROOT, "data", f"{ds}.csv"), L, ds)
        n_train = len(make_dataset(train, L, H)[0])
        Xv, Yv = make_dataset(val, L, H)
        Xv = add_ones(Xv)
        rows = [r for r in self.rows if r["dataset"] == ds and int(r["H"]) == H and float(r["lam"]) > 0]
        n_checked, min_gap = 0, np.inf
        for r in rows:
            # MSE có 141 λ: lấy mỗi λ thứ 10 (vẫn gồm λ lớn nhất); MAE, Huber lấy mọi λ
            if r["objective"] == "MSE" and rows.index(r) % 10 and float(r["lam"]) < 1e5:
                continue
            f = os.path.join(WEIGHTS, f"{ds}_H{H}_{r['model']}_{r['objective']}_lam{lam_key(r['lam'])}.npy")
            if not os.path.exists(f):
                continue
            W = np.load(f)
            R = Yv - Xv @ W.T
            got = {"val_mse": float((R ** 2).mean()), "val_mae": float(np.abs(R).mean()),
                   "val_huber": huber(R, HUBER_DELTA)}
            pen = float(r["lam"]) * np.einsum("hi,ij,hj->", W[:, :-1], self.pens[r["model"]], W[:, :-1])
            pen /= n_train * H
            for col, v in got.items():
                csv_v = float(r[col])
                with self.subTest(ds=ds, H=H, model=r["model"], obj=r["objective"], lam=r["lam"], col=col):
                    self.assertLessEqual(abs(v - csv_v), RTOL * abs(csv_v))
                    min_gap = min(min_gap, pen / abs(csv_v))
            n_checked += 1
        return n_checked, min_gap

    def test_val_columns_match_unpenalized(self):
        for ds, H in CELLS:
            n, gap = self.check_cell(ds, H)
            if n == 0:
                self.skipTest(f"không có W của {ds} H={H}")
            # phản chứng: số hạng phạt (tương đối) ở dòng nhỏ nhất vẫn lớn hơn ngưỡng so khớp ít nhất 1e3 lần
            self.assertGreater(gap, 1e3 * RTOL, f"{ds} H={H}: số hạng phạt quá nhỏ để phân biệt")


if __name__ == "__main__":
    unittest.main()
