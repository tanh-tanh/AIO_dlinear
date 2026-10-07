"""Kiểm src/outliers.py (docs/NHIEM_VU_6.md, A.2 và C)."""
import os
import unittest

import numpy as np
import pandas as pd

from src.data import load_ett, make_dataset, split_borders
from src.outliers import contaminate_series, contaminate_targets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ETTh1.csv")
L, H = 336, 96


def _full_series(name="ETTh1"):
    """Chuỗi đã chuẩn hóa đầy đủ (train + val + test) và các mốc chia, như load_ett."""
    train_end, val_end, test_end = split_borders(name)
    data = pd.read_csv(DATA).iloc[:, 1:].values
    mean, std = data[:train_end].mean(0), data[:train_end].std(0)
    return (data - mean) / std, train_end, val_end, test_end


class TestContaminateSeriesSynthetic(unittest.TestCase):
    def setUp(self):
        self.s = np.random.default_rng(0).standard_normal((1000, 7))

    def test_count(self):
        for rate in (0.01, 0.05, 0.1, 0.123):
            _, info = contaminate_series(self.s, rate, seed=1, protect_last=100)
            self.assertEqual(info["n_eligible"], 900 * 7)
            self.assertEqual(info["n_contaminated"], int(np.floor(rate * 900 * 7)))
            self.assertEqual(int(info["mask"].sum()), info["n_contaminated"])

    def test_protected_rows_untouched(self):
        new, info = contaminate_series(self.s, 0.1, seed=1, protect_last=100)
        self.assertFalse(info["mask"][-100:].any())
        self.assertEqual(new[-100:].tobytes(), self.s[-100:].tobytes())

    def test_magnitude_and_only_masked_change(self):
        new, info = contaminate_series(self.s, 0.1, magnitude=10.0, seed=1, protect_last=100)
        m = info["mask"]
        np.testing.assert_allclose(np.abs(new[m] - self.s[m]), 10.0, rtol=0, atol=1e-12)
        self.assertEqual(new[~m].tobytes(), self.s[~m].tobytes())
        d = new[m] - self.s[m]
        self.assertTrue((d > 0).any() and (d < 0).any())          # có cả hai dấu

    def test_seed(self):
        a, ia = contaminate_series(self.s, 0.05, seed=7, protect_last=100)
        b, ib = contaminate_series(self.s, 0.05, seed=7, protect_last=100)
        c, ic = contaminate_series(self.s, 0.05, seed=8, protect_last=100)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertTrue((ia["mask"] == ib["mask"]).all())
        self.assertFalse((ia["mask"] == ic["mask"]).all())

    def test_rate_zero(self):
        new, info = contaminate_series(self.s, 0.0, seed=1, protect_last=100)
        self.assertEqual(info["n_contaminated"], 0)
        self.assertEqual(new.tobytes(), self.s.tobytes())

    def test_input_not_modified(self):
        before = self.s.copy()
        contaminate_series(self.s, 0.1, seed=1, protect_last=100)
        self.assertEqual(before.tobytes(), self.s.tobytes())

    def test_window_fractions(self):
        _, info = contaminate_series(self.s, 0.01, seed=1, protect_last=100, L=50, H=20)
        mx, my = make_dataset(info["mask"].astype(float), 50, 20)
        self.assertAlmostEqual(info["frac_windows_x"], float((mx > 0).any(1).mean()))
        self.assertAlmostEqual(info["frac_windows_y"], float((my > 0).any(1).mean()))
        self.assertGreater(info["frac_windows_x"], 0)


@unittest.skipUnless(os.path.isfile(DATA), "thiếu data/ETTh1.csv")
class TestContaminateSeriesETTh1(unittest.TestCase):
    """Val và test phải giống hệt từng byte: ghép train đã chèn vào chuỗi đầy đủ rồi cắt lại."""

    def test_val_test_unchanged(self):
        full, train_end, val_end, test_end = _full_series()
        train, val, test = load_ett(DATA, L, "ETTh1")
        self.assertEqual(train.tobytes(), full[:train_end].tobytes())
        for rate in (0.01, 0.1):
            new_train, info = contaminate_series(train, rate, seed=2026, protect_last=L)
            # vùng cấm: L hàng cuối train là phần đầu của val
            self.assertFalse(info["mask"][train_end - L:].any())
            full_c = full.copy()
            full_c[:train_end] = new_train
            self.assertNotEqual(full_c.tobytes(), full.tobytes())
            val_c = full_c[train_end - L: val_end]
            test_c = full_c[val_end - L: test_end]
            for H_ in (96, 720):
                for a, b in ((val_c, val), (test_c, test)):
                    Xa, Ya = make_dataset(a, L, H_)
                    Xb, Yb = make_dataset(b, L, H_)
                    self.assertEqual(Xa.tobytes(), Xb.tobytes())
                    self.assertEqual(Ya.tobytes(), Yb.tobytes())

    def test_protect_too_small_would_touch_val(self):
        """Phản chứng: không bảo vệ L hàng cuối thì (với rate đủ lớn) val bị đổi."""
        full, train_end, val_end, _ = _full_series()
        train, val, _ = load_ett(DATA, L, "ETTh1")
        new_train, info = contaminate_series(train, 0.1, seed=2026, protect_last=0)
        self.assertTrue(info["mask"][train_end - L:].any())
        full_c = full.copy()
        full_c[:train_end] = new_train
        self.assertNotEqual(full_c[train_end - L: val_end].tobytes(), val.tobytes())


class TestContaminateTargets(unittest.TestCase):
    def setUp(self):
        self.Y = np.random.default_rng(0).standard_normal((500, 30))

    def test_count_magnitude(self):
        for rate in (0.01, 0.1):
            new, info = contaminate_targets(self.Y, rate, seed=3)
            self.assertEqual(info["n_contaminated"], int(np.floor(rate * self.Y.size)))
            m = info["mask"]
            self.assertEqual(int(m.sum()), info["n_contaminated"])
            np.testing.assert_allclose(np.abs(new[m] - self.Y[m]), 10.0, rtol=0, atol=1e-12)
            self.assertEqual(new[~m].tobytes(), self.Y[~m].tobytes())
            self.assertEqual(info["frac_windows_x"], 0.0)
            self.assertAlmostEqual(info["frac_windows_y"], float(m.any(1).mean()))

    def test_seed_and_zero(self):
        a, ia = contaminate_targets(self.Y, 0.05, seed=3)
        b, _ = contaminate_targets(self.Y, 0.05, seed=3)
        _, ic = contaminate_targets(self.Y, 0.05, seed=4)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertFalse((ia["mask"] == ic["mask"]).all())
        z, iz = contaminate_targets(self.Y, 0.0, seed=3)
        self.assertEqual(iz["n_contaminated"], 0)
        self.assertEqual(z.tobytes(), self.Y.tobytes())
        before = self.Y.copy()
        contaminate_targets(self.Y, 0.1, seed=3)
        self.assertEqual(before.tobytes(), self.Y.tobytes())


if __name__ == "__main__":
    unittest.main()
