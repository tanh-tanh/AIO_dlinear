"""Chèn outlier nhân tạo vào train (docs/tasks/NHIEM_VU_6.md, Phần A và C).

contaminate_series: chèn vào chuỗi train đã chuẩn hóa, trước khi cắt cửa sổ. Một điểm outlier vì vậy
    xuất hiện ở nhiều cửa sổ, cả trong x lẫn y.
contaminate_targets: chèn chỉ vào đích y của các cửa sổ train; x giữ nguyên.

Cả hai cộng ±magnitude (dấu ngẫu nhiên, xác suất 1/2) vào đúng ⌊rate × số điểm hợp lệ⌋ điểm, chọn
không lặp. Dữ liệu đã chuẩn hóa theo thống kê của train sạch, nên ±10 là ±10 độ lệch chuẩn của kênh;
scaler không tính lại sau khi chèn. Mọi hàm chỉ dùng numpy và không sửa mảng đầu vào.
"""
import numpy as np

from src.data import make_dataset


def contaminate_series(train_series, rate, magnitude=10.0, seed=0, protect_last=0, L=None, H=None):
    """Chèn outlier vào chuỗi train [T, C].

    Vùng được chèn: các hàng 0 … T − protect_last − 1. Với cách chia của LTSF-Linear, cửa sổ val bắt đầu
    lùi L hàng vào cuối train, nên protect_last = L giữ nguyên mọi hàng mà val dùng tới.

    Trả (chuỗi mới, info). info gồm n_contaminated, n_eligible, mask [T, C] (bool), và nếu cho L, H thì
    thêm frac_windows_x, frac_windows_y: tỷ lệ cửa sổ train (theo make_dataset: một cửa sổ cho mỗi kênh)
    có ít nhất một outlier trong x, trong y.
    """
    s = np.array(train_series, dtype=np.float64, copy=True)
    T, C = s.shape
    T_ok = T - protect_last
    if T_ok <= 0:
        raise ValueError("protect_last không để lại hàng nào để chèn")
    n_eligible = T_ok * C
    n = int(np.floor(rate * n_eligible))
    rng = np.random.default_rng(seed)
    flat = rng.choice(n_eligible, size=n, replace=False)          # chỉ số trong vùng [0, T_ok) × [0, C)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    rows, cols = np.divmod(flat, C)
    s[rows, cols] += sign * magnitude
    mask = np.zeros((T, C), dtype=bool)
    mask[rows, cols] = True
    info = {"n_contaminated": n, "n_eligible": n_eligible, "mask": mask}
    if L is not None and H is not None:
        mx, my = make_dataset(mask.astype(np.float64), L, H)
        info["frac_windows_x"] = float((mx > 0).any(axis=1).mean())
        info["frac_windows_y"] = float((my > 0).any(axis=1).mean())
    return s, info


def contaminate_targets(Y, rate, magnitude=10.0, seed=0):
    """Chèn outlier chỉ vào đích của các cửa sổ train, Y [n_windows, H] (hàng theo make_dataset).

    Chọn đúng ⌊rate × Y.size⌋ phần tử (cửa sổ, kênh, bước h) không lặp. Trả (Y mới, info) với
    n_contaminated, n_eligible, mask, frac_windows_x = 0, frac_windows_y.
    """
    Yc = np.array(Y, dtype=np.float64, copy=True)
    n_eligible = Yc.size
    n = int(np.floor(rate * n_eligible))
    rng = np.random.default_rng(seed)
    flat = rng.choice(n_eligible, size=n, replace=False)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    Yc.reshape(-1)[flat] += sign * magnitude
    mask = np.zeros(Yc.shape, dtype=bool)
    mask.reshape(-1)[flat] = True
    info = {"n_contaminated": n, "n_eligible": n_eligible, "mask": mask,
            "frac_windows_x": 0.0, "frac_windows_y": float(mask.any(axis=1).mean())}
    return Yc, info
