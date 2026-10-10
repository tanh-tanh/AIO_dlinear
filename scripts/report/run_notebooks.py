"""Chạy notebook từ đầu và lưu kèm output (NHIEM_VU_8, Phần C và A.5).

    python scripts/report/run_notebooks.py                      # 02, 03: chạy và ghi output vào notebook
    python scripts/report/run_notebooks.py --check 00 01        # chạy bản sao trong thư mục tạm, không ghi đè

Cần nbclient, nbformat, ipykernel (không có trong requirements.txt chính; xem README). Thư mục làm việc của kernel là
notebooks/, như khi mở bằng Jupyter. --check dùng cho các notebook cũ: chạy để kiểm đường dẫn, không đổi file.
"""
import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbclient
import nbformat

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks"
DEFAULT = ["02_loss_comparison.ipynb", "03_outliers_and_sgd.ipynb"]


def run(path, cwd, timeout=3600):
    nb = nbformat.read(path, as_version=4)
    t = time.time()
    nbclient.NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()
    outs = [o for c in nb.cells for o in c.get("outputs", [])]
    errs = sum(o.output_type == "error" for o in outs)
    imgs = sum("image/png" in o.get("data", {}) for o in outs)
    return nb, time.time() - t, errs, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", default=None, help="tiền tố tên notebook cần chạy thử (không ghi đè)")
    a = ap.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if a.check is None:
        for name in DEFAULT:
            nb, sec, errs, imgs = run(NB / name, NB)
            nbformat.write(nb, NB / name)
            print(f"{name}: {sec:.0f} s, {imgs} hình, {errs} lỗi")
        return
    names = [p.name for p in sorted(NB.glob("*.ipynb")) if any(p.name.startswith(c) for c in a.check)]
    bad = 0
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / name
            shutil.copy(NB / name, dst)
            try:
                _, sec, errs, imgs = run(dst, NB)
                print(f"{name}: chạy từ đầu được, {sec:.0f} s, {imgs} hình, {errs} lỗi")
            except nbclient.exceptions.CellExecutionError as e:
                bad += 1
                print(f"{name}: LỖI\n{str(e)[-1500:]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
