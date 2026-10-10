r"""Kiểm chéo số trong phần chữ của báo cáo (NHIEM_VU_8, B.3).

    python scripts/report/audit_numbers.py            # in danh sách để rà tay; mã thoát 0
    python scripts/report/audit_numbers.py --strict   # mã thoát 1 nếu còn số chưa được xếp loại

Tìm mọi số dạng thập phân (0,37 / 0{,}37 / 0.37) trong report/sections/*.tex nằm ngoài macro \nm... của
generated/numbers.tex, bỏ qua dòng chú thích. Mỗi số được xếp vào một loại ngoại lệ hợp lệ, hoặc "CẦN RÀ":
- thiết kế: tham số thiết kế (hằng số của δ_MAD, bội lr, lr gốc, ngưỡng tol...), ghi trong DESIGN;
- phiên bản: số phiên bản phần mềm, driver;
- công thức: nằm trong môi trường toán hiển thị (equation, align, \[ \]).
Số nguyên (L = 336, H, số seed, tỷ lệ outlier, độ lớn) là tham số thiết kế và không bị tìm.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "report" / "sections"

NUM = re.compile(r"(?<![\w\\{])(\d+(?:\{,\}|,|\.)\d+)")
DESIGN = {
    "1{,}345": "hằng số Huber (hiệu quả 95%)", "1{,}4826": "hằng số MAD", "0{,}5": "bội lr / hệ số 1/2",
    "0{,}005": "lr gốc ETTh1", "0{,}05": "lr gốc ETTh2", "0{,}0001": "lr gốc ETTm1",
}
VERSION = re.compile(r"(Python|numpy|pandas|scipy|matplotlib|PyTorch|driver|CUDA|torch)[^.\n]{0,12}$")


def display_math_spans(text):
    spans = []
    for pat in (r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", r"\\\[.*?\\\]"):
        spans += [m.span() for m in re.finditer(pat, text, flags=re.S)]
    return spans


def audit():
    rows = []
    for f in sorted(SECTIONS.glob("*.tex")):
        text = f.read_text(encoding="utf-8")
        spans = display_math_spans(text)
        for m in NUM.finditer(text):
            start = m.start()
            line_start = text.rfind("\n", 0, start) + 1
            line = text[line_start:text.find("\n", start)]
            if line.lstrip().startswith("%"):
                continue
            lineno = text.count("\n", 0, start) + 1
            v = m.group(1)
            key = v.replace(",", "{,}") if "{,}" not in v else v
            prefix = text[max(line_start, start - 40):start]
            if any(a <= start < b for a, b in spans):
                kind = "công thức"
            elif key in DESIGN:
                kind = f"thiết kế ({DESIGN[key]})"
            elif "." in v and VERSION.search(prefix):
                kind = "phiên bản"
            elif "." in v and re.search(r"(kế hoạch|mục)\s*$", prefix):
                kind = "tham chiếu mục tài liệu"
            else:
                kind = "CẦN RÀ"
            ctx = text[max(line_start, start - 50):min(len(text), m.end() + 30)].replace("\n", " ")
            rows.append((f.name, lineno, v, kind, ctx))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    rows = audit()
    todo = [r for r in rows if r[3] == "CẦN RÀ"]
    print(f"Số thập phân ngoài macro trong report/sections/*.tex: {len(rows)}; cần rà tay: {len(todo)}")
    for r in rows:
        print(f"  {r[0]}:{r[1]}  {r[2]:>10}  [{r[3]}]  …{r[4]}…")
    if a.strict and todo:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
