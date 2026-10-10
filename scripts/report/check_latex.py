r"""Kiểm tĩnh báo cáo LaTeX khi máy không có TeX (NHIEM_VU_8, B.4). Không thay cho biên dịch thật.

    python scripts/report/check_latex.py

Kiểm trên report/main.tex và mọi file nó \input:
1. mọi macro \nm... dùng trong phần chữ đều được định nghĩa trong generated/numbers.tex;
2. mọi \ref{...} có \label{...} tương ứng; mọi \label là duy nhất;
3. mọi \cite{...} có trong refs.bib;
4. ngoặc nhọn cân bằng trong từng file (bỏ \{ \} và dòng chú thích); \begin/\end khớp nhau;
5. mọi file \input và \includegraphics tồn tại;
6. mỗi bảng, hình (\label{tab:..}, \label{fig:..}) được nhắc tới (\ref) ở một vị trí trước nhãn của nó;
7. liệt kê mọi \TODO còn lại.
In lỗi và trả mã thoát 1 nếu có lỗi ở mục 1–5.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "report"


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def collect(f, out):
    t = strip_comments(f.read_text(encoding="utf-8"))
    out.append((f, t))
    for m in re.finditer(r"\\input\{([^}]+)\}", t):
        p = REP / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        if not p.exists():
            raise SystemExit(f"thiếu file \\input: {p}")
        if p.name != "numbers.tex" and "generated" not in p.parts:
            collect(p, out)
        else:
            out.append((p, strip_comments(p.read_text(encoding="utf-8"))))


def main():
    files = []
    collect(REP / "main.tex", files)
    text = "\n".join(t for _, t in files)
    errors, warns = [], []

    defined = set(re.findall(r"\\newcommand\{\\(nm[A-Za-z]+)\}", (REP / "generated" / "numbers.tex").read_text(encoding="utf-8")))
    used = set(re.findall(r"\\(nm[A-Za-z]+)", text)) - defined
    errors += [f"macro chưa định nghĩa: \\{u}" for u in sorted(used)]

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    dup = {l for l in labels if labels.count(l) > 1}
    errors += [f"nhãn trùng: {d}" for d in sorted(dup)]
    for r in sorted(set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))):
        if r not in labels and not r.startswith("#"):          # #1: tham số của \newcommand
            errors.append(f"\\ref không có \\label: {r}")

    bib = (REP / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    for c in re.findall(r"\\cite\w*\{([^}]+)\}", text):
        for k in c.split(","):
            if k.strip() not in keys:
                errors.append(f"\\cite không có trong refs.bib: {k.strip()}")
    uncited = keys - {k.strip() for c in re.findall(r"\\cite\w*\{([^}]+)\}", text) for k in c.split(",")}
    warns += [f"mục refs.bib không được trích: {k}" for k in sorted(uncited)]

    for f, t in files:
        if f.name == "preamble.tex":
            continue
        s = t.replace(r"\{", "").replace(r"\}", "")
        if s.count("{") != s.count("}"):
            errors.append(f"ngoặc nhọn lệch trong {f.relative_to(REP)}: {s.count('{')} mở, {s.count('}')} đóng")
        stack = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", t):
            if m.group(1) == "begin":
                stack.append(m.group(2))
            elif not stack or stack.pop() != m.group(2):
                errors.append(f"\\end{{{m.group(2)}}} không khớp trong {f.relative_to(REP)}")
        if stack:
            errors.append(f"\\begin chưa đóng trong {f.relative_to(REP)}: {stack}")
        if f.parent.name == "sections" and "$" in s.replace(r"\$", "") and s.replace(r"\$", "").count("$") % 2:
            errors.append(f"số dấu $ lẻ trong {f.relative_to(REP)}")

    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        if not any((REP / d / g).exists() for d in ("", "figures", "template")):
            errors.append(f"thiếu hình: {g}")

    # bảng, hình được nhắc trước khi xuất hiện
    for lab in [l for l in labels if l.startswith(("tab:", "fig:"))]:
        pos_label = text.find(f"\\label{{{lab}}}")
        refs = [m.start() for m in re.finditer(r"\\(?:ref|tabref|figref)\{" + re.escape(lab) + r"\}", text)]
        if not refs:
            warns.append(f"{lab} không được nhắc tới")
        elif min(refs) > pos_label:
            warns.append(f"{lab} được nhắc lần đầu sau khi xuất hiện")

    todos = [(f.relative_to(REP).as_posix(), m.group(1)) for f, t in files
             for m in re.finditer(r"\\TODO\{((?:[^{}]|\{[^{}]*\})*)\}", t)]

    print(f"{len(files)} file; {len(defined)} macro số; {len(labels)} nhãn; {len(keys)} mục bib")
    for w in warns:
        print("CẢNH BÁO:", w)
    print(f"\\TODO còn lại: {len(todos)}")
    for f, t in todos:
        print(f"  {f}: {t}")
    if errors:
        print("LỖI:")
        print("\n".join("  " + e for e in errors))
        sys.exit(1)
    print("Không có lỗi tĩnh.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
