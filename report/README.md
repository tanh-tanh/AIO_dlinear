# Báo cáo LaTeX

Báo cáo cuối của đề tài, theo template AI VIET NAM (AIO2026) ở `template/` (preamble chép nguyên vào `preamble.tex`; mọi chỗ đổi có chú thích `NV8`).

## Biên dịch

Template dùng **pdfLaTeX** với gói `vietnam` và `babel` tiếng Việt, `biblatex` (backend `bibtex`), `glossaries` (kiểu `noidx`, không cần makeindex). Cần một bản TeX đủ gói (TeX Live đầy đủ, hoặc MiKTeX tự cài gói khi thiếu) và `latexmk` (cần Perl).

```bash
cd report
latexmk main.tex          # cấu hình ở latexmkrc: pdflatex, bibtex, thư mục build/
# → build/main.pdf
```

Không có latexmk: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` (đặt `TEXINPUTS=./template//:` trước).

**Trạng thái (11/10/2026):** máy dùng để viết báo cáo không có TeX, nên PDF **chưa được biên dịch**. Thay vào đó đã chạy kiểm tĩnh `python scripts/report/check_latex.py` (macro, nhãn, trích dẫn, ngoặc, môi trường, file, `\TODO`). Lần biên dịch đầu tiên có thể còn lỗi gói hoặc cảnh báo overfull box cần sửa.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `main.tex` | khung tài liệu, thứ tự các mục |
| `preamble.tex` | preamble của template + vài macro thêm (`\TODO`, `\captab`, `\figref`, thông tin bìa) |
| `sections/` | nội dung: `00_tom_tat`, `00_thuat_ngu`, `01`–`07`, phụ lục `A`–`D` |
| `generated/` | **tự sinh, không sửa tay**: `numbers.tex` (macro số), `tab_*.tex` (bảng) |
| `figures/` | **tự sinh**: H1–H5 dạng PDF vector |
| `refs.bib` | tài liệu tham khảo; mục `% TODO: kiểm metadata` cần đối chiếu |
| `template/` | template gốc của khóa học, giữ nguyên |

## Sinh lại số, bảng, hình

```bash
python scripts/report/make_numbers.py     # → generated/numbers.tex
python scripts/report/make_tables.py      # → generated/tab_*.tex
python scripts/report/make_figures.py     # → figures/*.pdf
python scripts/report/audit_numbers.py    # số thập phân ngoài macro trong sections/
python scripts/report/check_latex.py      # kiểm tĩnh
```

Mọi con số đo được trong `sections/*.tex` là macro `\nm...` của `generated/numbers.tex` (quy ước tên ở đầu file đó). Ba script đọc `results/` qua `scripts/report/report_data.py` và không ghi gì vào `results/`.

## Thông tin bìa

Đã điền trong `preamble.tex`: `\ReportTitle`, `\ReportAuthor`, `\ReportDate`. Bìa không có dòng Proofreading.
