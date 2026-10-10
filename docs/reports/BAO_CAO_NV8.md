# Nhiệm vụ 8: dọn repo và báo cáo LaTeX

Ngày: 11/10/2026. Nhiệm vụ: [NHIEM_VU_8.md](../tasks/NHIEM_VU_8.md). Thay đổi code: [NHAT_KY_THAY_DOI.md](../NHAT_KY_THAY_DOI.md), mục 13. Không chạy thí nghiệm mới, trừ phép kiểm một ô (D.1), ghi vào thư mục tạm. Không sửa file nào trong `results/`, chỉ thêm `results/README.md`. Test: **63, tất cả đạt** sau mỗi phần.

**Trước khi bắt đầu, người dùng đã chọn** (hỏi ở đầu phiên vì sau đó máy sẽ tắt): không cài TeX; để `\TODO` cho thông tin bìa; vẫn tắt máy khi xong, và ghi lại thay vì chờ nếu gặp điểm dừng.

## 1. Cấu trúc repo sau khi dọn, và các quyết định ở A.4

Cấu trúc theo đúng A.1 (sơ đồ trong [README.md](../../README.md)). Bảng đường dẫn cũ → mới: [DUONG_DAN_CU.md](../DUONG_DAN_CU.md). Chỉ mục kết quả: [results/README.md](../../results/README.md).

- **Script:** `scripts/grid/` (runner, select_lambda, huber_deltas), `scripts/sgd/` (sgd_etth2, sgd_lr_tune, optim_etth2, official_ltsf), `scripts/checks/` (bench_5060ti, checks_mae, constant_windows, gd_early_stop, irls_dtype_check, mae_filtered, recheck_paths, ridge_w0, stopping_check, compare_cell mới), `scripts/report/` (report_* cũ, cùng make_numbers, make_tables, make_figures, audit_numbers, check_latex, run_notebooks, report_data mới).
- **Tài liệu:** `docs/tasks/` (NHIEM_VU_2–8, NHIEM_VU_5060TI), `docs/reports/` (BAO_CAO_*), `docs/plan/` (chỉ có CHECKLIST_CON_LAI.md; xem mục 6), `docs/refs/` (chưa có, PDF bị .gitignore).
- **Không xóa file nào** ngoài `__pycache__/`. `scratch/` (đã ignore) giữ nguyên, vì `official_ltsf.py` dùng `scratch/ltsf_official` và checkpoint của code gốc nằm ở đó. `checkpoints/` giữ ở gốc (notebook 00 dùng); A.1 không nhắc thư mục này.
- **A.4, LTSF-Linear:** giữ **git submodule**. Repo vốn đã dùng submodule, ghim đúng commit `0c11366` (`git ls-tree HEAD third_party/LTSF-Linear`). Cách này giữ nguyên lịch sử và `git clone --recurse-submodules` lấy đúng commit, không cần thêm logic clone vào `run_all.sh`. `run_all.sh` chỉ gọi `git submodule update --init` khi thiếu.
- **A.4, file lớn:** `results/weights/` (5,0 GB, không phải 1,9 GB như đề ghi) và mọi `.npy` lớn vẫn ở ngoài git. Cách sinh lại ghi trong `results/README.md`. `tests/test_val_metrics.py` tự bỏ qua khi thiếu `weights/`.
- **`.gitignore`:** thêm `.venv*/`, `docs/refs/*.pdf`, `report/build/*` (trừ `main.pdf`), các file phụ của LaTeX. **`.gitattributes`** (mới): `*.sh` và `report/latexmkrc` giữ LF. Máy này đặt `core.autocrlf = true`, nên thiếu dòng này thì `run_all.sh` trong bản clone mang CRLF và bash không chạy được.
- **A.5:** 63 test đạt; `git status` sạch. Notebook `00_dlinear_sgd_baseline`, `00_nlinear_closed_form`, `01_three_predictions` chạy từ đầu được với đường dẫn mới (chạy trên bản sao bằng `scripts/report/run_notebooks.py --check 00 01`, để không ghi đè output cũ). Notebook 00 ghi đè checkpoint đã track khi chạy, nên checkpoint đã được khôi phục bằng `git checkout`. `00_nlinear_closed_form` có một cell lỗi từ trước (`NameError: pred_V`, đã ghi trong checklist); cell này đã được sửa (mục 6). Notebook 02, 03 chạy từ đầu và lưu kèm output.

## 2. Báo cáo PDF

- **PDF chưa được biên dịch**, nên chưa biết số trang. Máy không có TeX (không có pdflatex, xelatex, lualatex, latexmk, MiKTeX), và người dùng chọn không cài. Đây là điểm dừng của mục 7; đã ghi lại thay vì chờ.
- Nguồn: `report/main.tex`, lệnh biên dịch ở [report/README.md](../../report/README.md) (`cd report && latexmk main.tex` → `report/build/main.pdf`). Template dùng **pdfLaTeX** (gói `vietnam`, `babel` tiếng Việt, `biblatex` backend `bibtex`), không phải XeLaTeX. Preamble chép nguyên từ `report/template/template.tex`; mọi chỗ đổi có chú thích `NV8`.
- **Thay cho biên dịch:** `scripts/report/check_latex.py` kiểm tĩnh 26 file: 623 macro số đều được định nghĩa; 50 nhãn, không nhãn trùng, mọi `\ref` có nhãn; mọi `\cite` có trong `refs.bib`; ngoặc nhọn và `\begin`/`\end` cân bằng; mọi file `\input`, `\includegraphics` tồn tại; mọi bảng, hình được nhắc trước khi xuất hiện. **Không có lỗi tĩnh.** Lần biên dịch đầu tiên vẫn có thể gặp lỗi gói hoặc overfull box mà kiểm tĩnh không bắt được.
- Hai chỗ đã chủ động tránh lỗi biên dịch dù kiểm tĩnh không bắt được: mục từ của bảng thuật ngữ (`\newglossaryentry`) đặt trong preamble (`sections/glossary_entries.tex`), vì `\makenoidxglossaries` yêu cầu như vậy; chú thích trong khối code `listings` (Phụ lục D) viết ASCII, vì listings dưới pdfLaTeX thường lỗi với ký tự UTF-8 nhiều byte.
- Độ dài: khoảng 9 900 từ (âm tiết) cho cả phần chính lẫn phụ lục, chưa kể bảng. Phần chính ước khoảng 20 trang, nhưng chưa đo được.
- **`\TODO` còn lại (người dùng điền):**
  1. tên đề tài chính thức (`\ReportTitle`, `report/preamble.tex`);
  2. tên tác giả (`\ReportAuthor`);
  3. tên TA hoặc người hướng dẫn (`\ReportReviewer`);
  4. ngày nộp (`\ReportDate`);
  5. đoạn Toner & Darlow ở mục II (`sections/02_lien_quan.tex`): chưa có PDF trong `docs/refs/`, nên chỉ có câu giữ chỗ, không mô tả nội dung;
  6. đầu mục VI (`sections/06_han_che.tex`): đối chiếu danh sách hạn chế với 8 hạn chế của kế hoạch 3.5 mục 2 (repo không có file kế hoạch, mục 6).

## 3. Kết quả `audit_numbers.py`

`python scripts/report/audit_numbers.py`: **20 số thập phân nằm ngoài macro trong `report/sections/*.tex`, 0 số cần rà tay.** Mọi số đo được (kể cả số đếm như "27/36") đi qua macro `\nm…` của `report/generated/numbers.tex`. 20 số ngoài macro thuộc các loại:

- 7 tham số thiết kế: hằng số 1,345 và 1,4826 của δ_MAD (trong văn bản mục II), lr gốc 0,005 / 0,05 / 0,0001, bội lr ×0,5 (2 lần);
- 2 số trong công thức hiển thị (định nghĩa δ_MAD, mục III);
- 8 số phiên bản phần mềm, CUDA, driver (Phụ lục D);
- 3 tham chiếu mục tài liệu ("kế hoạch 3.5 mục 4.1").

Số nguyên thiết kế (L = 336, H, số seed, tỷ lệ outlier, độ lớn, 36 ô, 216 ô) không bị tìm, đúng ngoại lệ của B.3. Bảng (`tab_*.tex`) sinh hoàn toàn từ `results/`. Số công bố của LTSF-Linear cũng dùng macro (đọc từ `sgd_b1_b4/summary.json`, vốn chép đúng mục 0.1 của NHIEM_VU_3).

Hình: `make_figures.py` gọi **đúng các hàm vẽ đã sinh PNG trong `results/`**, với cùng dữ liệu đầu vào. Chỉ đổi đầu ra (PDF vector vào `report/figures/`), bỏ tiêu đề chung, và đặt phông chung. Nhãn trục vẫn là nhãn tiếng Việt cũ, số trên trục dùng dấu chấm thập phân (matplotlib mặc định).

## 4. Kiểm trên môi trường trắng (D.2)

Bản clone mới (`git clone --recurse-submodules`, commit `b0028b5`) ở thư mục tạm, venv mới `.venv-run` tạo bằng Python 3.13.5 hệ thống, rồi `./run_all.sh quick`. **Mọi bước đạt, mã thoát 0, tổng 1147 s (19 phút 7 giây).**

| Bước | Kết quả | Thời gian |
|---|---|---|
| clone | có submodule LTSF-Linear | 5 s |
| 1. Môi trường | venv mới; torch 2.14.0+cu130 (từ cache pip), `requirements.txt`; CUDA khả dụng | 205 s |
| 2. Test | 63 test, OK (1 bỏ qua: `test_val_metrics` vì bản clone không có `results/weights/`) | 30 s |
| 3. Grid ETTh1 H = 96 chạy lại, so `results/` | **KHỚP**: 488/488 dòng, chênh 0 với MSE, MAE, Huber (ngưỡng 1e-10 / 1e-8) | 885 s |
| 4. Sinh số, bảng, hình; audit; kiểm tĩnh | đạt; 0 số cần rà; không lỗi tĩnh | 27 s |
| 5. Biên dịch | bỏ qua: máy không có latexmk/pdflatex | 0 s |

- **Vượt mục tiêu 15 phút**, chủ yếu do bước 3 (mục 6, điểm 11). Không tính bước 3 thì khoảng 4,5 phút; `SKIP_CELL=1` bỏ bước này.
- Sau khi chạy, `git status` của bản clone chỉ có `report/generated/numbers.tex` đổi, ở đúng macro `\nmReproHead`: lúc đó macro còn là HEAD. Đã sửa ở commit `d08d3ed` (dùng commit cuối đổi `results/`); sinh lại trên repo chính sau sửa không tạo khác biệt.
- Không thử trên Linux/macOS (không có máy), chỉ Windows + Git Bash.

## 5. `refs.bib`: mục cần người dùng kiểm metadata

Máy không có PDF nào trong `docs/refs/`, nên mọi mục được viết từ hiểu biết sẵn có, không đối chiếu với bản gốc. Chỉ giữ các mục mà tiêu đề, tác giả, nơi đăng, năm đều chắc chắn; không ghi số trang khi không chắc.

- **`toner2024analysis`** (Toner & Darlow, ICML 2024): đánh dấu `% TODO: kiểm metadata`. Tên bài và danh sách tác giả cần đối chiếu PDF. Báo cáo chỉ trích trong câu `\TODO`.
- **`holland1977robust`** (Holland & Welsch, 1977): đánh dấu `% TODO: kiểm metadata` (số tập, số trang).
- Nên kiểm lại cả các mục còn lại khi có PDF: `zeng2023transformers`, `zhou2021informer`, `huber1964robust`, `huber2009robust`, `ali2019continuous`, `bishop1995training`, `kingma2015adam`.

## 6. Những chỗ phải chọn một trong nhiều cách

1. **Kế hoạch 3.5 và 3.4 không có trong repo** (`docs/.../Ke_hoach_4_tuan_v3_5.md`; đã tìm trong repo, lịch sử git và thư mục người dùng). Mọi kết luận vì vậy lấy từ các báo cáo kỹ thuật (chủ yếu BAO_CAO_NV7 mục 1, 7; SGD_B1_B4; ETTH2_2 mục 6; OUTLIER) và từ outline. Dẫn xuất ở Phụ lục A viết lại từ notebook 01 và các công thức đã kiểm, không theo "kế hoạch 3.4 mục 1". Danh sách 8 hạn chế (mục VI) được lập từ các báo cáo, kèm `\TODO` đối chiếu. Hạn chế số 5 là lưới lr, đúng như đề yêu cầu.
2. **Kết luận về NLinear ở RQ1.** Bản nháp đầu viết "ở λ\*, cả hai khác biệt đều rất nhỏ". Kiểm lại trên B3 thì sai với NLinear: NLinear − Linear trên MSE test là −0,029 đến −0,348 ở ETTh2. Đã sửa ở tóm tắt, RQ1, thảo luận, kết luận: DLinear ≈ Linear ở λ\*, còn NLinear tách rõ trên ETTh2. Câu này đọc thẳng từ bảng B3, không có nguyên văn trong báo cáo kỹ thuật nào. **Người dùng nên rà.**
3. **Hai nhận xét đọc từ bảng, không có nguyên văn trong báo cáo kỹ thuật** (theo mục 7 của đề đây là điểm dừng; đã viết dưới dạng mô tả bảng và ghi lại ở đây để người dùng quyết định giữ hay bỏ):
   - RQ3: "với Linear và DLinear trên ETTh1, khoảng cách SGD − dạng đóng (lr gốc) tăng theo horizon, với NLinear thì không" (khung trả lời RQ3).
   - RQ1: "ràng buộc tổng hàng có lợi rõ trên ETTh2 (tăng theo H) và ở ETTh1 H = 720, nhỏ ở các ô ETTh1 còn lại, hơi bất lợi trên ETTm1".
   Giả thuyết "outlier trong x tác động như nhiễu đầu vào ≈ Tikhonov" (mục V) được gọi rõ là giả thuyết chưa kiểm, đúng NHIEM_VU_7 mục 8. Giả thuyết "DLinear có lr hiệu dụng lớn hơn" cũng vậy.
4. **Chỗ đặt `bench_5060ti.py`:** để ở `scripts/checks/` như A.1, dù `runner.py` (grid) import nó. Mọi script thêm cả bốn thư mục con vào `sys.path`, nên chỗ đặt không ảnh hưởng import. Cách này đổi ít code nhất (chỉ đường dẫn).
5. **Chú thích bảng:** template chỉ có kiểu hộp `captionbelowboxaivn` với số bảng gõ tay. Đã thêm bộ đếm `bang` và macro `\captab` (tương tự `\capfig` của template) để `\label`/`\ref` dùng được. Không đổi kiểu hộp. Bảng thuật ngữ dùng kiểu glossary `clean3col` có sẵn trong template.
6. **B3 trong phần chính:** "bản gọn" là 36 dòng (λ\* của ba mục tiêu, MSE test và MAE test). Bản đủ (cả λ = 0, λ\*, δ) ở Phụ lục C. B4 trong phần chính là bảng đếm nhãn (lr gốc 20 seed, lr gốc 15 seed mới, tinh chỉnh); số từng ô ở phụ lục.
7. **Số "thời gian grid":** báo cáo dùng tổng cột `seconds` của `lambda_path.csv` (14,6 giờ GPU), không dùng ước lượng 19 giờ của BAO_CAO_5060TI.
8. **Commit trong Phụ lục D:** macro `\nmReproHead` là commit cuối cùng thay đổi `results/`, không phải HEAD. Nhờ vậy sinh lại không tạo khác biệt sau mỗi commit.
9. **Sửa notebook cũ:** cell cuối của `00_nlinear_closed_form.ipynb` đổi `pred_V.sum(axis= 1)` → `V.sum(axis=1)`. Đây là vi phạm tổng hàng của nghiệm NLinear, song song với dòng `W_ols` ngay trên. Không đổi cell nào khác, không chạy lại để giữ output cũ.
10. **Thư viện chạy notebook** (`nbclient`, `nbformat`, `ipykernel`) cài vào `.venv-gpu` nhưng không thêm vào `requirements.txt`; README ghi là tùy chọn.
11. **`run_all.sh quick` dài hơn 15 phút một chút:** riêng chạy lại grid ETTh1 H = 96 (488 dòng, ba mục tiêu) mất **14 phút 50 giây** trên RTX 5060 Ti (MAE chiếm khoảng 13 phút). Đề yêu cầu grid đầy đủ cho ô này, nên giữ nguyên và cho `SKIP_CELL=1` để bỏ. Không có GPU thì bước này tự bỏ qua. Kết quả chạy lại trong phiên này: **488/488 dòng khớp `results/lambda_path.csv` tuyệt đối** (chênh 0 ở mọi metric, cả MSE, MAE, Huber).
12. **Tùy chọn của mục 0** (mở rộng lưới lr ở các ô đầu mút): không làm, như mặc định. Ghi ở hạn chế 5. Dữ liệu cho thấy 15/36 ô chọn lr ở đầu mút; đề ghi "9 ô" theo kế hoạch 3.5, nhưng chưa đối chiếu được vì thiếu file kế hoạch.

## Tái lập

```bash
./run_all.sh quick                                  # hoặc VENV=.venv-gpu ./run_all.sh quick
python scripts/report/run_notebooks.py              # chạy lại notebook 02, 03
python scripts/report/run_notebooks.py --check 00 01
./run_all.sh full --dry-run
```
