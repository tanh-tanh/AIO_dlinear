# Nhiệm vụ 8 (cuối cùng) cho Claude Code: dọn repo và viết báo cáo LaTeX hoàn chỉnh

*Lập ngày 08/10/2026. Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_8.md và làm theo".*

Đây là nhiệm vụ cuối của đề tài. **Không chạy thí nghiệm mới**, trừ phép kiểm tái lập ở Phần D và Phần 0 nếu người dùng yêu cầu. Mọi con số trong báo cáo lấy từ kết quả đã có.

## 0. Trước khi bắt đầu: ba thứ người dùng phải chuẩn bị

Kiểm đủ ba thứ sau. Thiếu thứ nào thì **dừng lại hỏi người dùng** ngay, chưa làm gì khác.

1. **Template LaTeX của khóa học** (AI VIET NAM, AIO2026), đặt ở `report/template/`: file `.cls` hoặc preamble `.tex`, logo, font nếu có, và một file ví dụ. Báo cáo phải dùng **đúng** template này: class, màu, header, footer, kiểu hộp, kiểu tiêu đề. **Không tự dựng lại template theo trí nhớ hay theo ảnh chụp.**
2. **Thông tin bìa:** tên đề tài chính thức, tên tác giả, tên TA hoặc người hướng dẫn (nếu có), ngày nộp. Thiếu thì để `\TODO{...}` và liệt kê trong báo cáo cuối phiên.
3. **Bài Toner & Darlow (ICML 2024)**, dạng PDF, ở `docs/refs/`. Nếu chưa có: phần công trình liên quan chỉ viết một câu giữ chỗ có `\TODO`, **không mô tả nội dung bài này theo trí nhớ**.

**Tùy chọn (chỉ làm nếu người dùng yêu cầu rõ):** mở rộng lưới learning rate ở 9 ô đầu mút (kế hoạch 3.5 mục 4.1). Mặc định **không làm**, và ghi vào hạn chế số 5.

## 1. Quy tắc bắt buộc

- **Không chạy thí nghiệm mới, không sửa kết quả cũ.** Không sửa file CSV, JSON trong `results/`.
- **Không sửa toán hay hành vi của `src/`.** Chỉ được đổi đường dẫn import khi dời file, và mọi test phải vẫn đạt.
- **Mọi con số trong báo cáo phải sinh tự động từ file kết quả** (mục 3.3), không gõ tay. Ngoại lệ: số trích từ bài báo khác (số công bố của LTSF-Linear lấy đúng như mục 0.1 của `NHIEM_VU_3.md`).
- **Chỉ viết những kết luận có trong kế hoạch 3.5 mục 1** (`docs/.../Ke_hoach_4_tuan_v3_5.md`) hoặc trong các báo cáo kỹ thuật. Giả thuyết chưa kiểm phải được gọi là giả thuyết.
- **Không bịa trích dẫn.** Chỉ đưa vào `refs.bib` những tài liệu có metadata xác định được (tiêu đề, tác giả, hội nghị hoặc tạp chí, năm). Mục nào không chắc thì đánh dấu `% TODO: kiểm metadata` trong `.bib` và liệt kê ở báo cáo cuối phiên.
- Dời file bằng `git mv` để giữ lịch sử. Commit theo từng phần, mỗi commit một việc.
- Sau mỗi phần: chạy lại toàn bộ test; kỳ vọng **63 test, tất cả đạt**.

---

## 2. Phần A: dọn thư mục

### A.1. Cấu trúc đích

```
dlinear-objective-lab/
├── README.md
├── run_all.sh
├── requirements.txt
├── .gitignore
├── src/                    # thư viện: data, operators, solvers, models, sgd, metrics, outliers
├── tests/
├── scripts/
│   ├── grid/               # runner.py, select_lambda.py, huber_deltas.py
│   ├── sgd/                # chạy SGD, tinh chỉnh lr, code gốc LTSF-Linear
│   ├── checks/             # các script kiểm của từng nhiệm vụ (dtype, bench, stopping, checks_mae,
│   │                       #   constant_windows, gd_early_stop, …)
│   └── report/             # report_*.py, make_tables.py, make_figures.py, make_numbers.py
├── notebooks/
│   ├── 00_*.ipynb, 01_three_predictions.ipynb, 02_loss_comparison.ipynb, 03_outliers_and_sgd.ipynb
│   └── archive/
├── results/                # giữ nguyên nội dung; thêm results/README.md (A.3)
├── report/                 # LaTeX (Phần B)
│   ├── template/  main.tex  sections/  generated/  figures/  refs.bib  README.md
│   └── build/main.pdf
├── docs/
│   ├── NHAT_KY_THAY_DOI.md
│   ├── DUONG_DAN_CU.md     # bảng đường dẫn cũ → mới (A.2)
│   ├── outline.docx
│   ├── plan/               # các bản kế hoạch, checklist
│   ├── tasks/              # NHIEM_VU_*.md
│   ├── reports/            # BAO_CAO_*.md (báo cáo kỹ thuật theo nhiệm vụ)
│   └── refs/               # PDF tài liệu tham khảo (không commit nếu có bản quyền; xem A.4)
└── third_party/            # LTSF-Linear (A.4)
```

Đây là cấu trúc đề xuất. Nếu repo đang có một quy ước khác hợp lý hơn, giữ quy ước đó và ghi lý do.

### A.2. Dời file

- Dời script, tài liệu vào đúng thư mục ở A.1. Sửa mọi import, đường dẫn trong code, notebook, test, `README.md` cho khớp.
- **Không sửa nội dung các báo cáo kỹ thuật cũ** (`docs/reports/BAO_CAO_*.md`), vì chúng là hồ sơ của từng phiên. Thay vào đó, ghi `docs/DUONG_DAN_CU.md`: bảng đường dẫn cũ → mới cho mọi file đã dời, để các lệnh "Tái lập" trong báo cáo cũ vẫn tra được.
- Các script một lần, đã có kết quả và không còn dùng: chuyển vào `scripts/checks/`, không xóa.
- Chỉ xóa: `__pycache__/`, `.ipynb_checkpoints/`, file tạm, `scratch/` (đã ignore). Trước khi xóa bất kỳ file nào khác, liệt kê và hỏi người dùng.

### A.3. Chỉ mục kết quả

Viết `results/README.md`: mỗi bảng (B1–B4) và hình (H1–H5) trong báo cáo → file dữ liệu → script sinh ra → báo cáo kỹ thuật mô tả. Thêm một dòng cho mỗi thư mục con của `results/`: nội dung, nhiệm vụ nào tạo, có commit vào git không.

### A.4. Phụ thuộc ngoài

- `third_party/LTSF-Linear`: chuyển thành git submodule ghim ở commit `0c11366`, hoặc bỏ khỏi git và để `run_all.sh` clone đúng commit đó. Chọn một cách và ghi lý do.
- `results/weights/` (khoảng 1,9 GB) và các file lớn khác: giữ ngoài git như hiện tại; ghi trong `results/README.md` cách sinh lại.
- `docs/refs/*.pdf`: không commit nếu là bài báo có bản quyền; thêm vào `.gitignore`.
- `.gitignore`: rà lại cho đủ (`.venv*`, `scratch/`, `report/build/` trừ `main.pdf` nếu người dùng muốn commit PDF, các file phụ của LaTeX).

### A.5. Kiểm sau khi dọn

- 63 test đạt.
- Mọi notebook chạy được từ đầu (Kernel → Restart & Run All) với đường dẫn mới, trừ notebook trong `archive/`.
- `git status` sạch.

---

## 3. Phần B: báo cáo LaTeX

### B.1. Ngôn ngữ và văn phong

- **Tiếng Việt**, văn phong học thuật, câu ngắn, rõ chủ ngữ. Giữ nguyên thuật ngữ tiếng Anh thông dụng (MSE, MAE, Huber, ridge, OLS, IRLS, SGD, Adam, seed, horizon, learning rate) và giải thích một lần ở lần xuất hiện đầu.
- Lập một bảng thuật ngữ trong `report/sections/00_thuat_ngu.tex` (ví dụ: nghiệm dạng đóng, nghiệm tối ưu, hàm mục tiêu, tập kiểm định, tập kiểm tra, đoạn cảm biến bị kẹt) và dùng nhất quán.
- Số thập phân dùng dấu phẩy như các báo cáo kỹ thuật. Nếu template quy định khác, theo template.
- Độ dài mục tiêu: phần chính khoảng 15–25 trang, phụ lục không giới hạn.

### B.2. Dàn ý (theo kế hoạch 3.5 mục 3 và outline)

| Mục | Nội dung | Nguồn |
|---|---|---|
| Bìa, tóm tắt | Tóm tắt khoảng 200 từ: vấn đề, phương pháp, ba kết quả chính | kế hoạch 3.5 mục 1 |
| 1. Giới thiệu | Bối cảnh LTSF-Linear; câu hỏi "ưu thế đến từ kiến trúc hay cách huấn luyện"; ba RQ; đóng góp | outline mục 1 |
| 2. Công trình liên quan | LTSF-Linear; Toner & Darlow (chỉ khi có PDF, mục 0); hồi quy bền vững và IRLS; dừng sớm và ridge (Ali, Kolter & Tibshirani, 2019); nhiễu đầu vào và Tikhonov (Bishop, 1995) | `docs/refs/` |
| 3. Phương pháp và thiết lập | Dữ liệu, cách chia, chuẩn hóa; ba mô hình viết dưới dạng W_eff; ba mục tiêu; nghiệm dạng đóng; IRLS (tóm tắt, chi tiết ở phụ lục); quy ước λ và hàm phạt từng mô hình; chọn λ*; δ_MAD; SGD (siêu tham số gốc, tinh chỉnh lr); thiết kế outlier; quy tắc đọc ba trường hợp; thống kê (seed, khoảng tin cậy) | kế hoạch 3.4 mục 3–4; `BAO_CAO_NV7.md` |
| 4.1. RQ1 | Bảng B2, hình H1; DLinear − Linear trên B3; ETTh1 H = 720; DLinear so với Linear khi huấn luyện bằng SGD (lr gốc và tinh chỉnh) | kế hoạch 3.5 mục 1 |
| 4.2. RQ2 | B3 (bản gọn trong phần chính, bản đủ ở phụ lục); MAE so với MSE; vị trí của Huber; outlier (H2, H5); độ nhạy δ (H4) | đoạn nháp RQ2, `BAO_CAO_NV7.md` mục 7 |
| 4.3. RQ3 | B1, B4 (lr gốc và tinh chỉnh), H3; tóm tắt các cơ chế đã loại trừ trên ETTh2 | đoạn nháp RQ3, `BAO_CAO_NV7.md` mục 7 |
| 4.4. ETTh2 và đoạn cảm biến bị kẹt | Phân tích độ nhạy | đoạn nháp, `BAO_CAO_ETTH2_2.md` mục 6 |
| 5. Thảo luận | Kiến trúc so với huấn luyện; outlier ở x so với ở y; giả thuyết nhiễu đầu vào (ghi rõ chưa kiểm); nhiễu theo seed trong các bảng công bố | kế hoạch 3.5 mục 1 |
| 6. Hạn chế và hướng phát triển | 8 hạn chế ở kế hoạch 3.5 mục 2 | |
| 7. Kết luận | Ba câu trả lời ngắn, mỗi RQ một đoạn | kế hoạch 3.5 mục 1 |
| Tài liệu tham khảo | `refs.bib` | |
| Phụ lục A | Dẫn xuất ba dự đoán của RQ1 (đầy đủ, có chứng minh) | kế hoạch 3.4 mục 1 |
| Phụ lục B | IRLS: lập luận MM, tách theo h, bias bằng cột 1, ràng buộc NLinear theo hàng, quy ước λ, chế độ pha độ chính xác, tiêu chí dừng, các phép kiểm | kế hoạch 3.4 mục 3; `BAO_CAO_TIEU_CHI_DUNG.md`; `BAO_CAO_IRLS_DTYPE.md` |
| Phụ lục C | Bảng đầy đủ: B3 (216 ô), B4, lr đã chọn, outlier | các CSV |
| Phụ lục D | Tái lập: môi trường, phần cứng, lệnh, thời gian chạy, commit | `README.md`, các báo cáo |

### B.3. Con số tự động

- Viết `scripts/report/make_numbers.py`: đọc các CSV, JSON trong `results/` và ghi `report/generated/numbers.tex`. File này gồm các macro, ví dụ `\newcommand{\ETThOneHsevtwentyLinearSGDMean}{0,506}`. Đặt tên macro có hệ thống và ghi quy ước đặt tên ở đầu file.
- Viết `scripts/report/make_tables.py`: sinh mọi bảng thành `report/generated/tab_*.tex` (`booktabs`, định dạng số thống nhất, đánh đậm giá trị tốt nhất trong ô theo đúng quy tắc của báo cáo kỹ thuật).
- Viết `scripts/report/make_figures.py`: sinh lại mọi hình thành PDF vector trong `report/figures/`, nhãn tiếng Việt, cùng phông, cùng bảng màu, kích thước hợp với khổ trang của template. Dữ liệu vẽ phải giống hệt các hình đã có trong `results/`.
- Phần chữ trong `sections/*.tex` **chỉ dùng macro** cho mọi con số đo được, kể cả các số đếm như "23/24 ô".
- **Kiểm chéo:** viết `scripts/report/audit_numbers.py`. Script này tìm mọi số dạng thập phân trong `sections/*.tex` không nằm trong macro, và liệt kê để rà tay. Ngoại lệ hợp lệ: số công bố của bài gốc, tham số thiết kế (L = 336, H, tỷ lệ outlier, độ lớn, số seed), số trong công thức. Kết quả audit đưa vào báo cáo cuối phiên.

### B.4. Biên dịch

- Dùng trình biên dịch mà template yêu cầu (thường là XeLaTeX hoặc LuaLaTeX cho tiếng Việt), qua `latexmk`. Ghi lệnh vào `report/README.md`.
- PDF phải biên dịch **không lỗi**. Cảnh báo về tham chiếu chưa xác định hay overfull box lớn cũng phải sửa.
- Nếu máy chưa có TeX: **dừng lại hỏi người dùng** trước khi cài, vì bản cài đầy đủ lớn vài GB.

### B.5. Đọc lại

Sau khi có PDF, tự rà một lượt theo danh sách:

- Mỗi RQ có câu trả lời rõ ở cả phần kết quả và phần kết luận, và khớp kế hoạch 3.5 mục 1.
- Mọi bảng, hình được nhắc trong văn bản trước khi xuất hiện, có chú thích đủ để đọc riêng.
- Mỗi giả thuyết chưa kiểm được gọi rõ là giả thuyết.
- Không còn `\TODO` nào ngoài các mục người dùng phải điền (mục 0); liệt kê các mục còn lại.
- Thuật ngữ thống nhất với bảng thuật ngữ.
- Chính tả tiếng Việt.

---

## 4. Phần C: notebook

- `notebooks/02_loss_comparison.ipynb`: B3, so sánh MAE với MSE, vị trí của Huber, H2, H4, H5.
- `notebooks/03_outliers_and_sgd.ipynb`: B1, B4 (cả bản tinh chỉnh), H3, DLinear so với Linear khi huấn luyện bằng SGD.
- Notebook **gọi các hàm trong `scripts/report/`** để không lặp code. Mỗi notebook mở đầu bằng một ô markdown nói nó trả lời câu hỏi nào và dữ liệu đến từ đâu.
- Chạy từ đầu, lưu kèm output.

## 5. Phần D: `run_all.sh` và `README.md`

### D.1. `run_all.sh`

Hai chế độ:

- `./run_all.sh quick` (mặc định), dưới 15 phút:
  1. tạo môi trường, cài phụ thuộc theo `requirements.txt` (kèm hướng dẫn cài torch bản CUDA phù hợp);
  2. chạy toàn bộ test;
  3. **chạy lại một ô làm phép kiểm**: grid đầy đủ cho ETTh1 H = 96 (cả ba mô hình và ba mục tiêu), ghi ra thư mục tạm, so với `results/` (MSE tới 1e-10, MAE và Huber tới 1e-8);
  4. sinh lại mọi bảng, hình, `numbers.tex` từ kết quả đã lưu;
  5. biên dịch báo cáo nếu có TeX.
- `./run_all.sh full`: chạy lại toàn bộ thí nghiệm theo đúng thứ tự các nhiệm vụ (grid, Huber δ_MAD, outlier, SGD, tinh chỉnh lr), ghi rõ thời gian dự kiến của từng bước (khoảng 19 giờ GPU cho grid cộng vài giờ CPU cho SGD) và phần cứng đã dùng. Chế độ này **không chạy trong phiên này**; chỉ kiểm cú pháp và chạy `--dry-run` nếu các script hỗ trợ.

Script phải chạy được trên Linux, macOS (bash), và trên Windows qua Git Bash. Nếu một bước không chạy được trên Windows, ghi rõ trong README.

### D.2. Kiểm trên môi trường trắng

Tạo một bản clone mới của repo ở thư mục tạm, venv mới, rồi chạy `./run_all.sh quick`. Mọi bước phải đạt. Ghi thời gian từng bước.

### D.3. `README.md`

- Một đoạn mô tả đề tài và ba câu hỏi nghiên cứu.
- **Bảng ba kết quả chính**, mỗi kết quả một dòng, dùng đúng số trong báo cáo.
- Cấu trúc repo (theo A.1).
- Cài đặt: Python, torch (bản cu130 cho RTX 50xx, bản CPU nếu không có GPU), các phụ thuộc khác.
- Tái lập: `run_all.sh quick` và `full`; trỏ tới `results/README.md` và `docs/DUONG_DAN_CU.md`.
- Trỏ tới PDF báo cáo và các báo cáo kỹ thuật trong `docs/reports/`.
- Trích dẫn: bài LTSF-Linear; dữ liệu ETT.

---

## 6. Phần E: báo cáo cuối phiên

Viết `docs/reports/BAO_CAO_NV8.md`, ngắn:

1. Cấu trúc repo sau khi dọn, và các quyết định ở A.4.
2. Đường dẫn PDF; số trang; các `\TODO` còn lại mà người dùng phải điền.
3. Kết quả `audit_numbers.py`.
4. Kết quả kiểm trên môi trường trắng (D.2), kèm thời gian.
5. Các mục trong `refs.bib` cần người dùng kiểm metadata.
6. Mọi chỗ đã phải chọn một trong nhiều cách, kèm lý do.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 7. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Thiếu template ở `report/template/` (mục 0).
- Cần xóa bất kỳ file nào ngoài các loại đã cho phép ở A.2.
- Có test không đạt sau khi dời file.
- Phép kiểm một ô trong `run_all.sh quick` không khớp `results/` trong ngưỡng.
- Máy chưa có TeX và cần cài.
- Template không biên dịch được tiếng Việt với trình biên dịch có sẵn.
- Một kết luận cần viết không có trong kế hoạch 3.5 hay các báo cáo kỹ thuật.

## 8. Không làm trong phiên này

- Không chạy thí nghiệm mới (trừ phép kiểm một ô ở D.1 và việc tùy chọn ở mục 0 nếu người dùng yêu cầu).
- Không sửa nội dung các báo cáo kỹ thuật cũ.
- Không mô tả bài Toner & Darlow nếu chưa có PDF.
- Không công khai repo. Việc này người dùng tự làm sau khi đọc lại báo cáo.
