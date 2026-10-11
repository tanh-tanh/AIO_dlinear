# Báo cáo nhiệm vụ 9: góp ý Tuần 3 của TA và hoàn thiện báo cáo

Ngày 11/10/2026. Nhiệm vụ: [docs/tasks/NHIEM_VU_9.md](../tasks/NHIEM_VU_9.md). Nhật ký code: [NHAT_KY_THAY_DOI.md](../NHAT_KY_THAY_DOI.md) mục 14. Mọi bảng dưới đây sinh từ `results/nv9/summary.json` bằng `scripts/report/report_nv9.py` (bản đầy đủ ở `results/nv9/tables_*.md`).

**Không chạm điểm dừng nào của mục 8:** 66/66 test đạt (gồm 3 test mới của δ theo h); RQ3 không đổi (ETTh1 + ETTm1: 23/24 → 23/24 ô SGD kém có ý nghĩa, không ô ngoài ETTh2 có nhãn "SGD tốt hơn thật"); code gốc với `drop_last=True` đạt số công bố ở 1/8 ô (ngưỡng dừng 5/8); code gốc chỉ thêm một tham số; mọi khẳng định về Toner & Darlow tìm được nguồn trong PDF; A dùng 0,84 giờ CPU (ngân sách 4 giờ), C.3 dùng 3 phút GPU (ngân sách 1 giờ).

## 1. Phần A: mở rộng lưới lr

15/15 ô biên khớp danh sách của nhiệm vụ (kiểm với `results/sgd_lr/selection.json`). Chạy `scripts/sgd/sgd_lr_extend.py` (5 tiến trình × 2 luồng, ETTm1 trước), 0,84 giờ thực. Không lần chạy nào phân kỳ.

| Ô | biên | val MSE TB các mức mới (5 seed) | lr cũ → mới | Δ test cũ → mới [±CI] | nhãn cũ → mới | còn ở biên |
|---|---|---|---|---|---|---|
| ETTm1 Linear H=96 | trên (×8: 0,38153) | ×16: 0,38133; ×32: 0,38123 | ×8 → **×32** | 0,0022 [±0,0014] → 0,0020 [±0,0015] | SGD khớp val quá mức (không đổi) | **có** |
| ETTm1 Linear H=192 | trên (×8: 0,48880) | ×16: 0,48898 | ×8 | không đổi | SGD khớp val quá mức | không |
| ETTm1 Linear H=336 | trên (×8: 0,61064) | ×16: 0,61634 | ×8 | không đổi | SGD khớp val quá mức | không |
| ETTm1 NLinear H=96 | trên (×8: 0,38447) | ×16: 0,38729 | ×8 | không đổi | SGD khớp val quá mức | không |
| ETTm1 NLinear H=720 | trên (×8: 0,96073) | ×16: 0,95823; ×32: 0,95855 | ×8 → **×16** | 0,0048 [±0,0029] → 0,0041 [±0,0021] | SGD khớp val quá mức (không đổi) | không |
| ETTh1 Linear H=192 | trên (×8: 0,85630) | ×16: 0,85694 | ×8 | không đổi | val khác test | không |
| ETTh1 DLinear H=720 | trên (×8: 1,20300) | ×16: 1,20653 | ×8 | không đổi | val khác test | không |
| ETTh1 Linear H=720 | dưới (×0,5: 1,19603) | ×0,25: 1,20811 | ×0,5 | không đổi | SGD khớp val quá mức | không |
| ETTh1 DLinear H=336 | dưới (×0,5: 1,04502) | ×0,25: 1,05106 | ×0,5 | không đổi | SGD khớp val quá mức | không |
| ETTh2 Linear H=720 | dưới (×0,5: 0,65305) | ×0,25: 0,65592 | ×0,5 | không đổi | không phân biệt được | không |
| ETTh2 DLinear H=192 | dưới (×0,5: 0,29584) | ×0,25: 0,29631 | ×0,5 | không đổi | SGD tốt hơn thật | không |
| ETTh2 NLinear H=96 | dưới (×0,5: 0,20546) | ×0,25: 0,20553 | ×0,5 | không đổi | dạng đóng tốt hơn thật | không |
| ETTh2 NLinear H=192 | dưới (×0,5: 0,27078) | ×0,25: 0,27016; ×0,125: 0,27106 | ×0,5 → **×0,25** | 0,0131 [±0,0042] → 0,0137 [±0,0036] | **val khác test → SGD khớp val quá mức** | không |
| ETTh2 NLinear H=336 | dưới (×0,5: 0,35856) | ×0,25: 0,36019 | ×0,5 | không đổi | val khác test | không |
| ETTh2 NLinear H=720 | dưới (×0,5: 0,60372) | ×0,25: 0,60360; ×0,125: 0,60128 | ×0,5 → **×0,125** | 0,0159 [±0,0073] → 0,0151 [±0,0051] | val khác test (không đổi) | **có** |

- lr đổi ở 4/15 ô; còn ở biên sau 2 mức mới: **ETTm1 Linear H = 96 (×32)** và **ETTh2 NLinear H = 720 (×0,125)** (ghi vào hạn chế 6). Ở ETTm1 Linear H = 96, ×16 và ×32 chỉ hơn ×8 khoảng 2–3·10⁻⁴ trên val.
- "SGD kém có ý nghĩa trên test": 27/36 → 27/36; ETTh1 + ETTm1: 23/24 → 23/24. Theo trung bình: 28/36 (không đổi). Nhãn (tinh chỉnh): "val khác test" 9 → 8, "SGD khớp val quá mức" 16 → 17, các nhãn khác không đổi (SGD tốt hơn thật 5, dạng đóng tốt hơn thật 2, không phân biệt được 4).
- C.3 (DLinear − Linear, cả hai tinh chỉnh): DLinear tốt hơn có ý nghĩa 0/12 → 0/12. Chỉ ETTm1 H = 96 đổi: +0,0002 [±0,0019] → +0,0004 [±0,0020]; không ô nào Linear tốt hơn có ý nghĩa.
- Sinh lại: Bảng 3 (C.3), bảng đếm nhãn (nay Bảng 7), B4 đầy đủ (nay Bảng 10, mã nhãn T1–T6), MSE val theo lr (nay Bảng 11, 9 cột ×0,125 … ×32, 5 chữ số), cùng mọi macro liên quan. Số bảng trong PDF mới lùi vì có thêm Bảng 6 (`drop_last`) và Bảng 8 (gốc/đã lọc) trong mục IV.
- **Sự cố đã xử lý:** ETTh1 DLinear H = 720 ×16 seed 2022 lần đầu bị ghi "thất bại" vì Windows khóa file checkpoint khi `torch.save` (mã lỗi 1224), không phải phân kỳ. Đã sửa driver (chỉ ghi thất bại khi phân kỳ, lỗi khác thử lại 3 lần rồi dừng, commit `4cc0054`), xóa file `.failed.json`, chạy lại seed đó: ×16 cho val 1,20653, kém ×8, lr giữ ×8. Một lỗi cùng loại ở ETTh2 NLinear H = 720 ×0,25 seed 2022 đã tự qua khi thử lại.

## 2. Phần B: `drop_last` trong code gốc

### B.1. Bằng chứng trong git log của `third_party/LTSF-Linear`

- `f7cbf46` (27/05/2022, "First upload"): `data_provider/data_factory.py`, nhánh `flag == 'test'`: `drop_last = True`.
- `6fe4c28` (15/03/2023 23:51 +0800, "Update data_factory.py"): đổi duy nhất dòng đó: `-        drop_last = True` → `+        drop_last = False`. Không commit nào khác chạm `data_factory.py` ở dòng này (log của file: `f7cbf46`, `27ef399` 13/02/2023 thêm train-only, `6fe4c28`).
- Về bài AAAI 2023, git log chỉ cho biết: README commit `571f26e` (23/11/2022) thêm tiêu đề "(AAAI 2023)" và dòng tin "[2022/11/23] Accepted to AAAI 2023 with three strong accept!"; README có tin "[2022/08/25] We update our paper" (arXiv 2205.13504). Cả hai mốc đều trước `6fe4c28`, tức trong suốt thời gian test loader còn `drop_last=True`. Ngày bài xuất hiện trong kỷ yếu không có trong git log.

### B.2–B.3. Chạy và đọc kết quả

Bản chép `scratch/ltsf_official/` (patch seed của nhiệm vụ 4) + một thay đổi: tham số `--test_drop_last` (`results/official_ltsf/patch_droplast.diff`). Siêu tham số của script gốc, seed 2021–2025, GPU (RTX 5060 Ti, torch 2.14.0+cu130), 90 lần chạy, 2,6 giờ tổng (khoảng 1,3 giờ thực). Kết quả: `results/official_ltsf_droplast/<dataset>/*_dl{0,1}.json`.

| Ô | công bố | `drop_last=True`: TB ± sd [min; max] | `drop_last=False`: TB ± sd [min; max] | True − False, cùng seed: TB [min; max] | bỏ / N cửa sổ | công bố ∈ [min; max] (True / False) |
|---|---|---|---|---|---|---|
| ETTh1 H=192 DLinear | 0,405 | 0,4201 ± 0,0142 [0,4070; 0,4434] | 0,4202 ± 0,0142 [0,4070; 0,4435] | −0,00004 [−0,00009; −0,00002] | 1 / 2689 | không / không |
| ETTh1 H=720 DLinear | 0,472 | 0,5091 ± 0,0176 [0,4909; 0,5384] | 0,5101 ± 0,0176 [0,4919; 0,5393] | −0,00095 [−0,00109; −0,00086] | 17 / 2161 | không / không |
| ETTm1 H=96 DLinear | 0,299 | 0,3019 ± 0,0010 [0,3008; 0,3031] | 0,3018 ± 0,0010 [0,3008; 0,3031] | +0,00002 | 1 / 11425 | không / không |
| ETTm1 H=192 DLinear | 0,335 | 0,3376 ± 0,0026 [0,3356; 0,3419] | 0,3376 ± 0,0026 [0,3356; 0,3419] | +0,00001 | 1 / 11329 | không / không |
| ETTm1 H=336 DLinear | 0,369 | 0,3723 ± 0,0025 [0,3695; 0,3761] | 0,3723 ± 0,0025 [0,3695; 0,3761] | −0,00001 | 1 / 11185 | không / không |
| ETTh2 H=336 NLinear | 0,357 | 0,3618 ± 0,0033 [0,3568; 0,3647] | 0,3670 ± 0,0032 [0,3621; 0,3698] | −0,00519 [−0,00530; −0,00507] | 17 / 2545 | **có** / không |
| ETTh2 H=720 NLinear | 0,394 | 0,4064 ± 0,0146 [0,3959; 0,4313] | 0,4087 ± 0,0149 [0,3980; 0,4340] | −0,00231 [−0,00278; −0,00205] | 17 / 2161 | không / không |
| ETTm1 H=336 NLinear | 0,375 | 0,3787 ± 0,0047 [0,3754; 0,3866] | 0,3787 ± 0,0047 [0,3754; 0,3866] | −0,00001 | 1 / 11185 | không / không |
| ETTh2 H=720 DLinear (đối chứng) | 0,605 | 0,6645 ± 0,0700 [0,5731; 0,7318] | 0,6655 ± 0,0699 [0,5746; 0,7328] | −0,00107 [−0,00143; −0,00067] | 17 / 2161 | có / có |

Epoch tốt nhất (seed 2021–2025, giống hệt giữa True và False ở mọi ô): ETTh1 192: 3, 4, 5, 5, 7; ETTh1 720: 3, 5, 5, 4, 1; ETTm1 DLinear 96: 4, 2, 5, 3, 2; 192: 5, 5, 1, 3, 2; 336: 3, 2, 3, 3, 10; ETTh2 NLinear 336: 10, 8, 8, 9, 8; 720: 9, 10, 6, 1, 6; ETTm1 NLinear 336: 8, 2, 4, 3, 5; ETTh2 DLinear 720: 9, 8, 5, 10, 4. Số epoch đã chạy ghi trong từng JSON (`n_epochs_run`).

**Kết luận:** với `drop_last=True`, số công bố nằm trong [min; max] của 5 seed ở **1/8** ô (ETTh2 NLinear H = 336); với `False`: 0/8 (số công bố thấp hơn mọi seed ở 8/8). `drop_last` **không đủ giải thích** khoảng chênh ở phần lớn ô. Mức chênh do `drop_last` không quá 0,0053; chỉ đáng kể ở ETTh1, ETTh2 (batch 32, bỏ tới 17 cửa sổ cuối, các cửa sổ khó hơn trung bình) và gần bằng 0 ở ETTm1 (batch 8, bỏ 1 cửa sổ). Huấn luyện của hai chế độ trùng nhau (chấm lại trên đủ cửa sổ bằng pipeline của repo: chênh dưới 1e-12 ở mọi seed). Code gốc trên GPU ở ETTm1 H = 96 (0,3018 ± 0,0010) trùng lần chạy CPU của nhiệm vụ 6 D.1 (0,3018 ± 0,0010).

B.4: đoạn số công bố trong báo cáo được viết lại theo cách TA đề nghị (mục 4 dưới đây); bỏ mọi câu "không tái lập được".

## 3. Phần C

### C.1. Họ ridge trên ETTh2 (đọc `results/lambda_path.csv`, 142 λ, không chạy lại)

"Ridge tốt nhất theo test" = MSE test nhỏ nhất trên toàn lưới: **mức tham chiếu chọn theo test, không hợp lệ để chọn mô hình**, chỉ để chỉ ra giới hạn của cả họ ridge.

| Ô | λ = 0: val / test | λ\*_MSE: λ; val / test | ridge tốt nhất theo test: λ; test | SGD lr gốc (20 seed) | SGD tinh chỉnh (15 seed) | tốt nhất − SGD gốc / − tinh chỉnh |
|---|---|---|---|---|---|---|
| Linear H=96 | 0,2219 / 0,3019 | 1584,89; 0,2212 / 0,3009 | 1995,26; 0,3009 | 0,2968 | 0,2989 | +0,0040 / +0,0020 |
| Linear H=192 | 0,3032 / 0,3972 | 1122,02; 0,3028 / 0,3966 | 1584,89; 0,3966 | 0,3961 | 0,3941 | +0,0005 / +0,0024 |
| Linear H=336 | 0,4090 / 0,4858 | 1000; 0,4087 / 0,4853 | 1584,89; 0,4853 | 0,4496 | 0,4508 | +0,0357 / +0,0345 |
| Linear H=720 | 0,6676 / 0,7407 | 1000; 0,6673 / 0,7404 | 1000; 0,7404 | 0,6967 | 0,7089 | +0,0437 / +0,0314 |
| DLinear H=96 | 0,2219 / 0,3019 | 1995,26; 0,2213 / 0,3008 | 2818,38; 0,3007 | 0,2913 | 0,2925 | +0,0094 / +0,0082 |
| DLinear H=192 | 0,3032 / 0,3972 | 1258,93; 0,3029 / 0,3966 | 1995,26; 0,3965 | 0,3807 | 0,3746 | +0,0158 / +0,0219 |
| DLinear H=336 | 0,4090 / 0,4858 | 1122,02; 0,4087 / 0,4853 | 1778,28; 0,4853 | 0,4529 | 0,4515 | +0,0324 / +0,0338 |
| DLinear H=720 | 0,6676 / 0,7407 | 1122,02; 0,6672 / 0,7403 | 1122,02; 0,7403 | 0,6869 | 0,6897 | +0,0534 / +0,0507 |

**Ở 8/8 ô, kể cả ridge tốt nhất theo test vẫn kém SGD trung bình** (cả lr gốc lẫn tinh chỉnh). Không ô nào có λ tốt nhất theo test ở đầu mút lưới. Đã thêm vào "Các cơ chế đã loại trừ": không giá trị λ nào của ridge đạt mức của SGD trên ETTh2, nên lợi thế của SGD không phải một dạng chính quy hóa L2 mạnh hơn. (Ridge tốt nhất theo test chỉ hơn ridge ở λ\* tối đa 0,0001.)

### C.2. Đoạn cảm biến bị kẹt (không chạy lại)

- **Cách phát hiện** (`scripts/checks/constant_windows.py`): cửa sổ một kênh, sau chuẩn hóa, có max − min của x hoặc của y nhỏ hơn 1e-12. Cửa sổ hằng của train chỉ ở MUFL, LUFL, LULL; HULL có cửa sổ đích hằng ở val (H = 96) và test (H = 96, 192).
- **Vị trí** (`report_nv9.constant_runs`, chuỗi gốc): 163 đoạn mà một kênh lặp lại cùng một giá trị ≥ 24 giờ; 63 trong số đó dài đúng 24 giờ và rơi vào ngày 31 của tháng (12 ngày 31, ở cả 7 kênh), quá ngắn để tạo cửa sổ hằng. 19 đoạn ≥ 96 giờ: 10 ở train (MUFL, LUFL, LULL; dài nhất 1025 giờ, MUFL 19/04–01/06/2017), 1 ở val (HULL), 8 ở test (HULL, LULL). Danh sách đủ (≥ 24 giờ): `results/nv9/tables_C2.md`; báo cáo in các đoạn ≥ 96 giờ (Bảng 13, Phụ lục C; khác yêu cầu "≥ 24 giờ" vì 163 dòng quá dài cho báo cáo, caption trỏ tới danh sách đủ) và Hình 6 (ba kênh tải, đoạn hằng, ranh giới train/val/test).
- **Bảng gốc / đã lọc** (Bảng 8 của PDF): MSE test của dạng đóng λ\*, SGD trung bình (20 seed; NLinear 10 seed, chưa từng chạy trên train đã lọc nên ghi "--"), MSE test của MAE λ = 0, lợi thế của MAE so với MSE (cùng λ = 0). DLinear ở λ = 0 trùng Linear nên cột MAE lấy của Linear (đánh dấu ‡).

### C.3. Huber với δ theo bước h

Code: `irls` nhận `delta` vectơ [H] (commit `6159f06`, 3 test mới đạt). Chạy `scripts/checks/huber_delta_h.py` (GPU, 3 phút, tol 1e-9, chế độ pha, đường λ như grid).

| Ô | δ_MAD | δ_h: min–max (TB) | MSE | MAE | Huber δ_MAD | Huber δ_h | hạng của Huber, δ_MAD / δ_h (MSE test; MAE test) |
|---|---|---|---|---|---|---|---|
| ETTh1 H=96 Linear | 0,5380 | 0,2426–0,5952 (0,5415) | 0,3702 / 0,3915 | 0,3653 / 0,3833 | 0,3658 / 0,3852 | 0,3656 / 0,3851 | 2 / 2; 2 / 2 |
| ETTh2 H=720 Linear | 0,6096 | 0,1743–0,8010 (0,6134) | 0,7404 / 0,6115 | 0,5893 / 0,5335 | 0,6277 / 0,5541 | 0,6357 / 0,5566 | 2 / 2; 2 / 2 |

(Ô dạng "MSE test ở λ\*_MSE / MAE test ở λ\*_MAE".) **δ theo bước không đổi vị trí của Huber** ở cả hai ô: vẫn giữa MSE và MAE theo cả hai metric. Ở ETTh2, δ_h làm Huber kém hơn một chút (MSE test 0,6277 → 0,6357).

## 4. Các câu và số đã đổi trong báo cáo (theo mục; đối chiếu `git diff 74f1000 -- report/`)

**Lỗi hiển thị (D.1):**
- Danh mục tham khảo: "andothers", "inProceedings", "jourvol", "number", "pages" → "et al.", "in Proceedings", "vol.", "no.", "pp." (`\DeclareLanguageMapping{vietnamese}{english}`). Cùng lúc tắt phép so thứ tự trích dẫn của biber (backend bibtex luôn báo "There were undefined references" và đòi chạy lại BibTeX vì biblatex 3.22a không nạp danh sách trích dẫn lần trước); PDF không có "??".
- Toner & Darlow: `@misc` arXiv → `@inproceedings` ICML (PMLR), tên đủ hai tác giả từ PDF.
- Bảng B4 đầy đủ: cột nhãn dài → mã T1–T6, chú giải trong caption; bảng quy tắc (mục III.7) thêm cột mã.
- Tham chiếu chéo: `\thesection` bỏ dấu chấm cuối (tiêu đề và mục lục tự thêm), danh sách hạn chế `ref=\arabic*`: "mục IV.3..", "mục V..", "hạn chế 5.)" → "mục IV.3.", "mục V.", "hạn chế 5)". Kiểm trên toàn PDF: không còn mẫu "số..", "số.)".
- Caption: bỏ "(B1)", "(B2)", "(B3, bản gọn)", "(B4)", "(H1)"…"(H5)"; chữ đầu viết hoa. Văn bản: "bảng B3" → "Bảng 4"/"Bảng 9"; "B1: tái lập số công bố" → "Số công bố"; "B4: khoảng cách…" → "Khoảng cách…"; tiêu đề phụ lục "B3 đầy đủ", "B4 đầy đủ" → "Nghiệm tối ưu, đủ mọi tổ hợp", "SGD so với nghiệm dạng đóng, từng ô".
- Vị trí caption: mọi caption bảng chuyển lên trên bảng; caption hình giữ dưới. Template cho phép (hộp màu dùng được ở bất kỳ chỗ nào), nhưng hộp của template tên là `captionbelowboxaivn`; giữ tên hộp, chỉ đổi vị trí.
- Dấu thập phân trong hình: dấu phẩy (chặn `ScalarFormatter` trong `make_figures.py`); kiểm bằng `pdftotext`: không còn "x.y" trên trục.
- Mọi bảng, hình được nhắc trước khi xuất hiện (kiểm tự động 23/23 nhãn: `scratch/nv9_reforder.py`).

**Nội dung (D.2):**
- Tóm tắt (1): "Trên mọi dataset và horizon" → "Trên ETTh1, ETTh2, ETTm1 và mọi horizon". (3): "Ngoại lệ chỉ có trên ETTh2." → "SGD tốt hơn nghiệm tối ưu chỉ xảy ra trên ETTh2, ở 5 ô, và không giá trị λ nào của ridge đạt được mức đó." (số ô không đổi sau Phần A).
- III.1: thêm "ETTh1 và ETTm1 đo cùng một máy biến áp, chỉ khác tần suất lấy mẫu; ETTh2 đo một máy biến áp khác. Vì vậy ba dataset chỉ là hai nguồn dữ liệu độc lập (hạn chế 1)."
- III.3: "δ = 1 của các phiên trước" → "δ = 1 (mặc định của `HuberLoss` trong PyTorch)"; thêm câu về chuẩn hóa theo σ̂_h (δ_h = 1,345·σ̂_h) và tham chiếu tới IV.2.
- III.6: thêm quy tắc mở rộng lưới lr (×0,25, ×0,125; ×16, ×32; tối đa 2 mức; phân kỳ = thất bại).
- IV.1: không đổi nội dung (số C.3 tinh chỉnh ETTm1 đổi theo Phần A: khoảng Δ test ETTm1 −0,0020–0,0002 → −0,0020–0,0004).
- IV.2: "cả 4 horizon của cả ba dataset" → "của ETTh1, ETTh2 và ETTm1"; "δ = 1 cố định của các phiên trước" → "(mặc định của `HuberLoss` trong PyTorch)"; đoạn mới "δ theo từng bước h" (kết quả C.3, Bảng 14).
- IV.3, số công bố: viết lại theo B.4: "Ở 8/36 ô còn lại, chúng tôi không đạt được số công bố với 20 seed (10 seed với NLinear trên ETTh2)… Chạy chính code gốc ở cả 8 ô, mỗi ô 5 seed, với `drop_last=False`, số công bố cũng thấp hơn mọi seed ở 8/8 ô (môi trường: …)". Bỏ câu "chúng tôi không quy khoảng chênh … cho may rủi theo seed". Đoạn mới về `drop_last` (bằng chứng git, Bảng 6, kết luận "không đủ giải thích … ở phần lớn ô").
- IV.3, lr tinh chỉnh: "lr được chọn trải đều trên các mức (×0,5: 8 ô; ×1: 7; ×2: 7; ×4: 7; ×8: 7). 15/36 ô chọn mức ở đầu mút" → "(×0,125: 1 ô; ×0,25: 1; ×0,5: 6; ×1: 7; ×2: 7; ×4: 7; ×8: 5; ×16: 1; ×32: 1). Với lưới ×0,5 đến ×8, 15/36 ô chọn mức ở đầu mút. Mở rộng lưới … đổi lr ở 4 ô; 2 ô vẫn chọn mức xa nhất (ETTm1 Linear H = 96 ở ×32; ETTh2 NLinear H = 720 ở ×0,125)". Ô nhiễu (mức chọn hơn mức thứ hai < 1 SE): 26 → 31/36 (lưới dày hơn nên có thêm mức sát nhau).
- IV.3, kết quả sau tinh chỉnh: số đổi qua macro: nhãn đổi so với B4 cũ 15 → 16 ô, trong đó do lr 13 → 14; khoảng Δ test ETTh1 + ETTm1 +0,0022–+0,0290 → +0,0020–+0,0290. Không đổi: 27/36 ô có ý nghĩa, 28/36 theo trung bình, 23/24 ở ETTh1 + ETTm1, "dạng đóng tốt hơn thật" 2, "SGD tốt hơn thật" 5. Bảng 7 (đếm nhãn): "val khác test" 9 → 8, "SGD khớp val quá mức" 16 → 17.
- IV.3, "Các cơ chế đã loại trừ": thêm mục "Chính quy hóa L2 mạnh hơn" (C.1, Bảng 12).
- IV.4: viết lại: "Cách phát hiện", "Vị trí" (163 đoạn ≥ 24 giờ, 63 đoạn ngày 31, 19 đoạn ≥ 96 giờ, Hình 6, Bảng 13), "Kết quả có lọc và không lọc" (Bảng 8); các gạch đầu dòng cũ giữ nguyên.
- V (thảo luận): "Linear chọn lr ×8 ở 3/4 horizon, còn DLinear chọn ×2 ở 3/4" → "Linear chọn lr từ ×8 trở lên ở 3/4 horizon, còn DLinear chọn không quá ×2 ở 3/4" (H = 96 của Linear nay ×32).
- VI: câu mở đầu "8 hạn chế của kế hoạch 3.5 (mục 2) và hai hạn chế …" → "Báo cáo có 11 hạn chế. Hạn chế 7 và 8 rút ra khi phân tích ETTh2 và số công bố; …". Hạn chế 1: thêm "chỉ là hai nguồn độc lập" (cùng máy biến áp). Hạn chế 5: "Chưa thử δ theo từng h" → kết quả C.3 (2 ô, Huber vẫn giữa; chưa kiểm toàn ma trận). Hạn chế 6: viết lại theo Phần A, bỏ "(kế hoạch 3.5 mục 4.1; không làm trong phiên cuối)"; 15 → 2 ô còn ở biên. Hạn chế 8: đổi tên "Pipeline SGD và số công bố"; thêm code gốc ETTm1 H = 96 trên CPU (0,3018 ± 0,0010, 5 seed), Phần B, môi trường, câu "không kết luận số công bố là không tái lập được".
- VII (kết luận) RQ3: "Ngoại lệ là ETTh2 …, mà chưa rõ cơ chế" → "SGD tốt hơn … chỉ xảy ra trên ETTh2 … Không giá trị λ nào của ridge … nên lợi thế này không phải một dạng chính quy hóa L2 mạnh hơn; cơ chế vẫn là câu hỏi mở"; thêm câu về số công bố (8/36 với SGD của repo; code gốc: 8/8 với `drop_last=False`, 1/8 trong phân phối với `True`).
- Phụ lục A: "đúng trên cả ba dataset" → "đúng trên ETTh1, ETTh2 và ETTm1".
- Phụ lục C: ba mục mới (họ ridge, vị trí đoạn hằng, δ theo bước); bảng lr 9 cột, 5 chữ số, bỏ cột "lr gốc".
- Phụ lục D: thêm mục "Seed"; scikit-learn; "theo đúng thứ tự các nhiệm vụ" → "theo thứ tự: …"; thời gian mở rộng lưới lr (0,8 giờ), code gốc với `drop_last` (2,6 giờ GPU), δ theo bước (3 phút).
- Đã rà "phiên", "kế hoạch", "nhiệm vụ" trong `report/sections/`: không còn.
- Mục II (Toner & Darlow): không đổi câu nào (mọi câu có nguồn, mục 5).

## 5. Đối chiếu Toner & Darlow (`docs/refs/toner2024analysis_2403.14587v2.pdf`, số trang in trên PDF)

| Câu trong báo cáo (mục II) | Nguồn |
|---|---|
| Xét tập hàm mỗi biến thể biểu diễn được (lớp mô hình) | §3, định nghĩa "model class", Definition 3.1, tr. 2 |
| DLinear cùng lớp mô hình với Linear, mọi ánh xạ affine x ↦ Ax + b | Lemma 3.2 và chứng minh, tr. 3 |
| FITS cũng vậy khi bỏ bộ lọc thông thấp và L ≥ T − 2 | Theorem 3.3, Remark về low-pass filter, tr. 3 ("M(DLinear) = M(Linear) = M(FITS) (when L ≥ T − 2 and without a low-pass filter)") |
| NLinear ≡ hồi quy tuyến tính, mỗi hàng của W tổng bằng 1, hệ số chặn tự do | Lemma 3.9, tr. 5; "NowNorm imposes the same weight matrix constraint as IN and RevIN, but does not constrain the bias", tr. 5; Table 1, tr. 5 |
| Linear + chuẩn hóa theo mẫu (RLinear, FITS+IN): cùng ràng buộc tổng hàng, hệ số chặn nhân với độ lệch chuẩn của cửa sổ | Lemma 3.5 (tr. 3–4), Lemma 3.7 (tr. 4), khung "IN and RevIN impose the constraints…" (tr. 5), Table 1 (tr. 5) |
| Vì bình phương tối thiểu lồi, mỗi lớp có nghiệm dạng đóng | §4 Discussion "solvable in closed form owing to convexity" (tr. 5); "Closed Form" (tr. 6) |
| 8 bộ dữ liệu, 4 horizon; nghiệm dạng đóng cho MSE test thấp hơn ở 23/32 cấu hình | §5 "8 benchmarking datasets" (tr. 6); §5.2 và Table 2: "OLS solutions were superior across 23 of 32 (72%) settings" (tr. 8) |
| "Kết luận của họ về DLinear và NLinear khớp với RQ1…", "Bài này mở rộng ở ba điểm…" | khẳng định của chúng tôi, không trích từ PDF |

Không có câu nào cần xóa hay sửa.

## 6. TODO còn lại, PDF, kiểm số

- **TODO metadata:** Toner & Darlow (ICML 2024, PMLR): số tập và số trang chưa có (PDF là bản arXiv v2, chỉ ghi "Preprint"; đầu trang "Submission and Formatting Instructions for ICML 2024" là phần giữ chỗ của template, không xác nhận nơi đăng). Nơi đăng ICML 2024 theo yêu cầu D.1.2 của nhiệm vụ. Holland & Welsch (1977): số tập, số trang (TODO có từ nhiệm vụ 8).
- **PDF:** `report/build/main.pdf`, 40 trang (trước: 35). Biên dịch không lỗi, không cảnh báo tham chiếu hay trích dẫn chưa xác định; còn 2 overfull box nhỏ (2,6pt ở dòng mục lục "VII.", 3,6pt ở một đoạn của IV.3).
- **`audit_numbers.py`:** 36 số thập phân ngoài macro, cần rà tay 0 (đều là hằng số thiết kế hoặc số phiên bản; thêm ×0,25, ×0,125 và scikit-learn vào danh sách). `check_latex.py`: không lỗi tĩnh, 0 `\TODO`. 678 macro số (trước 623).
- **`run_all.sh quick`** (`VENV=.venv-gpu SKIP_CELL=1`): mã thoát 0, 47 giây; 66 test đạt; latexmk build sạch 15 giây. Bước 3 (chạy lại grid ETTh1 H = 96) không chạy lại trong phiên này.

## 7. Ghi chú về commit trong kết quả

Commit trong JSON là HEAD lúc ghi file, không phải lúc khởi chạy. Các phép chạy dài khởi chạy trên commit sạch (`33522d7` cho A, `d3c31e2` cho B, `357b4a0` cho C.3), nhưng trong lúc chạy repo có thay đổi chưa commit, nên phần lớn JSON mới ghi hậu tố `-dirty`: `sgd_lr_ext` 141/156 JSON, `official_ltsf_droplast` 84/90; `huber_delta_h` sạch (`357b4a0`). Các thay đổi chưa commit lúc đó chỉ ở `scripts/report/`, `report/`, `tests/`, `src/solvers.py` (irls, không dùng trong SGD hay code gốc) và `scripts/sgd/sgd_lr_extend.py` (driver). `git diff <commit ghi trong JSON> HEAD` trên `src/sgd.py`, `src/models.py`, `src/data.py`, `src/operators.py`, `scripts/sgd/sgd_etth2.py`, `scripts/checks/bench_5060ti.py`, `scripts/grid/runner.py` rỗng với mọi commit đã ghi (`33522d7`, `d3c31e2`, `357b4a0`, `a196c2b`, `4cc0054`); `scripts/sgd/official_ltsf.py` và `patch_droplast.diff` không đổi từ `d3c31e2`; `src/solvers.py` và `huber_delta_h.py` không đổi từ `357b4a0`.

## 8. Tái lập

```bash
python scripts/sgd/sgd_lr_extend.py --workers 5 --threads 2                    # A (CPU)
python scripts/sgd/official_ltsf.py --prepare                                  # nếu chưa có scratch/ltsf_official
python scripts/sgd/official_ltsf.py --dataset ETTh2 --model NLinear --H 336 --seed 2021 2022 2023 2024 2025 \
    --test-drop-last 1 --out official_ltsf_droplast/ETTh2                      # B: mỗi ô × {1, 0} (danh sách ô: run_all.sh full, bước F)
python scripts/checks/huber_delta_h.py                                         # C.3 (GPU)
python scripts/report/report_nv9.py                                            # A, B, C.1–C.3 → results/nv9/
python scripts/report/make_numbers.py && python scripts/report/make_tables.py && python scripts/report/make_figures.py
cd report && latexmk main.tex
```
