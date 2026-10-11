# Nhiệm vụ 9 cho Claude Code: áp dụng góp ý Tuần 3 của TA và hoàn thiện báo cáo

*Lập ngày 14/10/2026, sau bản `report/build/main.pdf` (14/10/2026) và câu trả lời Tuần 3 của TA (`Answer_Week03.pdf`, mục V). Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_9.md và làm theo".*

Nhiệm vụ gồm **ba phép chạy nhỏ** (Phần A, B, C.3) và **sửa báo cáo** (Phần D). Ngoài ba phép chạy này, không chạy thí nghiệm mới.

## 0. Góp ý của TA và việc tương ứng

| # | TA nói | Việc | Phần |
|---|---|---|---|
| 1 | lr tốt nhất ở biên khoảng thử là chưa tinh chỉnh xong; nên mở rộng (×16, ×32 cho ETTm1), 2–3 giờ là đáng | Mở rộng lưới lr ở mọi ô biên | A |
| 2 | Viết mềm hơn về số công bố, kèm môi trường; kiểm `drop_last=True` ở test loader của code gốc, một lỗi đã biết của dòng codebase LTSF có thể làm số công bố tốt hơn thực tế | Chạy code gốc với `drop_last=True`; viết lại đoạn B1 | B |
| 3 | ETTh2: báo hiện tượng như câu hỏi mở là được; thử ridge với vài λ | Đọc từ `lambda_path.csv` (không chạy lại) | C.1 |
| 4 | Đoạn cảm biến bị kẹt: nói rõ phát hiện bằng cách nào, nằm ở đâu; đặt kết quả có lọc và không lọc cạnh nhau | Bảng và mô tả mới | C.2 |
| 5 | Một δ chung chấp nhận được nếu ghi rõ; gọn hơn là chuẩn hóa phần dư theo σ̂_h rồi dùng δ chung; ablation trên 1–2 ô | Ablation δ theo từng h | C.3 |
| 6 | `run_all.sh` hai chế độ là tốt; README cần phiên bản môi trường, seed, thời gian từng chế độ, kết quả mong đợi của ô kiểm | Bổ sung README | E |
| 7 | (Trả lời nhóm khác, áp dụng được) ETTh và ETTm là cùng một máy biến áp, nên không tính là nguồn độc lập | Sửa cách viết các khẳng định | D |
| 8 | (Trả lời nhóm khác, áp dụng được) Caption của bảng đặt phía trên, caption của hình phía dưới; mọi hình, bảng được nhắc trước khi xuất hiện | Sửa bố cục | D |

**Thứ tự:** C.1, C.2 (chỉ đọc dữ liệu) → A trên CPU, chạy song song với B và C.3 → D → E → F.

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của các nhiệm vụ trước:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi kết quả cũ trong `results/`; kết quả mới ghi vào thư mục mới.
- **Tập test không bao giờ dùng để chọn bất cứ thứ gì.** Riêng C.1 có một con số "ridge tốt nhất theo test", và nó phải được gọi rõ là **mức tham chiếu không hợp lệ để chọn mô hình**, chỉ dùng để chứng minh giới hạn của họ ridge.
- Mọi con số trong báo cáo đi qua `numbers.tex` và các bảng tự sinh; chạy `audit_numbers.py` sau khi sửa.
- Commit sạch trước mỗi lần chạy dài; ghi phần cứng và commit vào mọi JSON.

---

## 2. Phần A: mở rộng lưới learning rate

### A.1. Phạm vi

**Mọi ô đã chọn mức ở biên khoảng thử (15/36)**, không chỉ ETTm1. TA nhấn mạnh việc dùng cùng ngân sách tinh chỉnh cho mọi cấu hình, nên các ô biên ở mọi dataset đều được mở rộng theo cùng một quy tắc.

| Biên | Ô | Mở rộng |
|---|---|---|
| trên (×8) | ETTm1: Linear H = 96, 192, 336; NLinear H = 96, 720. ETTh1: Linear H = 192; DLinear H = 720 | ×16, rồi ×32 |
| dưới (×0,5) | ETTh1: Linear H = 720; DLinear H = 336. ETTh2: Linear H = 720; DLinear H = 192; NLinear H = 96, 192, 336, 720 | ×0,25, rồi ×0,125 |

Kiểm lại danh sách này với `results/sgd_lr/selection.json` trước khi chạy (phải đúng 15 ô).

### A.2. Quy tắc

1. Với mỗi ô, chạy mức kế tiếp theo hướng biên với 5 seed chọn lr (2021–2025).
2. Nếu mức mới tốt hơn trên MSE val trung bình, thử thêm một mức nữa (tối đa 2 mức mới mỗi ô). Dừng ngay khi mức mới không tốt hơn.
3. Nếu lr được chọn đổi, chạy 15 seed cuối (2026–2040) ở lr mới. Nếu không đổi, giữ kết quả cũ.
4. Nếu sau 2 mức mới vẫn chọn ở biên, dừng, ghi ô đó là "còn ở biên" vào hạn chế.
5. Có lr lớn sẽ làm SGD phân kỳ (NaN, hoặc MSE val vượt 10 lần mức ×1). Coi những lần chạy đó là thất bại của mức lr đó, giữ trong JSON, và không chọn mức đó.

### A.3. Ngân sách

4 giờ CPU thời gian thực, chạy song song như nhiệm vụ 7. ETTm1 đắt nhất nên chạy trước. Nếu vượt, chỉ làm một mức mới cho các ô ETTh2 và ghi rõ.

### A.4. Cập nhật

- Sinh lại: Bảng 3 (DLinear − Linear, lr tinh chỉnh), Bảng 6 (đếm nhãn), Bảng 8 (B4 đầy đủ), Bảng 9 (MSE val theo lr; thêm cột ×0,125, ×0,25, ×16, ×32), cùng mọi macro liên quan.
- Ghi **danh sách thay đổi:** ô nào đổi lr, Δ test cũ → mới, nhãn cũ → mới, số ô "SGD kém có ý nghĩa" cũ → mới, và kết quả C.3 của DLinear so với Linear.

---

## 3. Phần B: `drop_last` trong code gốc

### B.1. Tìm bằng chứng trong lịch sử repo gốc

Trong `third_party/LTSF-Linear`, tìm lịch sử của cấu hình test loader (thường ở `data_provider/data_factory.py`): tìm các commit mà `flag == 'test'` đi kèm `drop_last=True`, và commit đổi sang `False`. Ghi hash, ngày, dòng code. Thêm một câu: bài AAAI 2023 được nộp và công bố vào khoảng thời gian nào so với các commit đó. Chỉ ghi những gì đọc được trong git log, không suy đoán.

### B.2. Chạy

- Dùng bản chép code gốc ở `scratch/ltsf_official/` (cùng bản vá seed của nhiệm vụ 4). Thêm **một** thay đổi: tham số dòng lệnh chọn `drop_last` cho test loader. Ghi diff vào `results/official_ltsf/patch_droplast.diff`.
- Các ô: **8 ô B1 có số công bố thấp hơn mọi seed** (DLinear: ETTh1 H = 192, 720; ETTm1 H = 96, 192, 336. NLinear: ETTh2 H = 336, 720; ETTm1 H = 336), cộng ETTh2 H = 720 DLinear (đã kiểm ở nhiệm vụ 4) để đối chứng.
- Mỗi ô: seed 2021–2025, chạy với `drop_last=True` và `drop_last=False`, cùng siêu tham số của script gốc.
- Ghi: MSE, MAE test do code gốc in ra; **số cửa sổ test bị bỏ** khi `drop_last=True` (tính từ kích thước tập test và batch size); epoch dừng.

### B.3. Đọc kết quả

Với mỗi ô, so ba thứ: số công bố, phân phối 5 seed với `drop_last=True`, phân phối 5 seed với `drop_last=False`.

- Nếu với `drop_last=True` số công bố nằm trong [min, max] của các seed, ở phần lớn trong 8 ô: `drop_last` giải thích được khoảng chênh của B1.
- Nếu không: `drop_last` không đủ giải thích; ghi đúng như vậy.
- Ghi thêm mức chênh do `drop_last` (cùng seed, True − False) ở từng ô.

### B.4. Viết lại đoạn B1 trong báo cáo

Theo cách viết TA đề nghị, ví dụ: "Chúng tôi không đạt được số công bố ở 8/36 ô với 20 seed, kể cả khi chạy code gốc ở N ô (môi trường: …)", rồi một câu về kết quả `drop_last`. Bỏ mọi câu khẳng định "số công bố không tái lập được".

---

## 4. Phần C: ba phân tích nhỏ

### C.1. Họ ridge trên ETTh2 (chỉ đọc `lambda_path.csv`, không chạy lại)

Với Linear và DLinear, mục tiêu MSE, ETTh2, 4 H, trên **toàn bộ lưới 142 giá trị λ**, ghi:

- MSE val và MSE test ở λ = 0 và ở λ*_MSE;
- **MSE test nhỏ nhất trên toàn lưới** và λ tương ứng (mức tham chiếu chọn theo test, không hợp lệ để chọn mô hình);
- MSE test trung bình của SGD (lr gốc, 20 seed; và lr tinh chỉnh).

Kết luận cần trả lời: ở bao nhiêu ô, **kể cả ridge tốt nhất theo test** vẫn kém SGD trung bình. Nếu ở mọi ô ridge vẫn kém, viết vào mục "Các cơ chế đã loại trừ": không giá trị λ nào của ridge đạt được mức của SGD trên ETTh2, nên lợi thế của SGD không phải một dạng chính quy hóa L2 mạnh hơn. Đây là câu trả lời cho gợi ý của TA.

### C.2. Đoạn cảm biến bị kẹt: mô tả và bảng đặt cạnh nhau

Từ `scripts/checks/constant_windows.py` và các kết quả đã có, **không chạy lại**:

- **Cách phát hiện:** nêu đúng tiêu chí đã dùng (cửa sổ một kênh, sau chuẩn hóa, có max − min của x hoặc y nhỏ hơn 1e-12).
- **Vị trí:** với mỗi kênh của ETTh2, liệt kê các đoạn chuỗi hằng dài ít nhất 24 giờ: thời điểm bắt đầu, kết thúc, độ dài, giá trị, và đoạn đó thuộc train, val hay test. Thêm một hình nhỏ: chuỗi ba kênh tải (MUFL, LUFL, LULL) theo thời gian, tô đậm các đoạn hằng và vạch ranh giới train/val/test.
- **Bảng đặt cạnh nhau** (Linear, DLinear, NLinear × 4 H), mỗi ô gồm "gốc" và "đã lọc" cho: MSE test của dạng đóng λ*, SGD trung bình 20 seed, MAE λ = 0, và lợi thế của MAE so với MSE.

### C.3. Ablation δ theo từng bước h

- **Cách làm (theo TA):** chuẩn hóa phần dư theo σ̂_h của từng bước, rồi dùng một δ chung trên thang đã chuẩn hóa. Tương đương: δ_h = 1,345 · σ̂_h, với σ̂_h = 1,4826 · MAD của phần dư train của bước h (nghiệm MSE λ = 0).
- **Code:** cho phép `irls` nhận `delta` là vectơ độ dài H, giữ quy ước phạt Σ_h Σ_i ρ_{δ_h}(r_ih)/δ_h + λ Σ_h w_hᵀ Pen w_h. Thêm test:
  - `delta` vô hướng cho kết quả trùng từng bit với trước khi sửa;
  - vectơ có mọi phần tử bằng nhau cho kết quả trùng với vô hướng tới 1e-12;
  - với δ_h lớn ở mọi h, kết quả về đúng ridge MSE với quy đổi λ của từng h.
- **Ô:** ETTh2 H = 720 Linear (nơi δ nhạy nhất) và ETTh1 H = 96 Linear. Đường λ đầy đủ như grid, chế độ pha, `tol = 1e-9`.
- **Bảng:** MSE test ở λ*_MSE và MAE test ở λ*_MAE, cho δ_MAD chung và δ_h theo bước, đặt cạnh MSE và MAE của cùng ô. Kết luận: δ theo bước có đổi vị trí của Huber không.
- Ngân sách GPU: 1 giờ.

---

## 5. Phần D: sửa báo cáo

### D.1. Lỗi hiển thị

1. **Danh mục tài liệu tham khảo:** các chuỗi "andothers", "inProceedings", "jourvol", "number", "pages" đang hiện nguyên văn, do biblatex không có gói ngôn ngữ tiếng Việt. Đặt ngôn ngữ của danh mục là tiếng Anh (ví dụ `language=english` cho biblatex, hoặc bọc `\printbibliography` trong môi trường tiếng Anh), hoặc định nghĩa đủ các chuỗi. Kiểm lại mọi mục hiển thị đúng tên tác giả (không còn "andothers").
2. **Toner & Darlow:** đổi trích dẫn arXiv sang bản **ICML 2024** (Proceedings of Machine Learning Research). Lấy metadata (tên đầy đủ hai tác giả, tập, trang) từ chính PDF trong `docs/refs/`. Phần nào không đọc được thì đánh dấu `% TODO: kiểm metadata` và liệt kê trong báo cáo cuối phiên.
3. **Bảng 8 tràn lề:** cột nhãn bị cắt. Đặt bảng này trên trang ngang, hoặc thay nhãn bằng mã ngắn (ví dụ T1–T6) kèm chú giải trong caption.
4. **Tham chiếu chéo có hai dấu chấm** ("mục IV.3..", "mục V..", "hạn chế 5.)"): sửa định nghĩa số mục hoặc lệnh tham chiếu trong preamble cho hiển thị một dấu chấm, không sửa tay từng chỗ. Kiểm trên toàn PDF.
5. **Caption:** bỏ các mã nội bộ "(B1)", "(B2)", "(B3, bản gọn)", "(H1)"…"(H5)" khỏi caption. Báo cáo chỉ dùng số Bảng và Hình. Đổi các chỗ trong văn bản nhắc tới B1–B4, H1–H5 sang số Bảng, Hình tương ứng.
6. **Vị trí caption:** caption bảng đặt **phía trên** bảng, caption hình phía dưới, nếu template cho phép. Nếu template cố định vị trí, giữ template và ghi lại.
7. **Dấu thập phân trong hình:** dùng dấu phẩy cho thống nhất với văn bản, bằng cách đặt bộ định dạng số trục trong `make_figures.py`.
8. Kiểm mọi hình, bảng đều được nhắc trong văn bản **trước** khi xuất hiện.

### D.2. Nội dung

1. **Bỏ ngôn ngữ quy trình nội bộ.** Tìm và viết lại mọi chỗ có "phiên", "kế hoạch 3.5", "nhiệm vụ", "các phiên trước". Ví dụ: "δ = 1 của các phiên trước" → "δ = 1 (mặc định của `HuberLoss` trong PyTorch)".
2. **Tóm tắt, mục (3):** "Ngoại lệ chỉ có trên ETTh2" → "SGD tốt hơn nghiệm tối ưu chỉ xảy ra trên ETTh2". Số ô theo kết quả mới của Phần A.
3. **Hai nguồn dữ liệu độc lập.** ETTh1 và ETTm1 đo cùng một máy biến áp, chỉ khác tần suất lấy mẫu. Thêm điều này vào hạn chế 1, và rà mọi câu dạng "trên cả ba dataset" thành "trên ETTh1, ETTh2 và ETTm1 (hai máy biến áp)" hoặc tương đương. Không đổi con số.
4. **Mục VI:** sửa câu mở đầu cho khớp số hạn chế thực tế, bỏ nhắc tới "kế hoạch". Cập nhật:
   - hạn chế 5: kết quả C.3, cùng cách chuẩn hóa theo σ̂_h của TA;
   - hạn chế 6: kết quả Phần A (ô nào còn ở biên);
   - hạn chế 8: bổ sung lần chạy code gốc ở ETTm1 H = 96 (0,3018 ± 0,0010, nhiệm vụ 6 D.1) và kết quả Phần B.
5. **Mục IV.3:** viết lại đoạn B1 theo B.4. Thêm kết quả C.1 vào "Các cơ chế đã loại trừ".
6. **Mục IV.4:** thêm tiêu chí phát hiện, vị trí, hình và bảng C.2.
7. **Mục IV.2:** thêm một câu và một bảng nhỏ (hoặc vào Phụ lục C) cho ablation C.3.
8. **Phần Công trình liên quan:** đối chiếu **từng câu** về Toner & Darlow với PDF. Ghi số trang hoặc mục nguồn của mỗi khẳng định vào báo cáo cuối phiên (không ghi vào báo cáo chính). Câu nào không tìm được nguồn trong PDF thì xóa hoặc sửa cho đúng.
9. **Kết luận và tóm tắt:** cập nhật theo Phần A, B, C. Nếu Phần A làm đổi kết luận RQ3, xem điểm dừng ở mục 8.

### D.3. Biên dịch và rà

- Biên dịch không lỗi, không còn cảnh báo tham chiếu chưa xác định, không còn overfull box lớn.
- Chạy `audit_numbers.py`; mọi số mới đi qua macro.
- Rà chính tả và thuật ngữ theo bảng thuật ngữ.

---

## 6. Phần E: README

Bổ sung theo góp ý số 6 của TA:

- **Phiên bản môi trường:** Python, torch (bản cu130 và CPU), numpy, scipy, pandas, matplotlib; GPU và driver đã dùng.
- **Seed:** danh sách seed của SGD (gốc, chọn lr, chạy cuối) và seed vị trí outlier.
- **Thời gian chạy dự kiến** của `quick` và `full`, theo từng bước, kèm phần cứng đã đo.
- **Kết quả mong đợi của ô kiểm** trong chế độ `quick`: bảng nhỏ các giá trị của ETTh1 H = 96 (MSE, MAE test ở λ = 0 và λ* cho 3 mô hình × 3 mục tiêu) và ngưỡng so sánh, để người khác đối chiếu.

## 7. Phần F: báo cáo cuối phiên

Viết `docs/reports/BAO_CAO_NV9.md`:

1. Kết quả Phần A: bảng lr cũ → mới, Δ và nhãn cũ → mới, các ô còn ở biên.
2. Kết quả Phần B: bằng chứng git; bảng ba cột (công bố, `drop_last=True`, `drop_last=False`) cho 9 ô; kết luận.
3. Kết quả C.1, C.2, C.3.
4. **Danh sách mọi câu và số trong báo cáo đã đổi**, theo mục: cũ → mới, lý do.
5. Đối chiếu Toner & Darlow: mỗi khẳng định kèm trang hoặc mục nguồn.
6. Các `TODO` còn lại; số trang PDF; kết quả `audit_numbers.py`.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 8. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt, kể cả các test mới của δ theo h.
- **Phần A làm đổi kết luận RQ3:** số ô ETTh1, ETTm1 mà SGD kém có ý nghĩa giảm từ 23/24 xuống dưới 20/24, hoặc xuất hiện ô ngoài ETTh2 có nhãn "SGD tốt hơn thật". Báo cáo số liệu trước khi viết lại kết luận.
- **Phần B:** code gốc với `drop_last=True` đạt số công bố ở từ 5/8 ô trở lên. Báo cáo trước khi viết lại đoạn B1 và hạn chế 8.
- Code gốc cần sửa ngoài phạm vi cho phép ở B.2.
- Một khẳng định về Toner & Darlow trong báo cáo không tìm thấy trong PDF và không rõ nên sửa thế nào.
- Vượt ngân sách (A: 4 giờ CPU; C.3: 1 giờ GPU).

## 9. Không làm trong phiên này

- Không thêm dataset, mô hình, mục tiêu hay độ lớn outlier mới.
- Không tinh chỉnh tham số nào của SGD ngoài lr.
- Không mở thêm thí nghiệm về cơ chế ETTh2 ngoài C.1.
- Không sửa các báo cáo kỹ thuật cũ trong `docs/reports/`.
- Không công khai repo.
