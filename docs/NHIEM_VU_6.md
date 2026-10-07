# Nhiệm vụ 6 cho Claude Code: thí nghiệm outlier cho RQ2 (và hai phép kiểm tùy chọn)

*Lập ngày 07/10/2026, sau `docs/BAO_CAO_SGD_B1_B4.md`. Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_6.md và làm theo".*

## 0. Bối cảnh

Outline yêu cầu: "Chèn outlier nhân tạo vào train ở tỷ lệ 0/1/5/10% (seed cố định), đo MSE/MAE test của ba mục tiêu." Đây là thí nghiệm bắt buộc cuối cùng của đề tài.

Đã biết từ grid và các nhiệm vụ trước:

- Trên dữ liệu gốc, huấn luyện bằng MAE thường cho MSE test tốt hơn huấn luyện bằng MSE với Linear, nhất là trên ETTh2. Với NLinear, lợi thế đó gần như biến mất.
- Trên ETTh2, khoảng một nửa lợi thế của MAE với Linear đến từ các đoạn cảm biến bị kẹt (`BAO_CAO_ETTH2_2.md`, Phần C). Đây là một dạng "outlier tự nhiên".
- DLinear ≡ Linear khi λ = 0 với mọi mục tiêu, và gần như trùng ở λ*. Vì vậy thí nghiệm này **không chạy DLinear**.

Phiên này gồm bốn phần:

| Phần | Nội dung | Bắt buộc |
|---|---|---|
| A | Hàm chèn outlier, kèm test | có |
| B | Thí nghiệm chính: chèn vào chuỗi gốc | có |
| C | Thí nghiệm phụ: chèn chỉ vào đích | có, phạm vi nhỏ |
| D | Hai phép kiểm SGD tùy chọn, chạy trên CPU song song với B | không |
| E | Báo cáo | có |

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của các nhiệm vụ trước:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi toán của `irls`.
- Mọi số liệu có file tái lập; ghi phần cứng và git commit vào mọi JSON và mọi dòng CSV; commit sạch trước mỗi lần chạy dài.
- **Tập test không bao giờ được dùng để chọn bất cứ thứ gì.** λ* chọn trên val.
- Không ghi đè `results/lambda_path.csv`, `results/results.csv`, hay kết quả của các nhiệm vụ trước. Kết quả của nhiệm vụ này ghi vào `results/outliers/`.
- Cấu hình bộ giải giữ đúng như grid: chế độ pha, `chunk = 16`, cấu hình dừng (a′) cho MAE, `tol = 1e-9` cho Huber, MSE dạng đóng, lưới λ như grid (MSE 141 giá trị; MAE và Huber 10 giá trị, tự mở rộng ở đầu mút), warm start theo λ tăng dần, fallback float64 theo từng h.

## 2. Phần A: hàm chèn outlier

### A.1. Thiết kế (phương án mặc định của câu 11.4)

Viết `src/outliers.py`, hàm `contaminate_series(train_series, rate, magnitude=10.0, seed, protect_last=L)`:

- **Đầu vào:** chuỗi train **đã chuẩn hóa** (theo thống kê của train sạch), kích thước [T_train, 7].
- **Vùng được chèn:** chỉ các hàng **không xuất hiện trong bất kỳ cửa sổ val hay test nào**. Theo cách chia của LTSF-Linear, cửa sổ val bắt đầu lùi L hàng vào cuối train, nên L hàng cuối của train là vùng cấm. Kiểm điều này bằng code, không chỉ bằng lập luận.
- **Cách chèn:** chọn ngẫu nhiên đúng ⌊rate × (số điểm trong vùng được chèn)⌋ điểm (hàng, kênh), không lặp, độc lập giữa các kênh. Cộng vào mỗi điểm ±`magnitude` (dấu ngẫu nhiên, xác suất 1/2), tức ±10 độ lệch chuẩn của kênh đó.
- **Scaler:** giữ thống kê của train **sạch**. Không tính lại scaler sau khi chèn. Ghi rõ lựa chọn này trong báo cáo.
- **Cửa sổ:** tạo cửa sổ train từ chuỗi đã chèn bằng đúng hàm của `src/data.py`. Một điểm outlier vì vậy xuất hiện ở nhiều cửa sổ, cả trong x lẫn y. Đây là chủ ý, để mô phỏng outlier thực tế.
- Trả thêm thông tin: số điểm đã chèn, và tỷ lệ cửa sổ train có ít nhất một outlier trong x, trong y.

### A.2. Test (`tests/test_outliers.py`)

- Số điểm chèn đúng bằng ⌊rate × số điểm hợp lệ⌋.
- Không điểm nào nằm trong vùng cấm.
- **Val và test giống hệt từng byte** so với khi không chèn: các cửa sổ x, y của val và test.
- Cùng seed thì ra cùng kết quả; seed khác thì vị trí khác.
- rate = 0 trả lại đúng dữ liệu gốc.
- Mọi điểm bị chèn có |giá trị mới − giá trị cũ| = `magnitude`.

Chạy toàn bộ test; kỳ vọng: số test hiện có cộng các test mới, tất cả đạt.

## 3. Phần B: thí nghiệm chính (chèn vào chuỗi gốc)

### B.1. Phạm vi

| Trục | Mức |
|---|---|
| Mô hình | Linear (`pen = I`), NLinear (`constrained=True`, `pen = diag(1, …, 1, 0)`) |
| Mục tiêu | MSE, MAE, Huber (δ = 1) |
| Dataset | ETTh1 (sạch, không có đoạn cảm biến bị kẹt đáng kể) |
| Horizon | 96, 192, 336, 720 |
| Tỷ lệ outlier | 1%, 5%, 10% (0% lấy từ `lambda_path.csv`) |
| Seed chèn | 2026 (cố định) |
| λ | cả đường λ như grid; chọn λ* trên **val sạch**, hai cách (`val_objective`, `val_mse`) |

### B.2. Kiểm trước

Chạy runner với rate = 0 cho **Linear, ETTh1, H = 96, MSE và MAE**. Mọi dòng phải trùng `lambda_path.csv` tới 1e-10 với MSE, và tới 1e-8 với MAE. Nếu không trùng, dừng lại theo mục 7.

### B.3. Thời gian

- Dùng `runner.py --dry-run` (thêm tham số `--outlier-rate`, `--outlier-seed`, `--out`) để ước lượng. Grid ETTh1 đầy đủ (3 mô hình) mất khoảng 3 giờ, nên dự kiến khoảng 2 giờ cho mỗi tỷ lệ, tổng khoảng 6 giờ. Outlier có thể làm IRLS hội tụ nhanh hoặc chậm hơn, nên đo thật một đường λ trước khi chạy hết.
- **Ngân sách: 10 giờ GPU.** Nếu ước lượng vượt, giảm theo thứ tự:
  1. bỏ H = 192 và 336, giữ H ∈ {96, 720};
  2. bỏ tỷ lệ 5%.

  Ghi rõ đã cắt gì.
- Chạy theo thứ tự rate 10% → 1% → 5%, để có trước hai mức xa nhau nhất.

### B.4. Ghi kết quả

`results/outliers/lambda_path.csv` có cùng cột như `lambda_path.csv`, thêm `outlier_mode` (`series`), `outlier_rate`, `outlier_seed`, `n_contaminated`, `frac_windows_x`, `frac_windows_y`. Sinh `results/outliers/results.csv` bằng `select_lambda.py` (thêm tham số đường dẫn vào và ra nếu cần).

### B.5. Độ nhạy theo seed chèn

Một outlier đặt ở vị trí khác có thể cho kết quả khác. Chạy thêm **Linear, ETTh1, H = 96**, ba mục tiêu, rate ∈ {1%, 10%}, seed chèn 2027 và 2028. Báo cáo độ lệch giữa ba seed (2026–2028). Nếu độ lệch này lớn hơn khác biệt giữa các mục tiêu, ghi rõ trong kết luận.

## 4. Phần C: thí nghiệm phụ (chèn chỉ vào đích)

Đây là phiên bản "sạch" theo lý thuyết hồi quy bền vững: chỉ nhiễu y, x giữ nguyên. Nó giúp phân biệt MAE bền vững với outlier ở y hay ở x.

- Trên các **cửa sổ** train (không phải chuỗi): chọn ngẫu nhiên đúng ⌊rate × (số phần tử y của train)⌋ phần tử (cửa sổ, kênh, bước h), cộng ±10. x giữ nguyên.
- Viết hàm `contaminate_targets` cạnh `contaminate_series`, kèm test tương tự A.2.
- **Phạm vi:** Linear và NLinear, ETTh1, **H ∈ {96, 720}**, ba mục tiêu, rate ∈ {1%, 10%}, seed 2026.
- Ghi cùng file với B, `outlier_mode = target`.

## 5. Phần D: hai phép kiểm SGD tùy chọn

Chỉ làm khi B đã chạy ổn trên GPU. Chạy trên **CPU**, song song với B. Mỗi phép kiểm giới hạn 60 phút thời gian thực. Nếu B gặp sự cố, bỏ D.

### D.1. Code gốc trên ETTm1

Tám ô ở bảng B1 có số công bố thấp hơn mọi seed của repo, trong đó 5 ô là DLinear và 3 ô là DLinear ETTm1. Chạy code gốc của LTSF-Linear (bản chép ở `scratch/ltsf_official/`, cùng bản vá đã dùng ở nhiệm vụ 4) cho **DLinear, ETTm1, H = 96**, seed 2021–2025. So MSE test do code gốc in ra với phân phối 20 seed của repo (0,3017 ± 0,0018, khoảng [0,3003; 0,3070]) và với số công bố 0,299.

- Nếu code gốc cũng cho khoảng 0,300–0,307: số công bố không tái lập được, giống trường hợp ETTh2.
- Nếu code gốc cho dưới 0,300 ở phần lớn seed: pipeline của repo có khác biệt với DLinear trên ETTm1. Báo cáo và **không** tự sửa `sgd_etth2.py`.

### D.2. Linear với learning rate gấp đôi trên ETTm1

**Giả thuyết:** Adam chuẩn hóa bước cập nhật cho từng tham số. DLinear có hai nhánh trọng số cùng đóng góp vào W_eff, nên mỗi bước W_eff có thể dịch xa hơn, giống Linear với learning rate lớn hơn. Trên ETTm1, lr = 0,0001 rất nhỏ, và SGD vẫn kém nghiệm tối ưu, nên learning rate hiệu dụng lớn hơn có thể có ích.

- Chạy **Linear, ETTm1**, 4 H, seed 2021–2040, lr = 0,0002 (gấp đôi script gốc), mọi tham số khác giữ nguyên.
- So với DLinear (lr gốc) và Linear (lr gốc) đã có, bằng CI Welch trên val và test.
- **Đọc kết quả:** nếu Linear với lr gấp đôi không còn kém DLinear một cách có ý nghĩa ở phần lớn H, thì giả thuyết được ủng hộ. Nếu còn kém, thì giả thuyết không đủ. Lưu ý: chọn lr gấp đôi là chọn trước, không tinh chỉnh theo kết quả.
- Nếu thời gian cho phép, chạy thêm lr = 0,0004 để xem xu hướng. Ghi rõ đây là lần thử thêm.

## 6. Phần E: báo cáo

Viết `docs/BAO_CAO_OUTLIER.md`, cùng văn phong các báo cáo trước:

1. **Kết luận** ở đầu:
   - Mục tiêu nào bền vững nhất khi có outlier, và mức suy giảm MSE/MAE test từ 0% đến 10% của từng mục tiêu.
   - Chèn vào chuỗi gốc khác chèn chỉ vào đích thế nào.
   - Linear khác NLinear thế nào.
   - Độ nhạy theo seed chèn.
   - Kết quả D (nếu làm).
2. Thiết kế chèn outlier (A.1), kèm số điểm và tỷ lệ cửa sổ bị ảnh hưởng ở từng rate.
3. **Bảng chính** cho mỗi (mô hình, H, mục tiêu, rate): λ* (theo `val_objective`), MSE và MAE test, và mức suy giảm so với rate 0.
4. **Hình H2** (`results/outliers/H2_outlier_robustness.png`): MSE test (và một bản MAE test) theo rate. Mỗi ô một (mô hình, H), ba đường cho ba mục tiêu, nét liền cho chèn vào chuỗi gốc, nét đứt cho chèn chỉ vào đích (ở các H có chạy). Ghi đủ nhãn trục và chú thích.
5. Bảng độ nhạy theo seed (B.5), và bảng D (nếu có).
6. **Một đoạn nháp khoảng 200–250 từ** cho mục RQ2 của báo cáo cuối. Đoạn này gộp ba nguồn: grid trên dữ liệu gốc (`results/results.csv`), thí nghiệm outlier, và kết quả đoạn cảm biến bị kẹt trên ETTh2 (`BAO_CAO_ETTH2_2.md`, Phần C). Chỉ nêu những gì đã kiểm.
7. **Tái lập:** các lệnh đã chạy.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 7. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt.
- Hàm chèn làm thay đổi bất kỳ cửa sổ val hay test nào.
- B.2: rate = 0 không trùng `lambda_path.csv` trong ngưỡng.
- `irls` báo `RuntimeError: J tăng`, hoặc một đường λ có `converged = False` ở quá 2 giá trị λ.
- Ước lượng Phần B vượt 10 giờ kể cả sau khi cắt theo B.3.
- D.1: code gốc cho dưới 0,300 ở phần lớn seed.

## 8. Không làm trong phiên này

- Không chạy DLinear trong thí nghiệm outlier.
- Không chạy outlier trên ETTh2 hay ETTm1.
- Không đổi δ của Huber, `MAE_DELTA`, cấu hình dừng, hay lưới λ.
- Không sửa kế hoạch, checklist, hay các báo cáo cũ.
