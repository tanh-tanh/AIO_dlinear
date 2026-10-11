# Kế hoạch Topic Team 4 tuần (bản 3.5) — chuyển sang giai đoạn viết báo cáo

*Ngày 08/10/2026. Bản 3.4 (30/09) vẫn là tài liệu thiết kế chi tiết: dẫn xuất ba dự đoán (Mục 1), IRLS (Mục 3), ma trận thí nghiệm (Mục 4). Bản 3.5 không lặp lại các phần đó. Nó chốt kết quả cuối của cả ba câu hỏi nghiên cứu sau Nhiệm vụ 3–7, liệt kê nguồn của từng con số, và đặt kế hoạch cho phần còn lại: viết báo cáo, đóng gói repo.*

**Trạng thái: thí nghiệm đã đóng băng.** Ngoại lệ duy nhất là việc tùy chọn ở mục 4.1.

---

## 0. Những gì đã hoàn thành từ bản 3.4

| Nhiệm vụ | Nội dung | Báo cáo |
|---|---|---|
| Grid | 216 ô × 2 cách chọn λ*, mọi dòng hội tụ | Báo cáo tổng hợp grid |
| 3 | SGD trên ETTh2; giả thuyết co về W₀; lưới λ MSE dày (141 giá trị); đoạn hằng ETTh2 | `BAO_CAO_ETTH2_SGD.md` |
| 4 | SGD 20 seed trên ETTh2; SGD trên train đã lọc; MAE trên train đã lọc; code gốc LTSF-Linear | `BAO_CAO_ETTH2_2.md` |
| 5 | GD/Adam toàn batch trên ETTh2; SGD 20 seed cho ETTh1, ETTm1; bảng B1, B4; hình H3 | `BAO_CAO_SGD_B1_B4.md` |
| 6 | Outlier (chuỗi và đích, 1/5/10%, 10σ); code gốc trên ETTm1; Linear với lr ×2 | `BAO_CAO_OUTLIER.md` |
| 7 | Góp ý TA: δ Huber theo MAD; λ* theo metric báo cáo; outlier 3σ, 5σ; SGD tinh chỉnh lr | `BAO_CAO_NV7.md` |

Test: **63, tất cả đạt.** Mọi con số chính có file tái lập, kèm git commit.

**Quy ước chốt cho báo cáo cuối (theo góp ý TA):**

- MSE test báo ở λ*_MSE (argmin MSE val); MAE test báo ở λ*_MAE (argmin MAE val); cùng quy tắc cho mọi mô hình và mục tiêu. Metric val không chứa phạt (đã kiểm bằng test).
- Huber: δ = 1,345 · 1,4826 · MAD của phần dư train của nghiệm MSE λ = 0, tính riêng cho từng ô. δ = 1 cũ chỉ còn trong phân tích độ nhạy.
- SGD: lịch `type1` và siêu tham số của script gốc là số chính (tái lập); SGD tinh chỉnh lr là kết quả bổ sung.
- Bảng chính dùng `results/huber_mad/results_v2.csv` (B3) và `results/huber_mad/outliers/results_v2.csv`.

---

## 1. Kết quả cuối theo câu hỏi nghiên cứu

### RQ1: Tại nghiệm tối ưu, khi nào DLinear và NLinear thu về Linear?

| Kết luận | Bằng chứng | Nguồn |
|---|---|---|
| DLinear ≡ Linear khi λ = 0, **với mọi hàm mục tiêu** | chứng minh (lớp hàm); ETTh1 H = 96: chênh dự báo ≤ 2,2e-13 | kế hoạch 3.4 mục 1.1; notebook 01 |
| DLinear + weight decay = ridge với phạt N⁻¹, với MSE, MAE, Huber | MSE ≤ 3,6e-13; Huber 3,2e-11; MAE trùng từng vòng IRLS | `BAO_CAO_TIEU_CHI_DUNG.md` mục 5 |
| Tại λ*, DLinear gần như trùng Linear trên cả 36 tổ hợp (dataset, H, mục tiêu) | chênh MSE test dưới 5e-4 | B3 |
| NLinear tách khỏi Linear ở mọi λ, qua ràng buộc tổng hàng | ‖r‖²/s khớp 15 chữ số; NLinear tốt hơn rõ ở ETTh2 và ETTh1 H = 720 | notebook 01; B3 |
| Lợi thế của DLinear trong bài gốc là hiệu ứng của huấn luyện, không phải kiến trúc | ETTh1 H = 720: bài gốc Linear 0,624 so với DLinear 0,472; SGD 20 seed Linear 0,506; nghiệm tối ưu cả hai 0,470. ETTh2 H = 720: 0,605 của DLinear là một seed may mắn (code gốc 5 seed: 0,575–0,733) | `BAO_CAO_SGD_B1_B4.md`; `BAO_CAO_ETTH2_2.md` D |
| Với SGD lr gốc, DLinear tốt hơn Linear chỉ ở 4/12 ô (đều ETTm1); khi tinh chỉnh lr cho cả hai, 0/12 | Δ test −0,0032 đến −0,0047 → mọi CI chứa 0 | `BAO_CAO_NV7.md` mục 6.4 |

**Câu trả lời ngắn:** khi huấn luyện tới nghiệm tối ưu, phép tách xu hướng và mùa vụ của DLinear không thêm gì vào lớp mô hình; khác biệt chỉ còn ở metric của hàm phạt, và rất nhỏ trên thực nghiệm. Phép chuẩn hóa theo giá trị cuối của NLinear là khác biệt thật, có ích trên dữ liệu có dịch chuyển mức (ETTh2).

### RQ2: Hàm mục tiêu ảnh hưởng thế nào, và bền vững với outlier ra sao?

| Kết luận | Bằng chứng | Nguồn |
|---|---|---|
| Huấn luyện bằng MAE cho MSE test tốt hơn MSE với Linear và DLinear ở cả 12 (dataset, H) | chênh 0,001–0,015 (ETTh1), 0,023–0,151 (ETTh2), 0,002–0,006 (ETTm1) | `BAO_CAO_NV7.md` mục 3.4 |
| Với NLinear, lợi thế của MAE nhỏ hoặc âm; trên ETTh2 MAE kém ở cả 4 H | | như trên |
| Huber (δ_MAD) đứng giữa MSE và MAE ở 28/36 ô theo MSE test, 34/36 theo MAE test | | `BAO_CAO_NV7.md` mục 4.4 |
| Trên ETTh2, khoảng một nửa lợi thế của MAE với Linear ở H ≥ 336 đến từ đoạn cảm biến bị kẹt | lọc cửa sổ hằng: lợi còn 47% (H = 336), 60% (H = 720) | `BAO_CAO_ETTH2_2.md` C |
| Outlier chỉ ở đích: MAE và Huber bền vững, MSE suy giảm | 10%, 10σ: MSE test của MAE, Huber đổi ≤ 0,004; MSE +0,009 đến +0,014 | `BAO_CAO_OUTLIER.md`; `BAO_CAO_NV7.md` |
| Outlier trong chuỗi (có cả ở đầu vào): không mục tiêu nào bền vững; ở 10%, 10σ, MSE suy giảm ít nhất | suy giảm 0 → 10%: MSE < Huber < MAE ở 8/8 ô | như trên |
| Thứ hạng phụ thuộc độ lớn outlier | 3σ: MAE tốt nhất ở 1% và 10%; đảo chiều giữa 5σ và 10σ ở rate 10% (Linear, 1 seed) | `BAO_CAO_NV7.md` mục 5 |
| Với outlier trong chuỗi, λ* chọn trên val sạch về 0 | 48/48 ô ở rate 5%, 10% | `BAO_CAO_OUTLIER.md` |
| Ở 10%, 10σ, thứ tự MSE so với Huber nằm trong nhiễu theo vị trí outlier | độ lệch giữa 3 seed cùng cỡ khác biệt giữa mục tiêu | `BAO_CAO_OUTLIER.md` B.5 |

**Giả thuyết (ghi là giả thuyết, chưa kiểm):** outlier trong x hoạt động như nhiễu đầu vào, tương đương một số hạng phạt Tikhonov (Bishop, 1995). Điều này giải thích cả việc λ* về 0 lẫn việc MSE ít bị ảnh hưởng nhất. MAE kém nhất vì nghiệm L1 dễ bị kéo theo các điểm có đòn bẩy cao.

### RQ3: SGD kém nghiệm tối ưu bao nhiêu?

| Kết luận | Bằng chứng | Nguồn |
|---|---|---|
| Nghiệm dạng đóng bằng hoặc tốt hơn số công bố ở 28/36 ô | ngoại lệ: Linear, DLinear trên ETTh2 | `BAO_CAO_ETTH2_SGD.md` |
| SGD (20 seed, siêu tham số gốc) kém nghiệm dạng đóng trên test ở 24/24 ô ETTh1, ETTm1 | 0,3% đến 8% | `BAO_CAO_SGD_B1_B4.md` |
| Sau khi tinh chỉnh lr (15 seed mới), SGD vẫn kém ở 23/24 ô ETTh1, ETTm1 (có ý nghĩa) | Δ test +0,002 đến +0,029; tinh chỉnh chủ yếu cải thiện val | `BAO_CAO_NV7.md` mục 6.3 |
| SGD tốt hơn nghiệm dạng đóng trên cả val và test chỉ ở 5 ô, đều ETTh2 với Linear/DLinear | | như trên |
| Cơ chế trên ETTh2 chưa xác định; đã loại trừ ba ứng viên | không phải co về W₀ ≠ 0; không phải đoạn kẹt; không phải dừng sớm tự nó (GD toàn batch trùng ridge tới 3e-4) | `BAO_CAO_ETTH2_2.md`; `BAO_CAO_SGD_B1_B4.md` A |
| Số công bố có nhiễu theo seed lớn; 8/36 ô thấp hơn mọi seed của repo | 2 ô kiểm bằng code gốc (ETTh2 H = 720 DLinear, ETTm1 H = 96 DLinear): code gốc cũng không tái lập được | `BAO_CAO_SGD_B1_B4.md`; `BAO_CAO_OUTLIER.md` D.1 |

**Câu trả lời ngắn:** trên ETTh1 và ETTm1, SGD dừng sớm luôn kém nghiệm tối ưu, kể cả khi đã tinh chỉnh learning rate. Trên ETTh2, Adam mini-batch tìm được nghiệm tốt hơn mọi điểm trên đường ridge, ở cả val lẫn test, với cơ chế chưa xác định.

---

## 2. Hạn chế (đưa vào báo cáo)

1. Chỉ ba dataset ETT, chỉ mô hình tuyến tính, kênh dùng chung trọng số.
2. **Val và test của ETT thuộc hai chế độ khác nhau** (ETTh1: MSE val gấp 2–3 lần MSE test). Mọi lựa chọn dựa trên val (λ*, lr, epoch) chịu giới hạn này. Ví dụ: co ridge về giá trị cuối làm val tốt hơn nhưng test kém hơn ở 14/16 tổ hợp ETTh1, ETTm1.
3. Outlier nhân tạo: một dạng (cộng ±kσ vào từng điểm), một dataset (ETTh1); độ lớn 3σ, 5σ chỉ chạy Linear, H ∈ {96, 720}, một seed.
4. δ của Huber dùng chung cho cả ô, trong khi σ̂ theo từng bước h chênh nhau 2,5–5 lần.
5. **Tinh chỉnh lr chưa trọn:** 15/36 ô chọn mức ở đầu mút khoảng đã thử. Ở ETTm1, Linear và NLinear chọn ×8 mà MSE val vẫn còn giảm (mục 4.1).
6. Nghiệm MAE chỉ duy nhất ở mức hàm mục tiêu và metric (J lệch ≤ 2e-10); trọng số giữa các điểm khởi tạo lệch khoảng 3,5e-3.
7. Mọi metric tính trên dữ liệu đã chuẩn hóa, như LTSF-Linear.
8. Một số JSON ghi commit `-dirty` (đã ghi rõ trong từng báo cáo; code solver không đổi ở các commit đó).

---

## 3. Dàn ý báo cáo cuối và nguồn cho từng phần

Theo outline: vấn đề → thiết lập → kết quả → diễn giải → hạn chế → phụ lục toán.

| Phần | Nội dung | Nguồn sẵn có |
|---|---|---|
| 1. Vấn đề | Bối cảnh LTSF-Linear; ưu thế đến từ kiến trúc hay cách huấn luyện; ba RQ | outline mục 1 |
| 2. Công trình liên quan | LTSF-Linear; **Toner & Darlow (ICML 2024)**; hồi quy bền vững (Huber, IRLS); dừng sớm và ridge (Ali et al., 2019); nhiễu đầu vào và Tikhonov (Bishop, 1995) | **chưa đọc Toner & Darlow** |
| 3. Thiết lập | Dữ liệu, chia tập, chuẩn hóa; ba mô hình; ba mục tiêu; nghiệm dạng đóng và IRLS; quy ước λ và hàm phạt từng mô hình; chọn λ*, δ; SGD và tinh chỉnh lr; quy tắc đọc ba trường hợp | kế hoạch 3.4 mục 3–4; `BAO_CAO_NV7.md` |
| 4.1 Kết quả RQ1 | Bảng B2; hình H1; B3 phần DLinear − Linear; ETTh1 H = 720 | mục 1 ở trên |
| 4.2 Kết quả RQ2 | B3; ETTh2 đoạn kẹt; outlier (H2_v2, hình độ lớn); độ nhạy δ | **đoạn nháp RQ2** ở `BAO_CAO_NV7.md` mục 7 |
| 4.3 Kết quả RQ3 | B1, B4, B4-tinh chỉnh; hình H3; C.3 | **đoạn nháp RQ3** ở `BAO_CAO_NV7.md` mục 7 |
| 4.4 ETTh2 | Đoạn cảm biến bị kẹt và phân tích độ nhạy | **đoạn nháp** ở `BAO_CAO_ETTH2_2.md` mục 6 |
| 5. Diễn giải | Kiến trúc so với huấn luyện; outlier ở x so với ở y; giả thuyết nhiễu đầu vào; nhiễu theo seed trong các bảng công bố | mục 1 ở trên |
| 6. Hạn chế, hướng phát triển | mục 2 ở trên; thêm: dataset khác, mô hình phi tuyến, δ theo từng h, cơ chế ETTh2 | |
| Phụ lục A | Dẫn xuất ba dự đoán | kế hoạch 3.4 mục 1 |
| Phụ lục B | IRLS: MM, tách theo h, bias bằng cột 1, ràng buộc theo hàng, chế độ pha, tiêu chí dừng | kế hoạch 3.4 mục 3 |
| Phụ lục C | Bảng đầy đủ B3, B4, outlier | các file CSV |

**Bảng và hình cho báo cáo:**

| # | Nội dung | File | Trạng thái |
|---|---|---|---|
| B1 | SGD 20 seed so với số công bố, 36 ô | `results/sgd_b1_b4/` | ✅ |
| B2 | Bốn ô trùng/tách của RQ1 | `results/B2_three_predictions.md` | ✅ |
| B3 | Mục tiêu × metric, 216 ô (tiêu chí mới, Huber δ_MAD) | `results/huber_mad/results_v2.csv` | ✅ |
| B4 | SGD so với nghiệm tối ưu, nhãn ba trường hợp; kèm bản tinh chỉnh lr | `results/sgd_b1_b4/`, `results/nv7/tables_D.md` | ✅ |
| H1 | ‖ΔW‖ giữa DLinear và Linear theo λ | `results/H1_three_predictions.png` | ✅ |
| H2 | MSE/MAE test theo tỷ lệ outlier | `results/outliers/H2_v2_*.png` | ✅ |
| H3 | SGD theo seed so với nghiệm tối ưu và số công bố | `results/sgd_b1_b4/H3_sgd_vs_optimum.png` | ✅ |
| H4 | Độ nhạy theo δ | `results/huber_mad/A3_delta_sensitivity.png` | ✅ |
| H5 | Theo độ lớn outlier | `results/outliers_mag/C_outlier_magnitude.png` | ✅ |

---

## 4. Việc còn lại

### 4.1. (Tùy chọn, ngoại lệ duy nhất của việc đóng băng) Mở rộng lưới lr ở các ô đầu mút

Kết luận trung tâm của RQ3 là "SGD vẫn kém nghiệm tối ưu kể cả khi tinh chỉnh lr". Kết luận này yếu nhất ở chính những ô mà lr tốt nhất có thể nằm ngoài khoảng đã thử. Đó là 9 ô ETTh1 và ETTm1 chọn mức ở đầu mút:

- ETTm1: Linear H = 96, 192, 336 và NLinear H = 96, 720 (chọn ×8, MSE val còn giảm);
- ETTh1: Linear H = 192 và DLinear H = 720 (chọn ×8); Linear H = 720 và DLinear H = 336 (chọn ×0,5).

**Cách làm:** mở rộng thêm tối đa 2 mức theo đúng hướng (×16, ×32, hoặc ×0,25, ×0,125), 5 seed chọn lr như cũ; nếu mức chọn đổi, chạy lại 15 seed cuối. Khoảng 2–3 giờ CPU. Nếu không làm, ghi rõ hạn chế số 5.

### 4.2. Viết báo cáo

- [ ] **Đọc Toner & Darlow (ICML 2024)**, viết phần công trình liên quan, kiểm hai tuyên bố liên quan đã ghi trong kế hoạch.
- [ ] Viết phần 1–3 (vấn đề, công trình liên quan, thiết lập).
- [ ] Viết phần 4 từ ba đoạn nháp có sẵn, cập nhật số theo quy ước chốt.
- [ ] Viết phần 5–6 (diễn giải, hạn chế).
- [ ] Phụ lục toán A, B.
- [ ] Rà mọi con số trong báo cáo: mỗi số trỏ được về một file CSV hoặc JSON.

### 4.3. Đóng gói repo

- [ ] Notebook `02_loss_comparison.ipynb` (B3, H2, H4, H5) và `03_outliers_and_sgd.ipynb` (B1, B4, H3), dựng từ các script báo cáo đã có.
- [ ] `run_all.sh`: tái lập các bảng và hình từ đầu. Grid đầy đủ mất khoảng 19 giờ, nên có hai chế độ: đầy đủ, và nhanh (sinh lại bảng, hình từ CSV có sẵn, cộng chạy lại một ô làm phép kiểm).
- [ ] Kiểm `run_all.sh` (chế độ nhanh) chạy sạch trên môi trường trắng.
- [ ] Hoàn thiện `README.md`: mục tiêu, cấu trúc, cài đặt (torch cu130), lệnh tái lập, bảng tóm tắt kết quả.
- [ ] Công khai repo `dlinear-objective-lab`.

### 4.4. Nộp

- `WEEK03 - Outlier Robustness and SGD Gap`: B1, B4, H2, H3, kèm tóm tắt từ mục 1.
- `WEEK04 - Experimental Report and Toolkit`: báo cáo cuối, repo công khai.
