# Nhiệm vụ 7 cho Claude Code: áp dụng góp ý của TA

*Lập ngày 07/10/2026, sau `docs/BAO_CAO_OUTLIER.md` và câu trả lời của TA cho Tuần 2 (`Answer_Week02.pdf`, mục VI). Đặt file ở gốc repo rồi yêu cầu Claude Code: "Đọc NHIEM_VU_7.md và làm theo".*

**Đây là nhiệm vụ thí nghiệm cuối cùng.** Sau nhiệm vụ này, đề tài chuyển sang viết báo cáo. Không mở hướng mới.

## 0. Góp ý của TA và việc tương ứng

| # | TA góp ý | Việc | Phần |
|---|---|---|---|
| 1 | Lập luận "δ = 1 vì dữ liệu đã chuẩn hóa" sai: δ phải so với **thang của phần dư**. Gợi ý δ = 1,345 · σ̂, σ̂ = 1,4826 · MAD của phần dư từ một lần fit ban đầu; thêm phân tích độ nhạy theo δ | Tính δ cho từng ô; chạy lại Huber trong grid và trong thí nghiệm outlier; quét độ nhạy | A |
| 2 | Chọn λ* theo metric sẽ báo cáo trên test (báo MSE test thì chọn theo MSE val); cùng tiêu chí cho mọi mô hình; không đưa λ‖w‖² vào metric val | Đổi tiêu chí chính, sinh lại mọi bảng (không cần chạy lại) | B |
| 3 | Làm cả hai cách chèn outlier (đã làm). Có thể thử 3σ, 5σ, 10σ | Thêm độ lớn 3σ, 5σ, phạm vi nhỏ | C |
| 4 | Nên tinh chỉnh learning rate của GD trên val trước khi so với OLS, cùng ngân sách tinh chỉnh cho mọi cấu hình | Thêm biến thể "SGD đã tinh chỉnh lr" | D |
| 5 | Lịch lr của repo làm số chính nếu mục tiêu là tái lập | Giữ nguyên `type1` | — |

**Thứ tự làm:** B (không chạy gì) → A → C trên GPU, chạy song song với D trên CPU → E.

## 1. Quy tắc bắt buộc

Giữ nguyên mọi quy tắc của các nhiệm vụ trước:

- TF32 tắt tường minh; không nới ngưỡng test; không đổi toán của `irls`.
- Mọi số liệu có file tái lập; ghi phần cứng và git commit vào mọi JSON và mọi dòng CSV; commit sạch trước mỗi lần chạy dài.
- **Tập test không bao giờ được dùng để chọn bất cứ thứ gì** (λ, δ, lr, epoch).
- Không ghi đè kết quả cũ. Kết quả mới ghi vào thư mục mới, nêu ở từng phần.
- Cấu hình bộ giải như grid: chế độ pha, `chunk = 16`, cấu hình dừng (a′) cho MAE, `tol = 1e-9` cho Huber, MSE dạng đóng, warm start, fallback float64 theo từng h.

## 2. Phần B: đổi tiêu chí chọn λ* (làm trước, không chạy lại gì)

### B.1. Tiêu chí mới

Với mỗi (dataset, H, mô hình, mục tiêu huấn luyện):

- **λ*_MSE** = argmin MSE val, dùng để báo **MSE test**;
- **λ*_MAE** = argmin MAE val, dùng để báo **MAE test**.

Áp cùng một quy tắc cho mọi mô hình và mọi mục tiêu huấn luyện. Bỏ cách chọn `val_objective` khỏi bảng chính, chỉ giữ trong phụ lục.

### B.2. Việc

- Thêm cách chọn `val_mae` vào `select_lambda.py`. `val_mse` đã có.
- **Kiểm và ghi rõ:** các cột `val_mse`, `val_mae`, `val_huber` trong `lambda_path.csv` **không** chứa số hạng phạt λ‖w‖². Thêm một test khẳng định điều này: tính lại metric val từ W đã lưu và so với cột trong CSV.
- Sinh lại theo tiêu chí mới:
  - `results/results_v2.csv`: bảng **B3**, gồm 3 dataset × 4 H × 3 mô hình × 3 mục tiêu, mỗi ô có λ = 0 và λ* (MSE test ở λ*_MSE, MAE test ở λ*_MAE);
  - các bảng của thí nghiệm outlier (`results/outliers/results_v2.csv`).
- Bảng B4 (SGD so với dạng đóng) đã dùng λ* theo MSE val, nên không đổi. Kiểm lại cho chắc.

### B.3. Báo cáo những gì đổi

Lập danh sách **mọi kết luận trong các báo cáo trước bị đổi** do tiêu chí mới. Danh sách gồm:

- thứ hạng mục tiêu trong từng ô B3;
- câu "MAE cho MSE test tốt hơn MSE" (đếm lại theo dataset và mô hình);
- bảng xếp hạng mục tiêu trong thí nghiệm outlier;
- λ* của DLinear trên ETTh1 H = 96.

Phần Huber của B3 sẽ đổi thêm ở Phần A. Ghi rõ những điểm nào "chờ Phần A".

## 3. Phần A: δ của Huber theo thang phần dư

### A.1. Cách tính δ

Với mỗi (dataset, H, mô hình) và mỗi tập train (gốc, hoặc đã chèn outlier):

1. Lấy nghiệm **MSE λ = 0 của chính mô hình đó** trên **chính tập train đó** (có sẵn; với NLinear là nghiệm có ràng buộc).
2. Tính phần dư train R (mọi phần tử, kích thước n × H).
3. Tính σ̂ = 1,4826 · median(|R − median(R)|), gộp mọi phần tử.
4. δ = 1,345 · σ̂.

- Ghi δ, σ̂ của mọi ô vào `results/huber_mad/deltas.csv`. Ghi thêm, để tham khảo, khoảng σ̂ theo từng h (min, trung vị, max), để biết một δ chung cho cả ô có hợp lý không.
- Viết hàm `huber_delta_mad(R)` trong `src/solvers.py`, kèm test: với dữ liệu giả có phân phối chuẩn N(0, s²), σ̂ phải gần s (sai lệch tương đối dưới 5% với n lớn); một số outlier lớn không được làm σ̂ thay đổi nhiều.
- DLinear: dùng phần dư của chính nó ở λ = 0, vốn trùng Linear. Kiểm δ của hai mô hình trùng nhau.

### A.2. Chạy lại Huber

Kết quả ghi vào `results/huber_mad/`:

- **Grid:** 3 dataset × 4 H, Linear, DLinear (λ > 0; λ = 0 chép từ Linear như grid), NLinear. Đường λ như grid, tự mở rộng ở đầu mút. Lưu ý: quy ước phạt chia cho δ, nên đổi δ làm đổi thang của λ. Ghi số lần phải mở rộng đầu mút.
- **Outlier:** mọi lần chạy Huber trong `results/outliers/` (series 1/5/10%, target 1/10%, B.5 seed 2027–2028), với δ tính từ **tập train đã chèn** theo A.1. MAD vốn bền vững nên đây là cách làm đúng.
- **Kiểm trước:** chạy lại Huber với δ = 1 cho Linear ETTh1 H = 96 bằng code mới. Kết quả phải trùng dòng cũ tới 1e-10, để chắc thay đổi code không ảnh hưởng gì ngoài δ.
- Dùng `--dry-run` để ước lượng; ngân sách GPU cho A cộng C là **8 giờ**. Nếu vượt, cắt Phần C trước, rồi cắt B.5 khỏi phần outlier.

### A.3. Độ nhạy theo δ

- **Ô:** ETTh1 H ∈ {96, 720} và ETTh2 H = 720; mô hình Linear và NLinear.
- **δ:** {0,5; 1; 2; 4} × δ_MAD, cộng δ = 1 cố định (giá trị cũ).
- Với mỗi δ: λ* theo tiêu chí B.1; ghi MSE test, MAE test, và vị trí của Huber so với MSE và MAE trong ô.

### A.4. Kết luận cần trả lời

- δ_MAD nằm trong khoảng nào trên các ô, và khác δ = 1 bao nhiêu.
- Thứ hạng của Huber so với MSE và MAE trong B3 có đổi không, đổi ở bao nhiêu ô.
- Kết luận của thí nghiệm outlier về Huber có đổi không.
- Kết luận có nhạy với δ không (A.3).

## 4. Phần C: độ lớn outlier 3σ và 5σ

- **Phạm vi:** Linear, ETTh1, H ∈ {96, 720}; rate ∈ {1%, 10%}; chế độ series và target; seed 2026; ba mục tiêu (Huber dùng δ_MAD theo A.1).
- **Độ lớn:** 3 và 5 (10 đã có). Dùng tham số `magnitude` sẵn có của `contaminate_series` và `contaminate_targets`. Ghi cột `outlier_magnitude` vào CSV (các dòng cũ coi là 10).
- **Kết quả:** `results/outliers_mag/`. Bảng: MSE và MAE test theo (độ lớn, rate, chế độ, mục tiêu), và mục tiêu tốt nhất trong ô.
- **Câu hỏi:** với outlier nhỏ hơn (3σ, 5σ), MAE và Huber có bền vững hơn MSE khi chèn vào chuỗi gốc không? Ở độ lớn nào thứ hạng đảo?

## 5. Phần D: SGD đã tinh chỉnh learning rate (CPU, song song với A và C)

### D.1. Thiết kế (cùng ngân sách cho mọi ô)

Với mỗi (dataset, mô hình, H), 36 ô:

1. **Chọn lr:** thử lr ∈ {0,5; 1; 2; 4} × lr của script gốc, mỗi mức chạy seed 2021–2025. Chọn mức có **MSE val trung bình** thấp nhất. Kết quả mức ×1 với seed 2021–2025 đã có, dùng lại.
2. Nếu mức chọn là ×4 (đầu mút), thử thêm ×8 một lần với cùng 5 seed.
3. **Chạy cuối:** seed 2026–2040 (15 seed **mới**) ở lr đã chọn. Kết quả mức ×1 với các seed này đã có, dùng lại nếu lr chọn là ×1.

**Vì sao dùng seed mới cho phần cuối:** 5 seed đầu đã dùng để chọn lr trên val. Nếu tính Δ val trên chính chúng, Δ val sẽ lạc quan một cách có hệ thống cho SGD. Mọi thống kê cuối của "SGD đã tinh chỉnh" chỉ dùng 15 seed 2026–2040.

Các tham số khác (batch, epoch, patience, `lradj`) giữ như script gốc.

### D.2. Ngân sách

- **10 giờ CPU thời gian thực**, được chạy song song nhiều tiến trình.
- Thứ tự: ETTh2 → ETTh1 → ETTm1. ETTm1 đắt nhất (khoảng 60 giây một lần chạy).
- Nếu ước lượng vượt: với ETTm1, chọn lr bằng 3 seed (2021–2023) và chạy cuối 10 seed (2026–2035). Ghi rõ.

### D.3. Bảng

- **Bảng lr đã chọn** cho 36 ô, kèm MSE val trung bình của từng mức lr.
- **B4-tinh chỉnh:** như B4 (Δ val, Δ test so với dạng đóng λ*, CI 95% theo t, nhãn theo quy tắc ba trường hợp), với SGD đã tinh chỉnh (15 seed). Đặt cạnh nhãn của B4 cũ.
- **C.3-tinh chỉnh:** DLinear − Linear khi cả hai được tinh chỉnh lr (CI Welch). Đặc biệt: 4 ô ETTm1 trước đây DLinear tốt hơn có ý nghĩa thì giờ còn không.
- Đếm: số ô SGD đã tinh chỉnh vẫn kém dạng đóng trên test; số ô "SGD tốt hơn thật"; nhãn nào đổi so với B4 cũ.

## 6. Phần E: báo cáo

Viết `docs/BAO_CAO_NV7.md`, cùng văn phong các báo cáo trước:

1. **Kết luận** ở đầu, một đoạn cho mỗi phần A, B, C, D.
2. **Mục "Những kết luận đổi so với các báo cáo trước"**: liệt kê từng kết luận cũ, giá trị cũ, giá trị mới, lý do (tiêu chí λ*, δ mới, lr tinh chỉnh). Đây là mục quan trọng nhất cho việc viết báo cáo cuối.
3. Bảng của từng phần, và bảng **B3 hoàn chỉnh** theo tiêu chí mới (Huber dùng δ_MAD).
4. Cập nhật hình H2 (Huber với δ_MAD) thành `H2_v2_*.png`; thêm hình độ nhạy theo δ (A.3) và hình theo độ lớn outlier (C).
5. **Cập nhật hai đoạn nháp** cho mục RQ2 (từ `BAO_CAO_OUTLIER.md`) và RQ3 (từ `BAO_CAO_SGD_B1_B4.md`) theo kết quả mới, mỗi đoạn khoảng 200–250 từ. Chỉ nêu những gì đã kiểm.
6. **Tái lập:** các lệnh đã chạy.

Ghi thay đổi code vào `docs/NHAT_KY_THAY_DOI.md`.

## 7. Điểm dừng để hỏi người dùng

Dừng lại, báo cáo và chờ ý kiến nếu:

- Có test không đạt, kể cả test mới ở B.2 về metric val.
- Kiểm trước ở A.2 (δ = 1 với code mới) không trùng dòng cũ.
- `irls` báo `RuntimeError: J tăng`, hoặc có dòng `converged = False`.
- δ_MAD của một ô nhỏ hơn 0,05 hoặc lớn hơn 5 (bất thường; ví dụ phần dư bị các đoạn hằng của ETTh2 chi phối).
- Đường λ của Huber chạm đầu mút sau khi đã mở rộng tối đa ở quá 3 ô.
- Ước lượng vượt ngân sách kể cả sau khi cắt theo A.2 và D.2.

## 8. Không làm trong phiên này

- Không thêm dataset, mô hình, hay mục tiêu mới.
- Không tinh chỉnh δ trên val (δ theo công thức của TA là đủ; độ nhạy ở A.3 chỉ để báo cáo).
- Không tinh chỉnh tham số nào của SGD ngoài lr.
- Không điều tra thêm cơ chế ETTh2 hay giả thuyết "nhiễu đầu vào tương đương ridge"; chỉ ghi chúng là giả thuyết trong báo cáo.
- Không sửa kế hoạch, checklist, hay các báo cáo cũ.
