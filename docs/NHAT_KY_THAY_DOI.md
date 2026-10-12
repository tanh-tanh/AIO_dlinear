# Nhật ký thay đổi code: IRLS (30/09/2026)

Phạm vi: mục 11.2 (IRLS float32 hay float64 trên GPU), rồi thêm phạt λ và đổi `tol` mặc định. Kết quả thí nghiệm ở [BAO_CAO_IRLS_DTYPE.md](./reports/BAO_CAO_IRLS_DTYPE.md).

Test: đầu phiên 26 → cuối phiên **32, tất cả đạt** (`python -m unittest discover -s tests -t .`).

## Tóm tắt

| File | Loại | Nội dung |
|---|---|---|
| `src/solvers.py` | sửa | `irls`: thêm `lam`, `pen`, `gram_dtype`, `callback`; `tol` mặc định 1e-12 → 1e-9; ngưỡng "J tăng" theo ε của kiểu số; bước giải dạng hiệu chỉnh |
| `tests/test_irls.py` | sửa | thêm 6 test (float32, pha độ chính xác, 4 test λ); ghim `tol=1e-12` cho 3 test đo độ chính xác |
| `scripts/irls_dtype_check.py` | mới | thí nghiệm 11.2: float32 / float64 / pha, đo giờ trên GPU |
| `results/irls_dtype/` | mới | 6 cặp JSON + NPY (số đo, quỹ đạo J và sai số test theo vòng, W) |
| `docs/BAO_CAO_IRLS_DTYPE.md` | mới | báo cáo 11.2 |
| `docs/NHAT_KY_THAY_DOI.md` | mới | file này |
| `README.md` | sửa | thêm thư mục `scripts/` và quy ước λ cho MAE/Huber |

Không đổi: `src/data.py`, `operators.py`, `metrics.py`, `models.py`, `sgd.py`, các notebook, `docs/CHECKLIST_CON_LAI.md`.

## 1. `src/solvers.py`, hàm `irls`

Chữ ký:

```python
# trước
irls(Xt, Y, delta, W0, constrained=False, max_iter=1000, tol=1e-12)
# sau
irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9,
     callback=None, gram_dtype=None)
```

Mọi lời gọi cũ vẫn chạy (repo không có chỗ nào truyền `constrained` theo vị trí). Kết quả số có thể khác chút do mục 1.2 và 1.6.

### 1.1. Ngưỡng "J tăng" theo ε của kiểu số

- **Lỗi:** chạy float32 là báo `RuntimeError: J tăng` ngay, cả trên ETTh1 lẫn dữ liệu giả của test.
  - Nguyên nhân: ngưỡng cố định `J_new > J_old·(1 + 1e-10)` được chỉnh cho float64. Ở float32, J tính lại lệch 1 ulp (khoảng 1e-7 tương đối) do làm tròn.
- **Sửa:** `rise_tol = max(1e-10, 100·ε)`.
  - float64 giữ nguyên hành vi, vì 100ε₆₄ = 2e-14 < 1e-10.
  - Nếu J tăng trong ngưỡng thì coi là đã chạm độ chính xác của kiểu số và dừng. Tăng vượt ngưỡng vẫn báo lỗi.

### 1.2. Bước giải dạng hiệu chỉnh

- **Trước:** `W = A⁻¹c`, với `c = Xtᵀ diag(w) y`.
- **Sau:** `W = W + A⁻¹g`, với `g = Xtᵀ diag(w) r` và r là phần dư của W hiện tại.
- **Lý do:**
  - Trong số học chính xác hai dạng bằng nhau, vì W + A⁻¹(c − AW) = A⁻¹c.
  - Điểm dừng g = 0 được tính ở độ chính xác của phần dư, nên A cho phép kém chính xác hơn. Đây là điều kiện để mục 1.3 hoạt động.
  - Dạng này cũng giúp float32 thuần: MAE trên ETTh1 đi được 62 vòng thay vì 38.
  - Nhánh NLinear (`constrained=True`) đổi theo: ŵ = W + A⁻¹g, rồi chiếu lên ràng buộc như cũ.

### 1.3. Tham số `gram_dtype` (pha độ chính xác)

- **Cách làm:** chỉ lập A_h bằng kiểu này, ví dụ `torch.float32` khi Xt là float64. Đây là phần tốn H·n·p² phép tính. Phần dư, gradient, J và phép giải vẫn dùng kiểu của Xt.
- **Kết quả trên ETTh1 H = 96:**
  - Nghiệm như float64: Huber lệch W 2,2e-10; MAE ở cùng số vòng lệch MSE/MAE test khoảng 1e-6.
  - Tốc độ: 1,49 s/vòng, so với 1,29 s (float32) và 15,2 s (float64).
- **Mặc định `None`**: giữ hành vi cũ.

### 1.4. Tham số `callback(it, W, J_new)`

Gọi mỗi vòng, trước phép kiểm J. Dùng để ghi quỹ đạo, và để lấy lại W khi `irls` báo lỗi. Không đổi hành vi.

### 1.5. Tham số `lam`, `pen` (phạt ridge)

**Hàm mục tiêu** (dạng tổng, như quy ước MSE `‖Y − XWᵀ‖² + λ‖W‖²` của README):

    F(W) = Σ ρ_δ(r)/δ + λ Σ_h w_hᵀ Pen w_h

- ρ_δ(r) = r²/2 nếu |r| ≤ δ, δ(|r| − δ/2) nếu ngược lại.
- w_h là L hệ số đầu của hàng h. **Không phạt bias.**
- `Pen = I` nếu `pen=None`.

**Hệ quả:**

| Trường hợp | F |
|---|---|
| MAE (δ = 1e-6 → 0) | Σ\|r\| + λ‖W‖² |
| Huber δ = 1 | Σρ₁(r) + λ‖W‖² |
| δ lớn | ‖r‖²/(2δ) + λ‖W‖², tức ridge MSE với λ_mse = 2δλ |

- **Vì sao chia cho δ:** nếu không chia, λ của MAE sẽ phải nhỏ cỡ δ mới có tác dụng.
- **Lưu ý khi chọn lưới λ:** với Huber, vùng bậc hai là r²/2, nên λ của Huber tương ứng khoảng λ_mse/2 khi phần dư nhỏ.

**Cài đặt:**

- Trọng số của IRLS là w = δ/max(|r|, δ), tức là 1/max(|r|, δ) đã nhân δ. Vì vậy phạt được nhân tương ứng:
  - A_h = Xtᵀ diag(w_h) Xt + 2λδ·Pen
  - g_h = Xtᵀ diag(w_h) r_h − 2λδ·Pen w_h
- Hàng và cột bias của Pen bằng 0.
- **`pen`** nhận ma trận [L, L] đối xứng nửa xác định dương (numpy hoặc tensor), tự chuyển sang dtype và device của Xt.
- **DLinear có weight decay** dùng `pen = N⁻¹`, giống `ridge_pen`. Tương đương này chỉ phụ thuộc W_eff nên đúng với mọi hàm mất mát. Nhờ đó DLinear λ > 0 chạy được ở p = L + 1 thay vì 2L + 1, rẻ hơn khoảng 4 lần.
- **J báo ra** (qua `callback`, dùng để dừng) là δ·F/(n·H). Khi λ = 0 thì J đúng bằng `huber_objective` như trước.

**Chạy thử trên GPU** (ETTh1 H = 96, Huber δ = 1, chế độ pha, `pen = N⁻¹` với k = 25, λ = 281 chưa chọn trên validation):

- 7 vòng, 10,6 s.
- MSE test 0,367131 (λ = 0 là 0,367800).
- ‖W‖ giảm từ 3,554 xuống 3,353.

### 1.6. `tol` mặc định 1e-12 → 1e-9

- **Lý do:** MAE trên ETTh1 H = 96 chạy 1000 vòng vẫn chưa đạt 1e-12.
- **Với 1e-9:**
  - Dừng ở khoảng vòng 189 (khoảng 280 s ở chế độ pha).
  - MSE/MAE test lệch nghiệm ở vòng 1000 khoảng 2e-6 / 5e-6 (bảng ở mục 5 của báo cáo).
  - Huber δ = 1 dừng ở vòng 7 với test y như `tol = 1e-12`.

### 1.7. Docstring của module

Bỏ câu "nhân mọi trọng số với cùng hằng số không đổi nghiệm", vì câu này sai khi λ > 0. Thay bằng quy ước λ cho MAE/Huber.

## 2. `tests/test_irls.py`

**Thêm 6 test:**

| Test | Kiểm gì | Sai số đo được → ngưỡng |
|---|---|---|
| `TestMAE.test_float32_no_false_alarm` | float32 không báo lỗi giả; MAE (tính lại bằng float64) kém float64 rất ít | 2e-6 → 1e-4 |
| `TestMAE.test_mixed_precision_matches_float64` | `gram_dtype=float32` cho cùng W với float64; W trả về là float64 | 7e-12 → 1e-8 |
| `TestPenalty.test_large_delta_equals_ridge` | δ lớn + λ = ridge MSE với 2δλ, bias không phạt, 2 vòng | 4e-16 → 1e-10 |
| `TestPenalty.test_large_delta_penalty_matrix` | như trên với `pen` là ma trận xác định dương ngẫu nhiên | 4e-16 → 1e-10 |
| `TestPenalty.test_constrained_large_delta_equals_kkt` | NLinear + λ = hệ KKT có phạt | → 1e-10 |
| `TestPenalty.test_mae_matches_qp` | MAE + λ + `pen`: so F = Σ\|r\| + λwᵀPen w với quy hoạch bậc hai giải bằng SLSQP (scipy), so hai chiều; và phạt làm nghiệm khác hẳn λ = 0 | 1e-8–3e-8 → 1e-6 |

- **Vì sao `test_mae_matches_qp` so hai chiều:** nếu `irls` dùng sai quy ước λ (ví dụ lệch 2 lần), F của nó kém hẳn. Nếu SLSQP dừng sớm, `irls` sẽ tốt hơn hẳn. Cả hai trường hợp đều bị bắt.
- **SLSQP kết thúc với "Positive directional derivative"**, tức không cải thiện được nữa ở giới hạn chính xác. Đây không phải lỗi.
- **Không dùng phép kiểm "gradient bằng 0"**, dù đã thử trước. Ở δ = 1e-6, clamp(r/δ) nhạy tới mức r lệch 1e-12 làm gradient lệch 1e-6, nên gradient không giảm đều theo số vòng.

**Sửa 3 test:** `test_multi_init_agree`, `test_float32_no_false_alarm`, `test_mixed_precision_matches_float64` giờ truyền `tol=1e-12` rõ ràng. Ba test này kiểm độ chính xác của nghiệm, không kiểm tiêu chí dừng. Với `tol` mặc định mới, chúng dừng sớm hơn mức mà ngưỡng của chúng yêu cầu.

**Import:** thêm `minimize` cạnh `linprog`. Nếu thiếu scipy thì hai test bị bỏ qua.

## 3. `scripts/irls_dtype_check.py` (mới)

- **Phạm vi:** chạy một ô (ETTh1, L = 336, H = 96, Linear, λ = 0, khởi tạo OLS). Mỗi lần một mục tiêu (`mae`, `huber`) ở một kiểu (`float32`, `mixed`, `float64`).
- **Đo và ghi:**
  - Thời gian thật trên GPU, có khởi động trước và `cuda.synchronize`.
  - Đỉnh bộ nhớ.
  - κ(A_h) tính bằng float64 qua trị riêng.
  - J train, MSE/MAE test: cả ở điểm dừng lẫn theo từng vòng, đều tính lại bằng float64 sau khi đã bấm giờ.
- `--summary` so float32 và mixed với float64, cả ở điểm dừng lẫn ở cùng số vòng.
- Ghim `TOL = 1e-12` để tái lập đúng số của báo cáo, dù mặc định của `irls` đã đổi.
- Tự chuyển stdout sang UTF-8, vì console Windows mặc định cp1252 không in được tiếng Việt.

## 4. Dữ liệu và tài liệu

- `results/irls_dtype/{huber,mae}_{float32,mixed,float64}.{json,npy}`. MAE float64 chỉ chạy 200 vòng (khoảng 51 phút); đủ 1000 vòng sẽ mất khoảng 4,2 giờ.
- `docs/BAO_CAO_IRLS_DTYPE.md`: báo cáo 11.2 (thời gian, κ, kết quả, ngưỡng dừng, ước lượng cho cả grid, câu hỏi về GPU). Mục "Việc tiếp theo" đã đánh dấu phần `tol` và λ là xong.
- `README.md`: thêm dòng `scripts/` vào sơ đồ, và một gạch đầu dòng quy ước λ cho MAE/Huber.

## 5. Còn để ngỏ

- `docs/CHECKLIST_CON_LAI.md` chưa cập nhật. Các mục có thể đánh dấu:
  - "Cài IRLS cho MAE / Huber" và hai test của outline: đã có từ trước phiên này.
  - "Chạy IRLS bằng PyTorch trên GPU và đo thời gian chạy thật": làm trong phiên này.
- Chưa có `runner.py`. Khi viết nên:
  - dùng `gram_dtype=torch.float32`;
  - chọn lưới λ riêng cho MAE và Huber theo quy ước ở mục 1.5;
  - chia khối theo H cho ETTm1 và H = 720, vì không vừa 4 GB.

---

# Nhật ký thay đổi code: đo trên RTX 5060 Ti (30/09/2026)

Phạm vi: `NHIEM_VU_5060TI.md`, đo tốc độ, bộ nhớ và warm start của IRLS trên máy mới. Kết quả ở [BAO_CAO_5060TI.md](BAO_CAO_5060TI.md).

Test: đầu phiên 33 test, 1 lỗi nạp (`test_sgd`, xem mục 6.3) → cuối phiên **35, tất cả đạt**.

## Tóm tắt

| File | Loại | Nội dung |
|---|---|---|
| `src/solvers.py` | sửa | `irls`: thêm tham số `chunk=8`, chuyển thẳng cho `weight_gram` |
| `src/sgd.py` | khôi phục | đưa lại `lr_factor` và tham số `lradj` như ở commit d4173f1 |
| `scripts/bench_5060ti.py` | mới | đo chunk, chạy đầy đủ, từng ô, warm start khi quét λ, ước lượng grid |
| `results/bench_5060ti/` | mới | `chunk`, `full_{mae,huber}` (+ `.npy`), 12 file `cell_*`, `warmstart_{mae,huber}`, `grid_estimate` |
| `requirements.txt` | sửa | ghim `torch==2.14.0`, hướng dẫn cài bản CUDA trước; thêm `scipy` |
| `docs/BAO_CAO_5060TI.md` | mới | báo cáo |

Không đổi: toán của `irls`, `MAE_DELTA`, quy ước λ, các ngưỡng test, `docs/CHECKLIST_CON_LAI.md`.

## 6.1. `irls(..., chunk=8)`

- Chữ ký mới: `irls(Xt, Y, delta, W0, lam=0.0, pen=None, constrained=False, max_iter=1000, tol=1e-9, callback=None, gram_dtype=None, chunk=8)`.
- Chỉ chuyển xuống `weight_gram(Xg, w, chunk)`. Mặc định 8 giữ hành vi cũ.
- Trên RTX 5060 Ti nên dùng `chunk=16` (báo cáo mục 3.1).

## 6.2. `scripts/bench_5060ti.py`

- **Các bước:** `--step chunk | full | ettm1 | cell --data D --H H | warmstart [--loss mae|huber]`, và `--summary`. `--chunk` ghi đè chunk; mặc định đọc `best_chunk` từ `chunk.json`.
- **Chế độ và thiết lập:**
  - Luôn chạy chế độ pha.
  - Tắt TF32 ngay lúc import.
  - Mọi JSON có khối `hardware`: GPU, torch, CUDA, cuDNN, driver, OS, trạng thái TF32.
- **Chọn chunk:** lấy chunk nhỏ nhất có thời gian một vòng trong khoảng 2% so với nhanh nhất. Chênh lệch giữa các chunk cỡ nhiễu đo, còn bộ nhớ tăng tuyến tính theo chunk.
- **`full`:** so MSE/MAE test với `results/irls_dtype/*_mixed.json` ở cùng số vòng, ghi `pass_1e-6`.
- **`cell`:** tự giảm chunk một nửa khi OOM.
- **`warmstart`:**
  - Khởi tạo lạnh bằng `fit_with_bias(ridge_pen(2δλ·N⁻¹))`, theo mục 1.5. Với MAE, cách này thực chất là OLS.
  - Tự mở rộng lưới tối đa 2 giá trị nếu λ* ở đầu mút trên.
- **`--summary`:** ghi `grid_estimate.json`, gồm nội suy s/vòng ≈ a + b·n·H trên 4 ô góc, và chi phí quét λ.

## 6.3. `src/sgd.py`: khôi phục `lr_factor`

- **Lỗi:** commit 86c012c ("add irls") đã thay `lr_factor` và `lradj` bằng `StepLR`, nhưng `tests/test_sgd.py` (từ d4173f1) vẫn import `lr_factor`. Test này không nạp được, trên cả venv CPU cũ.
- **Sửa:** theo yêu cầu người dùng, lấy lại `src/sgd.py` của d4173f1 (`git checkout 86c012c~1 -- src/sgd.py`).
- **Hệ quả:** `train_sgd` lại mặc định `lradj="type1"`, đúng lịch lr của LTSF-Linear. Lịch cũ của notebook là `lradj="step"`.

## 6.4. Môi trường

- **Venv mới `.venv-gpu`:** torch 2.14.0+cu130, có sm_120. Không commit, vì venv tự sinh `.gitignore`.
- **`requirements.txt`:** thêm ghi chú phải cài torch từ index CUDA trước. `torch==2.14.0` khớp bản `+cu130` (PEP 440 bỏ qua nhãn local), nên `pip install -r` không cài đè bằng bản CPU; đã kiểm bằng `--dry-run`.

---

# Nhật ký thay đổi code: nhiệm vụ 2, tiêu chí dừng, ba phép kiểm, `runner.py` (30/09/2026)

Phạm vi: `docs/NHIEM_VU_2.md`. Kết quả ở [BAO_CAO_TIEU_CHI_DUNG.md](BAO_CAO_TIEU_CHI_DUNG.md).

## 7.1. `irls(..., patience=1)`

- **Chữ ký mới:** thêm `patience=1` vào cuối (sau `chunk`).
- **Cách dừng:** chỉ dừng khi J giảm tương đối ít hơn `tol` trong `patience` vòng liên tiếp. Một vòng giảm ≥ `tol` đặt lại bộ đếm về 0.
- **Lý do:** khi warm start từ nghiệm của λ liền trước, vòng đầu J giảm rất ít dù chưa hội tụ. Với `patience = 1`, MAE dừng ngay sau 1 vòng ở λ = 1, 3, 10 (BAO_CAO_5060TI mục 5).
- **J tăng trong ngưỡng `rise_tol`** (chạm độ chính xác của kiểu số): vẫn dừng ngay, không chờ `patience`. Trước đây nhánh này rơi vào phép so `< tol` vì mức giảm âm; giờ tách thành phép kiểm `J_new > J_old` riêng.
- **Mặc định `patience = 1`** cho đúng hành vi cũ, kể cả W và số vòng (test `test_patience_one_is_default` so bằng `torch.equal`).
- Với `tol = 0`, `irls` chạy đến `max_iter`, trừ khi J tăng trong ngưỡng. Dùng cho nghiệm tham chiếu.

Test mới trong `tests/test_irls.py`, lớp `TestPatience`:

| Test | Kiểm gì | Đo được |
|---|---|---|
| `test_patience_one_is_default` | `patience=1` cho cùng W và cùng số vòng như khi không truyền | trùng từng bit |
| `test_warm_start_does_not_stop_after_one_iteration` | MAE, warm từ nghiệm λ = 1 sang λ = 1,1, `tol = 1e-6`: `patience = 5` chạy nhiều vòng hơn `patience = 1` và J không lớn hơn | 1 vòng → 5 vòng |

## 7.2. Test mới cho Phần B (`tests/test_irls.py`)

| Lớp / test | Kiểm gì | Đo được → ngưỡng |
|---|---|---|
| `TestDLinearEquivalence.test_huber` | DLinear giải trực tiếp trong 2L chiều (Z = [X Pᵀ, X (I − P)ᵀ] + cột 1, `pen = I` cỡ 2L) cho W_eff và bias như L chiều với `pen = N⁻¹`; L = 24, k = 5, n = 400, λ = 5 | W_eff 1e-15, bias 0 → 1e-8 |
| `TestDLinearEquivalence.test_mae` | như trên với MAE: hàm mục tiêu có phạt (L chiều) của hai nghiệm, so hai chiều; và với phạt 2L chiều của chính nghiệm 2L | 0 → 1e-6 |
| `TestNLinearMAE.test_matches_constrained_lp` | NLinear MAE (`constrained=True`) so với `linprog` có `A_eq`: MAE không kém quá 1e-6 tương đối, tổng hàng = 1 tới 1e-10, và nghiệm khác MAE không ràng buộc | đạt |

- Hai dãy lặp IRLS (2L chiều và N⁻¹) trùng nhau từng vòng trong số học chính xác, vì mỗi vòng là một bài ridge có trọng số. Vì vậy test này kiểm phép tương đương và cách cài `pen`, không kiểm hội tụ.
- Để test không đạt tầm thường, `check_penalty_matters` đòi nghiệm phạt N⁻¹ khác nghiệm phạt I hơn 1e-2 tương đối (đo được khoảng 0,11).
- Test: 37 → **40, tất cả đạt**.

## 7.3. Script mới

| File | Nội dung |
|---|---|
| `scripts/stopping_check.py` | A.2 (nghiệm tham chiếu, `tol = 0`), A.3 (quét λ lạnh/warm theo `tol`, `patience`), A.4 (ba tiêu chí → `acceptance.json`). Bỏ qua file đã có, `--force` để chạy lại |
| `scripts/checks_mae.py` | B.1 (ba điểm khởi tạo), B.2 (2L chiều so với N⁻¹ trên dữ liệu thật; nếu 2L lỗi thì ghi chẩn đoán trị riêng của A_0 và vài vòng với A_h float64) |
| `scripts/runner.py` | **soạn, chưa chạy**: đường λ của cả grid → `results/lambda_path.csv` và `results/weights/`; `STOP` để trống chờ quyết định Phần A |
| `scripts/select_lambda.py` | **soạn, chưa chạy**: `lambda_path.csv` → `results/results.csv` (hai cách chọn λ*) |
| `scripts/bench_5060ti.py` | thêm `git_commit()` và `save_json()`, ghi `git_commit` vào mọi JSON; `run_irls` nhận `patience`, `constrained`, `quiet`; thêm cột `converged` |
| `.gitignore` | thêm `results/weights/` (W của cả grid khoảng 1,9 GB) |

## 7.4. Sau khi người dùng quyết định

- **Phần A, phương án (a′):** `stopping_check.py --tol0` cho `tol` riêng ở λ = 0 (`EXTRA_MAE_CONFIGS`). Cấu hình `tol = 1e-10, patience = 1`, λ = 0: `3e-11` đạt A.4 và được điền vào `STOP` của `runner.py`.
- **B.2, MAE:** thêm `checks_mae.py --check dlinear_f64`, so 2L chiều và N⁻¹ ở cùng số vòng, mọi thứ float64.
- **NLinear trong `runner.py`:** đổi từ `pen = I` sang `pen = diag(1, …, 1, 0)`, tức weight decay thật của NLinear, giống notebook 01.
  - MSE của NLinear giờ là dạng đóng (ridge trên `nlinear_transform`), nên bỏ nhánh `irls` δ lớn.
  - Lý do: với `pen = I`, λ* của NLinear luôn bằng 0 (BAO_CAO_TIEU_CHI_DUNG mục 6).
- **README:** thêm mục "Chạy grid" và quy ước phạt của NLinear.
- **Test mới** `TestPenalty.test_nlinear_penalty_equals_transform_ridge`: NLinear ràng buộc với `pen = diag(1, …, 1, 0)`, δ lớn, bằng ridge trên `nlinear_transform`. Đo được khoảng 1e-15, ngưỡng 1e-10. Test cũng đòi nghiệm khác phạt I. Test: 40 → **41, tất cả đạt**.

## 7.5. Còn để ngỏ (trước khi quyết định; giữ lại để đối chiếu)

- ~~**Phần A không đạt A.4**~~ **Đã xử lý** bằng phương án (a′), mục 7.4.: chỉ trượt ở MSE val tại λ = 0, lệch 1,04e-5–1,1e-5 ở `tol = 1e-10`. Chờ người dùng chọn phương án (BAO_CAO_TIEU_CHI_DUNG mục 1), rồi điền `STOP` trong `runner.py`.
- ~~**B.2 với MAE trên dữ liệu thật**~~ **Đã xử lý** bằng phép so float64 ở cùng số vòng (`dlinear_f64`), mục 7.4.: cách giải 2L chiều lỗi `J tăng` ở chế độ pha (A_h suy biến, sai số float32 cỡ phạt 2λδ). Chưa chạy bản float64 vì mất khoảng 75 phút.

## 7.6. `irls(..., gram_fallback=True)`: fallback float64 theo từng h

- **Lỗi:** khi chạy grid, ETTh2 H = 720, Linear MAE λ = 0 báo `RuntimeError: J tăng ở vòng 16: 5,075604e-07 → 5,077229e-07`.
- **Chẩn đoán** (BAO_CAO_TIEU_CHI_DUNG mục 8):
  - Chỉ 1 trong 720 bước h hỏng: h = 3. Ở đó κ(A_3) = 8,3e7, nên κ·ε₃₂ ≈ 5.
  - A_3 lập bằng float32 không còn xác định dương, J_3 tăng 75%. Lập bằng float64 thì J giảm bình thường.
- **Sửa:** chỉ ở chế độ pha (`gram_dtype` khác kiểu của Xt). Sau mỗi bước, tính J_h theo từng h. Bước h nào có J_h tăng quá `rise_tol` thì lập lại A_h đó bằng kiểu của Xt (float64) và giải lại.
  - J tách theo h, nên các h khác giữ nguyên. Toán của IRLS không đổi.
  - Chi phí: khoảng 0,04 s cho mỗi h hỏng ở ETTh2 H = 720, so với 26,6 s nếu lập cả 720 A_h bằng float64.
- **Mặc định bật.** Khi không có bước h nào hỏng, kết quả trùng từng bit với `gram_fallback=False`, vì J tính theo đúng công thức cũ.
- **`irls.last_fallbacks`** ghi số bước h đã lập lại trong lần gọi gần nhất. `runner.py` in số này ra log, nhưng không thêm cột vào CSV để giữ header cũ.
- **Test mới (`TestGramFallback`):**
  - Hai cột gần cộng tuyến: không fallback thì báo J tăng ở vòng 48; có fallback thì MAE khớp float64 tới khoảng 3e-13.
  - Dữ liệu điều kiện tốt: kết quả trùng từng bit.
  - Test: 41 → **43, tất cả đạt**.
- **Lưu ý:** các dòng grid đã chạy trước khi sửa không bị lỗi, nhưng có thể đã có bước h đi sai rồi tự hồi phục mà không báo, vì tổng J vẫn giảm. Chưa kiểm.

## 7.7. Fallback lập A_h theo khối nhỏ hơn

- **Lỗi:** ETTm1 H = 720, Linear MAE λ = 0 chạy hơn 79 phút (ước lượng khoảng 37 phút).
  - Tiến trình dùng 14,5 GB VRAM cộng 1,3 GB bộ nhớ dùng chung, tức đã tràn sang RAM hệ thống.
  - Nguyên nhân: fallback float64 gọi `weight_gram` với cùng `chunk = 16`, nên tensor trung gian n·16·p·8 byte khoảng 10 GB ở ETTm1. Ở ETTh2 chỉ khoảng 2,3 GB nên không lộ ra.
- **Sửa:** fallback dùng `fb_chunk = chunk · (cỡ kiểu gram)/(cỡ kiểu Xt)`, tức 8 với float32/float64. Tensor trung gian của fallback vì vậy không lớn hơn của bước thường. Toán không đổi.
- **Đo trên ETTm1 H = 720:** bước float32 chunk 16 thêm 5,95 GiB; fallback float64 chunk 8 (40 bước h) thêm 4,75 GiB. Test 43/43 đạt.
- Đã dừng runner (mất phần đang chạy dở của Linear MAE λ = 0) và chạy tiếp.
- **Sau khi sửa vẫn tràn 1,36 GB.** Lần chạy lại vẫn dùng 14,4 GB VRAM cộng 1,36 GB bộ nhớ dùng chung, ngay từ lần chạy đầu. Hai nguyên nhân thêm:
  1. Phép tính J_h theo h của fallback tạo thêm 3–4 tensor tạm cỡ n·H float64, mỗi cái khoảng 1,3 GB. Sửa: `objective(per_h=True)` tính theo khối 64 cột. J vẫn tính như cũ, nên test trùng từng bit vẫn đạt.
  2. Phân mảnh bộ nhớ: cấp phát thật 11,6 GiB nhưng PyTorch dự trữ tới 15,2 GiB. Sửa trong `runner.py`:
     - đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` trước khi import torch;
     - dùng `chunk = 8` cho ô có n·H > 1e8 (chỉ ETTm1 H = 720). Đo trên ETTm1 H = 720: dự trữ 14,05 GiB, 5,0 s/vòng (chunk 16: 4,8 s/vòng). Cột `chunk` của CSV ghi đúng giá trị đã dùng.

---

# Nhật ký thay đổi code: nhiệm vụ 3, SGD trên ETTh2, kiểm giả thuyết, ba việc rẻ (02/10/2026)

Phạm vi: `docs/NHIEM_VU_3.md`. Kết quả ở [BAO_CAO_ETTH2_SGD.md](BAO_CAO_ETTH2_SGD.md).

## 8.1. `src/models.py`: Linear, NLinear, `effective_weights`

- **Lớp mới `Linear`, `NLinear`:** chép đúng `models/Linear.py`, `models/NLinear.py` của LTSF-Linear với `individual=False`.
  - NLinear giữ `nn.Linear(L, H)` đủ L cột (cột cuối nhân với x_L − x_L = 0, không có gradient), để khởi tạo tiêu thụ cùng số ngẫu nhiên như repo gốc.
- **`MODELS`:** từ điển tên → lớp, cho cả ba mô hình.
- **`effective_weights(model)`:** trả (W_eff [H, L], b [H]) float64.
  - DLinear: W_eff = W_t P + W_s (I − P), b = b_t + b_s.
  - NLinear: `nlinear_effective(W[:, :-1])`, mỗi hàng cộng bằng 1.
- **Test mới (`tests/test_models.py`):** X W_effᵀ + b trùng forward của mô hình tới 1e-12 (float64, trọng số đã xáo khỏi khởi tạo); NLinear tổng hàng bằng 1; Linear và NLinear cùng khởi tạo với `nn.Linear` khi cùng seed. Test: 43 → **46, tất cả đạt**.
- `src/sgd.py` **không đổi**: `train_sgd` đã có đủ lr, epoch, patience, `lradj`; batch size và `drop_last` đặt ở DataLoader.

## 8.2. `scripts/runner.py`

- **Lưới MSE:** `MSE_LAMS` = {0} ∪ logspace(−2, 5, **141**) (20 giá trị mỗi bậc, C.2). Lưới cũ 36 giá trị là tập con (đã kiểm theo `lam_key`), nên chạy lại chỉ thêm 105 λ mới mỗi đường.
- **Cột `n_fallback`:** số bước h phải lập lại A_h bằng float64 (`irls.last_fallbacks`), ghi cho mỗi dòng IRLS.
  - `Writer` mở file đã có với **đúng header của nó** và `extrasaction="ignore"`. `lambda_path.csv` vì vậy giữ header cũ (không có cột này); file mới (`results/recheck/`) có cột.
- **`Cell.set_train(X, Y)`:** tách phần tính x̄, ȳ, A, C (và bản NLinear) ra khỏi `__init__`, để dùng lại với train đã bỏ bớt cửa sổ (C.3). Kết quả không đổi (cùng phép tính).

## 8.3. Script mới

- `scripts/sgd_etth2.py`: `--step a2` (tái lập ETTh1 H = 96, seed 42) và `--step run` (ETTh2, 3 mô hình × 4 H × 3 seed). CPU. Đánh giá bằng W_eff, bias float64 trên cùng ma trận cửa sổ với nghiệm dạng đóng; kiểm số cửa sổ và kiểm chéo với forward float32.
- `scripts/recheck_paths.py`: C.1, chạy lại ba đường MAE vào `results/recheck/`, so với `lambda_path.csv`.
- `scripts/ridge_w0.py`: B.2, ridge co về W₀ ∈ {0, 1/L, lag cuối}.
- `scripts/constant_windows.py`: C.3, cửa sổ hằng của ETTh2.
- `scripts/gd_early_stop.py`: B.3, GD toàn batch lr cố định trên Linear.
- `scripts/report_etth2.py`: lập các bảng của báo cáo.

## 8.4. Sửa sau khi chạy

- **`scripts/ridge_w0.py` (commit `2fdebb9`):** lần chạy đầu gọi `runner.Cell.evaluate`, hàm này tính cả phần dư trên train (n·H) cho mỗi λ. Sau gần 1 giờ mới xong 44/72 tổ hợp, chưa tới ETTm1. Đã dừng lần chạy đó.
  - Thay bằng `evaluate` riêng: chỉ MSE, MAE của val và test, đúng biểu thức của `Cell.evaluate`. Kiểm `zero` vẫn khớp `lambda_path.csv` tới 2,2e-16.
  - Mỗi ô ghi `results/ridge_w0/cells/{ds}_H{H}.csv` (ghi file tạm rồi đổi tên), nên chạy lại là chạy tiếp. Cả 12 ô mất khoảng 50 phút.
- **`scripts/report_etth2.py` (commit `72d5611`):** glob `*_H*_seed*.json` bắt nhầm cả JSON của A.2. Đổi thành `*Linear_H*_seed*.json`.
- **`.gitignore`:** thêm `results/sgd_etth2/*.npy` (32 MB) và `results/recheck/weights/` (47 MB), giống `results/weights/`. Các W này tái lập được trùng từng bit bằng script.

---

# Nhật ký thay đổi code: nhiệm vụ 4, chốt các câu hỏi về ETTh2 (03/10/2026)

Phạm vi: `docs/NHIEM_VU_4.md`. Kết quả ở [BAO_CAO_ETTH2_2.md](BAO_CAO_ETTH2_2.md).

## 9.1. `src/sgd.py`: huấn luyện trên train đã lọc

- **`MaskedETTDataset(data, L, H, keep)`:** như `ETTDataset`, trả thêm `keep [C]` (bool) cho mỗi cửa sổ.
- **`masked_mse(pred, y, keep)`:** MSE trung bình trên các phần tử của cặp (cửa sổ, kênh) có `keep = True`.
- **`train_one_epoch`:** batch 3 phần tử thì dùng `masked_mse`; batch 2 phần tử chạy đúng như cũ. Seed 2021–2023 của nhiệm vụ 3 chạy lại trùng từng bit (36/36 file W).
- **Lý do chọn mask thay vì bỏ mẫu:** mỗi mẫu của SGD gồm cả 7 kênh, còn cửa sổ hằng tính theo từng kênh. Mask giữ nguyên batch, thứ tự xáo và khởi tạo, và tương đương bỏ đúng các hàng đó khỏi hồi quy dạng đóng.
- **Test mới (`tests/test_sgd.py`):** giữ hết thì bằng `MSELoss`; bỏ cặp thì bằng MSE trên các hàng còn lại, và giá trị ở cặp bị bỏ không ảnh hưởng; dataset trả đúng `keep`. Test: 46 → **49, tất cả đạt**.

## 9.2. `scripts/sgd_etth2.py`

- **`--out`:** thư mục con của `results/` (mặc định `sgd_etth2`), để không ghi đè kết quả nhiệm vụ 3.
- **`--filtered`:** train đã lọc theo `constant_windows.constant_mask` (hàm `train_keep`). JSON ghi thêm `filtered_train` và `n_train_pairs_dropped`.

## 9.3. Script mới

- **`scripts/mae_filtered.py`:** MAE λ = 0 trên train đã lọc (`Cell.set_train`), khởi tạo từ MSE λ = 0 trên cùng train, cấu hình dừng như grid. Ghi số vòng, số bước fallback, thời gian.
- **`scripts/official_ltsf.py`:** `--prepare` chép code gốc sang `scratch/ltsf_official/` và áp `results/official_ltsf/patch.diff`; chạy code gốc, lưu log, parse MSE/MAE và epoch tốt nhất, chấm chéo checkpoint bằng pipeline của repo.
  - **Lỗi đã sửa (commit `926efb2`):** `git apply` chạy với cwd là `scratch/ltsf_official` (thư mục con của repo này) hiểu đường dẫn theo gốc repo và **lặng lẽ bỏ qua** patch. Giờ áp từ gốc repo với `--directory=scratch/ltsf_official` và kiểm hai dấu hiệu của patch.
- **`scripts/report_etth2_2.py`:** lập mọi bảng của báo cáo; kiểm tất định của seed 2021–2023.
- **`.gitignore`:** thêm `scratch/`.

---

# Nhật ký thay đổi code: nhiệm vụ 5, trình tối ưu trên ETTh2 và SGD 20 seed cho B1, B4 (06–07/10/2026)

Báo cáo: [BAO_CAO_SGD_B1_B4.md](BAO_CAO_SGD_B1_B4.md). Không đổi `src/`, không đổi test, không đổi toán của `irls`.

## 10.1. `scripts/sgd_etth2.py` (commit `33e6101`)

- **`--dataset {ETTh2, ETTh1, ETTm1}`** (mặc định ETTh2): chọn file dữ liệu, ô dạng đóng (`load_cell`) và siêu tham số trong bảng `HPS`, đọc từ `scripts/EXP-LongForecasting/Linear/{etth1,etth2,ettm1}.sh` dòng 24/38/52/66 và `run_longExp.py` dòng 66/68/72 của LTSF-Linear. JSON ghi `dataset` thật thay vì cố định "ETTh2".
- **`--threads`:** `torch.set_num_threads` (mặc định không đặt). JSON ghi thêm `num_threads`.
- **Kiểm:** với ETTh2, seed 2021–2023 × 3 mô hình × H ∈ {96, 720} chạy lại cho **18/18 file W trùng từng byte** với `results/sgd_etth2_20seed/`.

## 10.2. Script mới

- **`scripts/optim_etth2.py`** (Phần A; commit `3c915fc`, sửa ở `8f5b9a3`, `fe1f5aa`): GD toàn batch (`gd_zero`, `gd_init`, lr = 1/λ_max) và Adam toàn batch (`adam_full`) trên Linear ETTh2 H = 720, float64 trên CPU qua ma trận Gram. λ_max bằng power iteration, kiểm chéo bằng `eigvalsh`.
  - **Lỗi đã sửa (`8f5b9a3`):** `ridge_path` đã trả W dạng [H, L], code cũ chuyển vị thêm lần nữa nên phép so ‖W_gd − W_ridge(λ)‖ lỗi kích thước. Lỗi làm script dừng, không tạo ra số sai.
  - **`fe1f5aa`:** `gd_zero` (n = 1) không có CI nên không gán nhãn ba trường hợp, chỉ ghi dấu.
- **`scripts/report_sgd_b1_b4.py`** (commit `27e393c`, `a4fde50`): bảng B1, B4 (kèm đếm nhãn), C.3 và hình `H3_sgd_vs_optimum.png`. Số công bố lấy từ mục 0.1 của `NHIEM_VU_3.md`; nhãn ba trường hợp dùng chung hàm `label` của `optim_etth2.py`.
- **`.gitignore`:** thêm `results/optim_etth2/*.npy`, `results/sgd_b1_b4/**/*.npy`.

## 10.3. Ghi chú môi trường

- Mọi lần chạy và report dùng `.venv-gpu` (torch 2.14.0+cu130, như `BAO_CAO_5060TI.md`). Python hệ thống và `.venv` có torch 2.6.0+cpu (cài từ 29/09, không đổi); chạy script bằng chúng sẽ dừng ở `bench_5060ti.hw_info` vì cần CUDA.
- Test (`.venv-gpu/Scripts/python -m unittest discover -s tests -t .`): **49, tất cả đạt** (07/10/2026).
- 107 JSON ETTm1 viết từ 12:33 đến 13:12 ngày 06/10 có `git_commit` rỗng, nguyên nhân chưa xác định; HEAD lúc đó là `a4fde50` (xem mục 2.3 của báo cáo).

---

# Nhật ký thay đổi code: nhiệm vụ 6, thí nghiệm outlier và hai phép kiểm SGD (07/10/2026)

Báo cáo: [BAO_CAO_OUTLIER.md](BAO_CAO_OUTLIER.md). Không đổi toán của `irls`, không đổi cấu hình dừng, δ hay lưới λ.

## 11.1. `src/outliers.py` (mới) và `tests/test_outliers.py` (commit `de98749`)

- **`contaminate_series(train_series, rate, magnitude=10, seed, protect_last, L=None, H=None)`:** chọn đúng ⌊rate × (T − protect_last) × C⌋ điểm (hàng, kênh) không lặp trong vùng được chèn, cộng ±magnitude (dấu ngẫu nhiên). Trả chuỗi mới và `n_contaminated`, `n_eligible`, `mask`; nếu cho L, H thì thêm `frac_windows_x`, `frac_windows_y`. Không sửa mảng đầu vào.
- **`contaminate_targets(Y, rate, magnitude=10, seed)`:** như trên nhưng trên các phần tử của Y train; x giữ nguyên.
- **11 test mới:** số điểm, vùng cấm, độ lớn đúng ±10 và chỉ điểm được chọn thay đổi, cùng/khác seed, rate = 0 trả đúng dữ liệu gốc, không sửa đầu vào, tỷ lệ cửa sổ; trên ETTh1: ghép train đã chèn vào chuỗi đầy đủ rồi cắt lại thì cửa sổ val, test trùng từng byte (H = 96, 720), và phản chứng `protect_last = 0` làm val đổi. Test: 49 → **60, tất cả đạt**.

## 11.2. `scripts/runner.py`

- **`--out`:** ghi vào `results/<out>/lambda_path.csv` và `results/<out>/weights/` thay vì `results/`.
- **`--outlier-mode {series, target}`, `--outlier-rate`, `--outlier-seed`** (cần `--out`): `Cell.contaminate` thay train bằng bản có outlier qua `Cell.set_train`; val, test giữ nguyên. Thêm cột `outlier_mode`, `outlier_rate`, `outlier_seed`, `n_contaminated`, `frac_windows_x`, `frac_windows_y`; khóa chạy tiếp lọc theo (mode, rate, seed); tên file W có tiền tố cấu hình.
- Không có `--outlier-mode` thì chạy y như trước. Kiểm: rate = 0 (series) cho 154/154 dòng trùng tuyệt đối `results/lambda_path.csv` (Linear ETTh1 H = 96, MSE và MAE).

## 11.3. `scripts/select_lambda.py`

- **`--in`, `--out`** (mặc định như cũ). File có cột outlier thì mỗi (mode, rate, seed) là một nhóm riêng và các cột đó được giữ. Kiểm: chạy với mặc định cho `results.csv` trùng từng byte bản cũ.

## 11.4. Phần D (commit `238f89b`)

- **`scripts/official_ltsf.py`:** `--dataset ETTm1` (lr 0,0001, batch 8 theo `Linear/ettm1.sh`; tự chép `data/ETTm1.csv` vào `scratch/ltsf_official/dataset/`), `--out`, `--cpu` (đặt `CUDA_VISIBLE_DEVICES` rỗng cho tiến trình con, vì `--use_gpu` của code gốc là `type=bool` nên không tắt được bằng tham số). Checkpoint ETTm1 lưu với tiền tố dataset. Mặc định (ETTh2) không đổi.
- **`scripts/sgd_etth2.py`:** `--lr` thay lr của script chính thức; mặc định không đặt. Kiểm: ETTh2 Linear H = 96 seed 2021 vẫn trùng byte `results/sgd_etth2_20seed/`.

## 11.5. Khác

- **`scripts/report_outliers.py`** (mới): mọi bảng và hình H2 của báo cáo.
- **`.gitignore`:** thêm `results/outliers/**/weights/`, `results/outliers/d2_lr/**/*.npy`.

---

# Nhật ký thay đổi code: nhiệm vụ 7, áp dụng góp ý của TA (08/10/2026)

Báo cáo: [BAO_CAO_NV7.md](BAO_CAO_NV7.md). Không đổi toán của `irls`, không đổi cấu hình dừng, `MAE_DELTA` hay lưới λ.

## 12.1. `src/solvers.py`, test (commit `7c59c8c`)

- **`huber_delta_mad(R, k=1,345)`:** trả (δ, σ̂), σ̂ = 1,4826 · median(|R − median(R)|) gộp mọi phần tử, δ = k·σ̂. Hằng `MAD_SCALE`, `HUBER_K`.
- **`tests/test_solvers.py`:** 2 test mới: dữ liệu N(0,2; s²) với s ∈ {0,1; 0,5; 1; 3} cho σ̂ lệch s dưới 5%; thêm 5% phần tử ±50 làm σ̂ đổi dưới 10% (độ lệch chuẩn đổi hơn 10 lần).
- **`tests/test_val_metrics.py` (mới, B.2):** tính lại `val_mse`, `val_mae`, `val_huber` từ W đã lưu (ETTh1 H = 96, ETTh2 H = 336; mọi λ > 0 của MAE, Huber, mỗi λ thứ 10 của MSE; 201 dòng W × 3 cột) và so với `lambda_path.csv` tới 1e-12 tương đối. Phản chứng: số hạng phạt (chuẩn hóa như `train_obj`) nhỏ nhất vẫn bằng 2,3e-8 tương đối, lớn hơn ngưỡng 2·10⁴ lần. Bỏ qua khi thiếu `results/weights/` (bị .gitignore).
- Test: 60 → **63, tất cả đạt**.

## 12.2. `scripts/select_lambda.py`

- **`--selections`** (mặc định `val_objective val_mse` như cũ): thêm `val_mae`.
- **`--in` nhận nhiều file:** một đường λ (cùng khóa nhóm) có ở nhiều file thì lấy trọn đường của file sau cùng; dùng để thay đường Huber δ = 1 bằng δ_MAD. Dừng nếu một đường có λ lặp hoặc trộn nhiều δ.
- Cột `outlier_magnitude` (nếu có) vào khóa nhóm; dòng thiếu cột coi là 10.
- Kiểm: chạy với mặc định cho `results/results.csv` và `results/outliers/results.csv` trùng từng byte bản cũ.

## 12.3. `scripts/runner.py`

- **`--huber-delta mad|<số>`, `--delta-scale`:** δ của Huber theo từng (ô, mô hình, tập train) qua `Cell.huber_delta` (phần dư train của `closed_form(mô hình, 0)` trên tập train hiện tại, kể cả tập đã chèn outlier). δ vào `irls`, `train_obj`, `val_huber` (với dòng Huber), cột `delta`, `lam_convention`; cột mới `delta_rule`, `sigma_hat`. DLinear λ = 0 vẫn chép từ Linear, kèm assert δ hai mô hình trùng nhau.
- **`--skip-mse`:** không chạy đường MSE; λ = 0 của MAE/Huber khởi tạo từ `closed_form(mô hình, 0)` tính lại.
- **`--outlier-magnitude`** (mặc định 10): truyền cho `contaminate_*`; cột `outlier_magnitude` (file mới), vào khóa chạy tiếp và tên file W khi khác 10.
- Không đặt cờ mới thì chạy y như trước. **Kiểm (A.2):** `--huber-delta 1 --skip-mse` cho Linear ETTh1 H = 96: 10/10 dòng trùng `lambda_path.csv` tuyệt đối (chênh 0 ở mọi cột metric), cùng số vòng.

## 12.4. Script mới

- **`scripts/huber_deltas.py`:** δ_MAD, σ̂ và σ̂ theo từng h (min, trung vị, max) cho 88 tập train → `results/huber_mad/deltas.csv`; cột `mask_sha1` để kiểm vị trí outlier trùng nhau giữa các độ lớn; dừng nếu δ ngoài [0,05; 5].
- **`scripts/sgd_lr_tune.py`:** Phần D. Điều phối `sgd_etth2.py --lr` cho 36 ô (chọn lr trên 5 seed, ×8 khi ×4 thắng, chạy cuối 15 seed mới), dùng lại kết quả ×1 có sẵn và Linear ETTm1 ×2, ×4 của nhiệm vụ 6; ghi `results/sgd_lr/selection.json`.
- **`scripts/report_nv7.py`:** mọi bảng và hình của báo cáo.
- **`.gitignore`:** thêm `results/huber_mad/**/weights/`, `results/outliers_mag/**/weights/`, `results/sgd_lr/**/*.npy`.

# Nhật ký thay đổi code: nhiệm vụ 8, dọn repo và báo cáo LaTeX (11/10/2026)

Báo cáo: [reports/BAO_CAO_NV8.md](reports/BAO_CAO_NV8.md). Không đổi toán hay hành vi của `src/`, không chạy thí nghiệm mới (trừ phép kiểm một ô ETTh1 H = 96, ghi vào thư mục tạm), không sửa file nào trong `results/` (chỉ thêm `results/README.md`). Từ mục này, các đường dẫn trong nhật ký là đường dẫn mới; mục 1–12 giữ đường dẫn cũ (tra `DUONG_DAN_CU.md`).

## 13.1. Dời file (commit `069f88c`)

- `git mv` mọi script vào `scripts/{grid,sgd,checks,report}/`, báo cáo vào `docs/reports/`, nhiệm vụ vào `docs/tasks/` (kể cả `NHIEM_VU_5060TI.md` ở gốc), checklist vào `docs/plan/`. Template LaTeX `TEMPLATE/` (chưa track) → `report/template/`. Bảng đầy đủ: `docs/DUONG_DAN_CU.md`.
- Mỗi script: `ROOT = Path(__file__).resolve().parents[1]` → `parents[2]`; `sys.path.insert(0, <thư mục script>)` và `sys.path.insert(0, ROOT / "scripts")` → thêm cả bốn thư mục `scripts/{grid,sgd,checks,report}`, nên import chéo (`from bench_5060ti import …`, `import runner as R`, …) không đổi. `sgd_lr_tune.py` gọi `scripts/sgd/sgd_etth2.py`. Docstring, chú thích đổi `scripts/X.py` → đường dẫn mới, `docs/NHIEM_VU_*`, `docs/BAO_CAO_*` → `docs/tasks/`, `docs/reports/` (cả trong `src/`, `tests/`: chỉ chú thích).
- Kiểm: 63 test đạt; mọi script có argparse chạy `--help` được; chạy lại năm script `report_*.py` cho nội dung `results/` trùng bản cũ (chỉ khác `git_commit`, `date` trong `summary.json`; đã khôi phục bằng `git checkout`).
- `.gitignore`: `.venv*/`, `docs/refs/*.pdf`, `report/build/*` (trừ `main.pdf`), file phụ của LaTeX. `.gitattributes`: `*.sh` và `report/latexmkrc` giữ LF.

## 13.2. Báo cáo LaTeX (`scripts/report/`, `report/`)

- **`report_data.py` (mới):** đọc `results/` cho báo cáo và notebook (B3 theo tiêu chí λ\*_MSE / λ\*_MAE, outlier, B1, B4, C.3, lr đã chọn, ETTh2). Chỉ đọc.
- **`make_numbers.py` (mới):** → `report/generated/numbers.tex`, macro `\nm…` (quy ước tên ở đầu file). **`make_tables.py` (mới):** → `report/generated/tab_*.tex` (booktabs, dấu phẩy thập phân, đậm giá trị nhỏ nhất giữa ba mục tiêu). **`make_figures.py` (mới):** H1–H5 → `report/figures/*.pdf`; gọi đúng `figure_h2_v2`, `figure_sensitivity`, `figure_c` (`report_nv7.py`), `figure` (`report_sgd_b1_b4.py`) và ô vẽ H1 của notebook 01, chặn `Figure.savefig` để ghi PDF (không ghi `results/`), PDF tất định; `capture`, `capture_png` cho notebook.
- **`audit_numbers.py`, `check_latex.py` (mới):** số thập phân ngoài macro trong `report/sections/`; kiểm tĩnh LaTeX (macro, nhãn, trích dẫn, ngoặc, môi trường, file, `\TODO`) vì máy không có TeX.
- **`scripts/checks/compare_cell.py` (mới):** so một ô grid chạy lại với `results/lambda_path.csv` (MSE 1e-10, MAE/Huber 1e-8).
- `report/preamble.tex`: preamble của `report/template/template.tex` chép nguyên, đổi tên file `.bib`, tiêu đề PDF, khối tiêu đề và tác giả (chú thích `NV8`), thêm vài macro.

## 13.3. Notebook

- **`notebooks/02_loss_comparison.ipynb`, `03_outliers_and_sgd.ipynb` (mới):** gọi `report_data.py`, `make_figures.capture_png`; đã chạy, lưu output. **`scripts/report/run_notebooks.py` (mới):** chạy 02, 03 và lưu; `--check` chạy bản sao của notebook cũ.
- **`00_nlinear_closed_form.ipynb`:** cell cuối dùng `pred_V` chưa định nghĩa (`NameError`, đã ghi trong checklist) → `V.sum(axis=1)` (vi phạm tổng hàng của nghiệm NLinear, song song với dòng `W_ols` ngay trên). Xóa output lỗi cũ của cell đó; các cell khác giữ nguyên.
- `.venv-gpu`: cài thêm `nbclient`, `nbformat`, `ipykernel` (không thêm vào `requirements.txt`; README ghi là tùy chọn).

## 13.4. `run_all.sh`, `README.md`

- **`run_all.sh` (mới):** `quick` (venv, torch đúng bản CUDA qua `TORCH_INDEX`, test, chạy lại grid ETTh1 H = 96 vào thư mục tạm và `compare_cell.py`, sinh số/bảng/hình, biên dịch nếu có latexmk); `full` (thứ tự các nhiệm vụ, có `--dry-run`).
- **`README.md`:** viết lại theo NHIEM_VU_8 D.3.

# Nhật ký thay đổi code: nhiệm vụ 9, góp ý Tuần 3 của TA (11/10/2026)

Báo cáo: [reports/BAO_CAO_NV9.md](reports/BAO_CAO_NV9.md). Không sửa file nào đã có trong `results/`; kết quả mới ở `results/sgd_lr_ext/`, `results/official_ltsf_droplast/`, `results/huber_delta_h/`, `results/nv9/` (và `results/official_ltsf/patch_droplast.diff`).

## 14.1. `src/solvers.py`: `irls` nhận δ theo từng bước h (commit `6159f06`)

- `delta` là số thực (như cũ) hoặc vectơ độ dài H. Hàm mục tiêu với vectơ: Σ_h Σ_i ρ_{δ_h}(r_ih)/δ_h + λ Σ_h w_hᵀ Pen w_h; trong IRLS, phạt của hàng h thành 2λδ_h·Pen (A_h, g_h, J_h), J = Σ_h δ_h F_h/(n·H). δ vô hướng đi đúng đường code cũ (`pen_A`, `pen_W` trả `A + Pen`, `W @ Pen` y như trước).
- Trọng số IRLS ở nhánh vectơ tính bằng `reciprocal(max(|r|, δ)) · δ`: torch tính `số / tensor` theo cách này, nên vectơ có mọi δ_h bằng nhau cho đúng trọng số của δ vô hướng (chia tensor/tensor lệch 1 ulp; với MAE trên dữ liệu xấu điều kiện, chênh này thành 4e-10 trên W và lệch 3 vòng).
- **Test mới** (`tests/test_irls.py`, `TestDeltaPerH`; 66 test): δ vô hướng trùng từng bit với bản chép `tests/solvers_before_nv9.py` ở 6 trường hợp (Huber, MAE, λ có Pen, NLinear, chế độ pha có fallback); vectơ đều = vô hướng tới 1e-12 (đo: 0 ở 5/6, 6e-17 ở trường hợp có Pen); δ_h lớn ở mọi h = ridge MSE với λ_h = 2δ_hλ của từng h (có và không ràng buộc NLinear).

## 14.2. Script mới

- **`scripts/sgd/sgd_lr_extend.py`:** Phần A. Đọc 15 ô biên từ `results/sgd_lr/selection.json`, chạy mức kế tiếp (×16, ×32 hoặc ×0,25, ×0,125) bằng `sgd_etth2.py --lr`, tối đa 2 mức, dừng khi mức mới không tốt hơn trên val; lần chạy lỗi (NaN) ghi `*.failed.json`, val > 10 × mức ×1 coi là thất bại; lr đổi thì chạy 15 seed cuối. → `results/sgd_lr_ext/`.
- **`scripts/checks/huber_delta_h.py`:** C.3. Đường λ của Huber với δ_h = 1,345·σ̂_h (`Cell.huber_delta(residual_stats=True)`), như grid (IRLS_LAMS, warm start, pha, tol 1e-9, mở rộng đầu mút) → `results/huber_delta_h/`.
- **`scripts/report/report_nv9.py`:** Phần A (gọi `report_nv7.part_d` với lựa chọn lr đã gộp), B, C.1, C.2 (đoạn hằng ≥ 24 giờ theo kênh), C.3 → `results/nv9/summary.json`, `tables_*.md`.

## 14.3. Script đã có

- **`scripts/sgd/official_ltsf.py`:** `--test-drop-last {0,1}` truyền `--test_drop_last` cho code gốc (patch mới `results/official_ltsf/patch_droplast.diff`, áp sau `patch.diff`; `--prepare` áp cả hai); hậu tố `_dl0/_dl1` cho JSON, log, checkpoint; JSON ghi số cửa sổ test và số cửa sổ bị bỏ. Thêm `--dataset ETTh1`, `--model NLinear`. Không đặt cờ mới thì chạy y như trước. Bản chép `scratch/ltsf_official/` đã áp patch mới.
- **`scripts/report/report_nv7.py`:** `part_d(summ, sel=None, find=None, out=None, mult_cols=…)`; mặc định cho kết quả trùng tuyệt đối với `results/nv7/` (đã kiểm: `D_B4`, `D_C3`, `tables_D.md`).
- **`scripts/report/report_data.py`:** B4, C.3 tinh chỉnh và lr đã chọn đọc từ `results/nv9/` và lựa chọn đã gộp (`report_nv9.merged_selection`); thêm `nv9()`.
- **`make_numbers.py`:** nhóm macro mới `Drop`, `Off`, `Ridge`, `Stuck`, `DeltaH`; `Tuned` đọc nv9; thời gian chạy mới. **`make_tables.py`:** mã nhãn T1–T6 (bảng B4 đầy đủ), cột lr ×0,125 … ×32, năm bảng mới (`tab_droplast`, `tab_ridge`, `tab_etth2filt`, `tab_runs`, `tab_deltah`). **`make_figures.py`:** dấu phẩy thập phân trên trục (chặn `ScalarFormatter`), hình mới H6 (đoạn hằng của ETTh2).
- **`report/preamble.tex`:** `\DeclareLanguageMapping{vietnamese}{english}` (chuỗi biblatex); tắt phép so thứ tự trích dẫn của biber (backend bibtex luôn báo "undefined references"); `\thesection` không kèm dấu chấm (tiêu đề và mục lục thêm dấu chấm), hết "IV.3..".
- **`run_all.sh`:** kỳ vọng 66 test; `full` thêm bước F (Phần A, B, C.3) và `report_nv9.py`.
