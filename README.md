# dlinear-objective-lab

Bài LTSF-Linear (Zeng et al., 2023) cho thấy các mô hình tuyến tính Linear, DLinear, NLinear dự báo chuỗi thời gian dài tốt hơn nhiều mô hình Transformer, nhưng mọi so sánh đều qua huấn luyện bằng SGD. Đề tài này so ba mô hình **tại nghiệm tối ưu tất định**: nghiệm dạng đóng cho MSE (OLS, ridge, NLinear có ràng buộc) và IRLS trên GPU cho MAE, Huber. Ma trận thí nghiệm gồm 3 mô hình × 3 mục tiêu × {λ = 0, λ\*} × 3 dataset (ETTh1, ETTh2, ETTm1) × 4 horizon, cộng outlier nhân tạo và so sánh với SGD nhiều seed. Ba câu hỏi:

- **RQ1:** tại nghiệm tối ưu, khi nào DLinear và NLinear thu về Linear, và khi nào thì không?
- **RQ2:** hàm mục tiêu (MSE, MAE, Huber) ảnh hưởng thế nào đến sai số test, và mức bền vững với outlier ra sao?
- **RQ3:** SGD kém nghiệm tối ưu (ridge ở λ\*) bao nhiêu, và khoảng cách thay đổi thế nào theo dataset, horizon?

## Ba kết quả chính

Số lấy từ `report/generated/numbers.tex` (sinh tự động từ `results/`); chi tiết ở báo cáo.

| | Kết quả |
|---|---|
| RQ1 | DLinear trùng Linear khi λ = 0 và chỉ tách khi có weight decay (ridge trong metric N⁻¹). Ở λ\*, MSE test của DLinear và Linear chênh không quá **0,0006** trên cả 12 cặp (dataset, H). NLinear (ràng buộc tổng hàng bằng 1) tách ở mọi λ, lợi rõ trên ETTh2 (MSE test −0,029 đến −0,348 so với Linear). |
| RQ2 | Huấn luyện bằng MAE cho MSE test thấp hơn MSE ở **30/36** ô, thấp nhất trong ba mục tiêu ở 24/36. Huber (δ = 1,345·σ̂_MAD) đứng giữa MSE và MAE ở 28/36 ô. Outlier chỉ ở đích: MAE, Huber đổi MSE test không quá 0,003, MSE tăng tới 0,014; outlier trong chuỗi ở 10% và ±10σ: MSE suy giảm ít nhất. |
| RQ3 | Sau khi tinh chỉnh learning rate (lưới mở rộng ở mọi ô chọn mức đầu mút), SGD vẫn kém nghiệm dạng đóng trên test ở **27/36** ô (CI 95%). "SGD tốt hơn thật" chỉ ở 5 ô, đều ETTh2 với Linear/DLinear; không giá trị λ nào của ridge, kể cả λ chọn theo test, đạt mức SGD ở đó. DLinear tốt hơn Linear khi huấn luyện bằng SGD: 4/12 ô với lr gốc (đều ETTm1), **0/12** khi tinh chỉnh lr cho cả hai. |

## Cấu trúc

```
├── README.md, run_all.sh, requirements.txt, .gitignore
├── src/                 thư viện: data, operators, solvers (OLS, ridge, NLinear, IRLS), models, sgd, metrics, outliers
├── tests/               66 test (dữ liệu giả, kiểm hồi quy số của notebook, metric val không chứa phạt, δ theo bước h)
├── scripts/
│   ├── grid/            runner.py (grid), select_lambda.py (λ*), huber_deltas.py (δ_MAD)
│   ├── sgd/             sgd_etth2.py (SGD như LTSF-Linear, mọi dataset), sgd_lr_tune.py, sgd_lr_extend.py, optim_etth2.py,
│   │                    official_ltsf.py (code gốc LTSF-Linear, có tùy chọn drop_last của test loader)
│   ├── checks/          đo đạc và các phép kiểm (bench, stopping, checks_mae, irls_dtype, compare_cell, huber_delta_h, …)
│   └── report/          report_*.py (bảng của từng nhiệm vụ); make_numbers/make_tables/make_figures (báo cáo LaTeX);
│                        audit_numbers.py, check_latex.py, run_notebooks.py; report_data.py (đọc results/)
├── notebooks/           00_* (baseline, solver dạng đóng), 01_three_predictions (RQ1), 02_loss_comparison (RQ2),
│                        03_outliers_and_sgd (RQ3); archive/ (bản cũ, không chạy lại)
├── results/             mọi kết quả, xem results/README.md (weights/ và .npy lớn không commit)
├── report/              báo cáo LaTeX theo template AIO2026, xem report/README.md
├── docs/                tasks/ (NHIEM_VU_*), reports/ (BAO_CAO_*: báo cáo kỹ thuật), plan/, NHAT_KY_THAY_DOI.md,
│                        DUONG_DAN_CU.md (đường dẫn cũ → mới), outline.docx; refs/ (PDF, không commit)
├── checkpoints/         checkpoint DLinear SGD của notebook 00
├── data/                ETTh1.csv, ETTh2.csv, ETTm1.csv
└── third_party/         LTSF-Linear (git submodule, ghim commit 0c11366)
```

## Cài đặt

Đã kiểm với Python 3.13.5. Cài **torch trước**, đúng bản CUDA của GPU, rồi mới cài phần còn lại (xem chú thích trong `requirements.txt`):

```bash
python -m venv .venv && source .venv/bin/activate        # Windows/Git Bash: source .venv/Scripts/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # RTX 50xx (Blackwell, CUDA ≥ 12.8)
# hoặc, không có GPU NVIDIA:  pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git submodule update --init                               # LTSF-Linear, cho notebook 01 và official_ltsf.py
pip install nbclient nbformat ipykernel                   # tùy chọn: chạy notebook bằng scripts/report/run_notebooks.py
```

Grid (`scripts/grid/runner.py`) cần GPU CUDA. SGD chạy trên CPU. Test, notebook 02, 03 và việc sinh bảng, hình không cần GPU.

## Tái lập

```bash
./run_all.sh quick     # venv, 66 test, chạy lại grid ETTh1 H = 96 và so với results/, sinh lại số/bảng/hình, biên dịch báo cáo
./run_all.sh full      # chạy lại toàn bộ thí nghiệm theo thứ tự (xem bảng thời gian bên dưới)
./run_all.sh full --dry-run   # chỉ in các lệnh
```

`VENV=.venv-gpu ./run_all.sh` dùng một venv có sẵn; `TORCH_INDEX=…/whl/cpu` cài torch bản CPU. Chạy được trên Linux, macOS và Windows qua Git Bash. Trên Windows, bước 3 của `quick` cần GPU NVIDIA; không có GPU thì bước này tự bỏ qua. Bước 5 cần `latexmk` và `pdflatex`; không có thì bỏ qua.

### Môi trường đã dùng để sinh mọi kết quả

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.5 |
| torch | 2.14.0+cu130 (CUDA 13.0, cuDNN 92400): grid, IRLS, code gốc LTSF-Linear trên GPU; SGD chạy trên CPU bằng chính bản này. Bản 2.14.0+cpu (`TORCH_INDEX=…/whl/cpu`) cho máy không có GPU; chưa dùng để sinh kết quả trong `results/` |
| numpy, scipy, pandas, matplotlib, scikit-learn | 2.3.1, 1.18.1, 2.3.1, 3.11.0, 1.9.0 (`requirements.txt`) |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB, driver 616.92; TF32 tắt tường minh |
| CPU | Intel 12 luồng (Family 6 Model 151) |
| Hệ điều hành | Windows 11 (10.0.26100), lệnh chạy qua Git Bash |
| TeX | TinyTeX: pdfLaTeX, latexmk, biblatex 3.22a (backend bibtex) |

Mỗi JSON và mỗi dòng `lambda_path.csv` ghi phần cứng và commit đã sinh ra nó.

### Seed

| Thí nghiệm | Seed |
|---|---|
| SGD, lr gốc | 2021–2040 (NLinear trên ETTh2: 2021–2030) |
| SGD tinh chỉnh lr: chọn lr | 2021–2025 (mỗi mức lr) |
| SGD tinh chỉnh lr: chạy cuối | 2026–2040 (15 seed, không trùng seed chọn lr) |
| Code gốc LTSF-Linear (`official_ltsf.py`) | 2021–2025 |
| Vị trí outlier nhân tạo | 2026; thêm 2027, 2028 ở ETTh1 H = 96 Linear |

Grid (dạng đóng, IRLS) tất định, không có seed.

### Thời gian chạy dự kiến (đo trên máy ở trên)

`quick` (đo trên bản clone mới với venv mới, `docs/reports/BAO_CAO_NV8.md` mục 4; bước 5 đo lại khi có TeX, `docs/reports/BAO_CAO_NV9.md`):

| Bước | Thời gian |
|---|---|
| 1. Môi trường | khoảng 3,5 phút với venv mới, torch từ cache pip (lần đầu tải torch thì lâu hơn); 7 giây với venv có sẵn |
| 2. Test (66) | 30 giây |
| 3. Chạy lại grid ETTh1 H = 96 (488 dòng, GPU) | khoảng 15 phút (MAE khoảng 13 phút); `SKIP_CELL=1` để bỏ |
| 4. Sinh số, bảng, hình; audit; kiểm tĩnh | 10–30 giây |
| 5. Biên dịch báo cáo (latexmk: pdfLaTeX nhiều lượt + bibtex) | 15 giây khi build sạch; khoảng 1 giây nếu không có gì đổi |
| **Tổng** | khoảng 20 phút với venv mới; dưới 5 phút với `SKIP_CELL=1` (47 giây với venv có sẵn) |

`full` (tổng thời gian các lần chạy; các bước SGD chạy song song nên thời gian thực ngắn hơn):

| Bước | Phần cứng | Thời gian |
|---|---|---|
| A. Grid: 3 dataset × 4 H × 3 mô hình × 3 mục tiêu | GPU | 14,6 giờ (MAE chiếm gần hết) |
| B. Huber δ_MAD (grid, độ nhạy) | GPU | 0,6 giờ, cộng khoảng 10 phút CPU |
| C. Outlier nhân tạo | GPU | khoảng 8 giờ (độ lớn 3σ, 5σ chiếm gần 6 giờ) |
| D. SGD lr gốc, 20 seed | CPU | 5,5 giờ CPU |
| E. SGD tinh chỉnh lr (5 tiến trình × 2 luồng) | CPU | 2,3 giờ thời gian thực |
| F. Mở rộng lưới lr ở 15 ô biên; code gốc với `drop_last`; Huber δ theo bước h | CPU; GPU; GPU | 0,8 giờ thực; 2,6 giờ (tổng các lần chạy; khoảng 1,3 giờ thực với 5 tiến trình song song); 3 phút |
| G. Bảng, số, hình | CPU | dưới 5 phút |

### Kết quả mong đợi của ô kiểm (`quick`, bước 3)

`scripts/checks/compare_cell.py` so **mọi dòng** của grid ETTh1 H = 96 chạy lại (488 dòng: 142 λ của MSE, 10–12 λ của MAE và Huber δ = 1, cho 3 mô hình) với `results/lambda_path.csv`, trên các cột `train_obj`, `val_mse`, `val_mae`, `val_huber`, `test_mse`, `test_mae`. Ngưỡng chênh tuyệt đối: **1e-10** với mục tiêu MSE, **1e-8** với MAE và Huber; mọi dòng của ô phải có ở cả hai phía. Trên máy ở trên, chênh đo được là 0 ở mọi dòng. Một vài giá trị để đối chiếu nhanh (làm tròn 4 chữ số; λ\*_MSE = argmin MSE val, λ\*_MAE = argmin MAE val; Huber ở đây là δ = 1 của grid):

| Mô hình | Mục tiêu | λ = 0: MSE / MAE test | λ\*_MSE: MSE test | λ\*_MAE: MAE test |
|---|---|---|---|---|
| Linear | MSE | 0,3702 / 0,3915 | 0,3702 (λ = 0) | 0,3915 (λ = 141,254) |
| Linear | MAE | 0,3653 / 0,3833 | 0,3653 (λ = 0) | 0,3833 (λ = 0) |
| Linear | Huber | 0,3678 / 0,3880 | 0,3678 (λ = 1) | 0,3879 (λ = 30) |
| DLinear | MSE | 0,3702 / 0,3915 | 0,3698 (λ = 446,684) | 0,3914 (λ = 446,684) |
| DLinear | MAE | 0,3653 / 0,3833 | 0,3647 (λ = 300) | 0,3832 (λ = 100) |
| DLinear | Huber | 0,3678 / 0,3880 | 0,3671 (λ = 300) | 0,3879 (λ = 100) |
| NLinear | MSE | 0,3702 / 0,3915 | 0,3695 (λ = 1412,54) | 0,3913 (λ = 794,328) |
| NLinear | MAE | 0,3662 / 0,3841 | 0,3638 (λ = 3000) | 0,3839 (λ = 300) |
| NLinear | Huber | 0,3685 / 0,3887 | 0,3674 (λ = 1000) | 0,3886 (λ = 300) |

- Chỉ mục mọi bảng, hình, thư mục kết quả và script sinh ra chúng: [results/README.md](results/README.md).
- Các báo cáo kỹ thuật cũ dùng đường dẫn trước khi dọn repo; tra [docs/DUONG_DAN_CU.md](docs/DUONG_DAN_CU.md).

## Báo cáo

- Báo cáo cuối (LaTeX): `report/main.tex` → `report/build/main.pdf` (biên dịch theo [report/README.md](report/README.md)).
- Báo cáo kỹ thuật theo từng nhiệm vụ: [docs/reports/](docs/reports/), mới nhất là [BAO_CAO_NV8.md](docs/reports/BAO_CAO_NV8.md) và [BAO_CAO_NV9.md](docs/reports/BAO_CAO_NV9.md).
- Nhật ký thay đổi code: [docs/NHAT_KY_THAY_DOI.md](docs/NHAT_KY_THAY_DOI.md).

## Quy ước

- λ phạt trên **tổng** bình phương lỗi: `‖Y − XWᵀ‖² + λ‖W‖²`, không phạt bias; tương ứng `weight_decay ≈ 2λ/(n·H)` của PyTorch khi loss là MSE trung bình.
- MAE, Huber (`irls`) dùng cùng kiểu tổng: MAE là `Σ|r| + λ‖W‖²`, Huber là `Σ ρ_δ(r)/δ + λ‖W‖²`. DLinear có weight decay dùng `pen = N⁻¹`; NLinear phạt L − 1 hệ số đầu của W_eff.
- Mọi mô hình dùng chung trọng số cho 7 kênh (`individual=False`), chuẩn hóa từng kênh bằng mean/std của train, giống baseline SGD.
- λ, δ, learning rate và epoch chỉ chọn trên validation. MSE test báo ở λ\*_MSE (argmin MSE val), MAE test ở λ\*_MAE.

## Trích dẫn

- Zeng, A., Chen, M., Zhang, L., Xu, Q. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. Code: https://github.com/cure-lab/LTSF-Linear
- Dữ liệu ETT: Zhou, H. et al. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021. https://github.com/zhouhaoyi/ETDataset
