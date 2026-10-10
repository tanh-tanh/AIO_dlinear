# Chỉ mục kết quả

Mọi file trong `results/` do script sinh ra; **không sửa tay**. Báo cáo kỹ thuật ở `docs/reports/`, nhiệm vụ ở `docs/tasks/`. Đường dẫn script trong các báo cáo cũ là đường dẫn trước khi dời (tra `docs/DUONG_DAN_CU.md`); bảng dưới dùng đường dẫn mới. Mọi số, bảng, hình của báo cáo LaTeX đọc các file này qua `scripts/report/report_data.py`.

## Bảng và hình của báo cáo

| Mục | Nội dung | File dữ liệu | Script sinh dữ liệu | Script sinh bảng/hình | Báo cáo kỹ thuật |
|---|---|---|---|---|---|
| B1 | SGD (lr gốc, 20 seed) so với số công bố | `sgd_b1_b4/summary.json` (`B1`); JSON từng seed ở `sgd_b1_b4/ETTh1/`, `sgd_b1_b4/ETTm1/`, `sgd_etth2_20seed/` | `scripts/sgd/sgd_etth2.py` | `scripts/report/report_sgd_b1_b4.py`, `make_tables.py` (`tab_b1`) | `BAO_CAO_SGD_B1_B4.md` mục 4 |
| B2 | Ba dự đoán của RQ1 (ETTh1, H = 96) | `B2_three_predictions.md`, `three_predictions_sweep.csv` | `notebooks/01_three_predictions.ipynb` | `make_tables.py` (`tab_b2`) | `BAO_CAO_TIEN_DO.md`; notebook 01 |
| B3 | Nghiệm tối ưu, 3 mô hình × 3 mục tiêu × {λ = 0, λ\*} × 3 dataset × 4 H | `huber_mad/results_v2.csv` (MSE, MAE từ grid; Huber δ_MAD); `results_v2.csv` (Huber δ = 1); đường λ: `lambda_path.csv`, `huber_mad/lambda_path.csv` | `scripts/grid/runner.py`, `scripts/grid/select_lambda.py`, `scripts/grid/huber_deltas.py` | `make_tables.py` (`tab_b3_main`, `tab_b3_full`) | `BAO_CAO_NV7.md` mục 3–4 |
| B4 | SGD − dạng đóng λ\*, quy tắc ba trường hợp; lr gốc và tinh chỉnh | `sgd_b1_b4/summary.json` (`B4`, `C3`); `nv7/summary.json` (`D_B4`, `D_C3`); `sgd_lr/selection.json` | `scripts/sgd/sgd_etth2.py`, `scripts/sgd/sgd_lr_tune.py` | `report_sgd_b1_b4.py`, `report_nv7.py --part D`, `make_tables.py` (`tab_b4_*`, `tab_c3`, `tab_lr`) | `BAO_CAO_SGD_B1_B4.md` mục 5–6, `BAO_CAO_NV7.md` mục 6 |
| H1 | RMS chênh dự báo DLinear/NLinear với Linear theo λ | `three_predictions_sweep.csv`; PNG cũ `H1_three_predictions.png` | notebook 01 | `make_figures.py` → `report/figures/H1_three_predictions.pdf` | notebook 01 |
| H2 | Sai số test theo tỷ lệ outlier | `outliers/results_v2.csv` (MSE, MAE), `huber_mad/outliers/results_v2.csv` (Huber δ_MAD); PNG `outliers/H2_v2_outlier_robustness{,_mae}.png` | `runner.py --outlier-mode …` | `report_nv7.py` (`figure_h2_v2`), `make_figures.py` → `H2_outlier_{mse,mae}.pdf` | `BAO_CAO_OUTLIER.md`, `BAO_CAO_NV7.md` mục 4.5 |
| H3 | SGD theo seed, dạng đóng λ\*, số công bố | như B1, B4; PNG `sgd_b1_b4/H3_sgd_vs_optimum.png` | `sgd_etth2.py` | `report_sgd_b1_b4.py` (`figure`), `make_figures.py` → `H3_sgd_vs_optimum.pdf` | `BAO_CAO_SGD_B1_B4.md` mục 7 |
| H4 | Độ nhạy của Huber theo δ | `huber_mad/sens_x{0.5,2,4}/results_v2.csv`, `nv7/summary.json` (`A_sensitivity`); PNG `huber_mad/A3_delta_sensitivity.png` | `runner.py --huber-delta mad --delta-scale …` | `report_nv7.py` (`figure_sensitivity`), `make_figures.py` → `H4_delta_sensitivity.pdf` | `BAO_CAO_NV7.md` mục 4.6 |
| H5 | Sai số test theo độ lớn outlier (3σ, 5σ, 10σ) | `outliers_mag/results_v2.csv`, `nv7/summary.json` (`C`); PNG `outliers_mag/C_outlier_magnitude.png` | `runner.py --outlier-magnitude …` | `report_nv7.py` (`figure_c`), `make_figures.py` → `H5_outlier_magnitude.pdf` | `BAO_CAO_NV7.md` mục 5 |

## Thư mục con

"Git": file nhỏ (CSV, JSON, MD, PNG) được commit; trọng số `.npy` lớn bị `.gitignore`. Cột "Sinh lại" cho biết cách tạo lại phần không commit.

| Thư mục | Nội dung | Nhiệm vụ / báo cáo | Git | Sinh lại |
|---|---|---|---|---|
| `lambda_path.csv`, `results.csv`, `results_v2.csv` (gốc) | grid đầy đủ: mọi λ của 3 dataset × 4 H × 3 mô hình × 3 mục tiêu (Huber δ = 1); λ\* theo `val_objective`/`val_mse` (`results.csv`) và thêm `val_mae` (`results_v2.csv`) | NV2–3, NV7 B | có | `runner.py`, `select_lambda.py` |
| `weights/` | W của mọi dòng `lambda_path.csv` (khoảng 5 GB) | grid | **không** | `python scripts/grid/runner.py` (khoảng 15 giờ GPU), cần cho warm start và `tests/test_val_metrics.py` (test bỏ qua khi thiếu) |
| `bench_5060ti/` | đo tốc độ IRLS, chunk, warm start, ước lượng thời gian grid | `BAO_CAO_5060TI.md` | có | `scripts/checks/bench_5060ti.py` |
| `irls_dtype/` | float32 / float64 / pha độ chính xác cho IRLS | `BAO_CAO_IRLS_DTYPE.md` | có | `scripts/checks/irls_dtype_check.py` |
| `stopping/`, `checks_mae/` | tiêu chí dừng của IRLS; tính duy nhất của MAE, DLinear ≡ phạt N⁻¹ | `BAO_CAO_TIEU_CHI_DUNG.md` | có | `scripts/checks/stopping_check.py`, `checks_mae.py` |
| `sgd_etth2/` | SGD 3 seed trên ETTh2 (nhiệm vụ 3) | `BAO_CAO_ETTH2_SGD.md` | JSON có, `.npy` không | `scripts/sgd/sgd_etth2.py` |
| `ridge_w0/`, `recheck/`, `constant_windows/`, `etth2_report/` | ridge co về W₀ ≠ 0; chạy lại đường λ; cửa sổ hằng của ETTh2; bảng nhiệm vụ 3 | `BAO_CAO_ETTH2_SGD.md` | có (`recheck/weights/` không) | `scripts/checks/{ridge_w0,recheck_paths,constant_windows}.py`, `scripts/report/report_etth2.py` |
| `sgd_etth2_20seed/`, `sgd_etth2_filtered/`, `mae_filtered/`, `official_ltsf/`, `etth2_report_2/` | ETTh2: 20 seed; SGD và MAE trên train đã lọc cửa sổ hằng; code gốc LTSF-Linear | `BAO_CAO_ETTH2_2.md` | JSON có, `.npy` không | `sgd_etth2.py`, `scripts/checks/mae_filtered.py`, `scripts/sgd/official_ltsf.py`, `scripts/report/report_etth2_2.py` |
| `optim_etth2/` | GD/Adam toàn batch trên Linear ETTh2 H = 720 | `BAO_CAO_SGD_B1_B4.md` mục 3 | JSON có, `.npy` không | `scripts/sgd/optim_etth2.py` |
| `sgd_b1_b4/` | SGD 20 seed trên ETTh1, ETTm1; bảng B1, B4, C.3; hình H3 | `BAO_CAO_SGD_B1_B4.md` | JSON có, `.npy` không | `sgd_etth2.py`, `report_sgd_b1_b4.py` |
| `outliers/` | grid với outlier nhân tạo (MSE, MAE, Huber δ = 1); D.1, D.2 (SGD trên ETTm1) | `BAO_CAO_OUTLIER.md` | CSV/JSON có, `weights/` và `.npy` của `d2_lr/` không | `runner.py --outlier-mode …`, `report_outliers.py` |
| `huber_mad/` | Huber δ_MAD: δ của từng ô (`deltas.csv`), grid, outlier, độ nhạy `sens_x*`; B3 hoàn chỉnh `results_v2.csv` | `BAO_CAO_NV7.md` Phần A | CSV có, `**/weights/` không | `huber_deltas.py`, `runner.py --huber-delta mad …` |
| `outliers_mag/` | outlier 3σ, 5σ | `BAO_CAO_NV7.md` Phần C | CSV có, `weights/` không | `runner.py --outlier-magnitude …` |
| `sgd_lr/` | SGD tinh chỉnh lr (990 lần chạy), `selection.json` | `BAO_CAO_NV7.md` Phần D | JSON có, `.npy` không | `scripts/sgd/sgd_lr_tune.py` |
| `nv7/` | bảng và tóm tắt nhiệm vụ 7 | `BAO_CAO_NV7.md` | có | `scripts/report/report_nv7.py --part all` |

Lệnh đầy đủ cho từng thư mục nằm ở mục "Tái lập" của báo cáo tương ứng; `./run_all.sh full` chạy lại theo thứ tự.
