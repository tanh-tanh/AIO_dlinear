# Bảng đường dẫn cũ → mới (Nhiệm vụ 8, Phần A)

Ngày dời: 11/10/2026. Các báo cáo kỹ thuật cũ trong `docs/reports/` giữ nguyên nội dung, nên các lệnh "Tái lập" và liên kết trong đó vẫn dùng đường dẫn cũ. Tra bảng dưới để đổi sang đường dẫn mới. Mọi file dời bằng `git mv` (trừ `docs/NHIEM_VU_8.md` và `TEMPLATE/`, chưa được track trước khi dời), nên `git log --follow <đường dẫn mới>` vẫn thấy lịch sử.

Các script đổi `ROOT = Path(__file__).resolve().parents[1]` thành `parents[2]` và thêm cả bốn thư mục `scripts/{grid,sgd,checks,report}` vào `sys.path`, nên import chéo giữa các script (ví dụ `from bench_5060ti import ...`) không đổi. Lệnh cũ `python scripts/X.py ...` thành `python scripts/<thư mục>/X.py ...` với cùng tham số.

| Đường dẫn cũ | Đường dẫn mới |
|---|---|
| `scripts/runner.py` | `scripts/grid/runner.py` |
| `scripts/select_lambda.py` | `scripts/grid/select_lambda.py` |
| `scripts/huber_deltas.py` | `scripts/grid/huber_deltas.py` |
| `scripts/sgd_etth2.py` | `scripts/sgd/sgd_etth2.py` |
| `scripts/sgd_lr_tune.py` | `scripts/sgd/sgd_lr_tune.py` |
| `scripts/official_ltsf.py` | `scripts/sgd/official_ltsf.py` |
| `scripts/optim_etth2.py` | `scripts/sgd/optim_etth2.py` |
| `scripts/bench_5060ti.py` | `scripts/checks/bench_5060ti.py` |
| `scripts/checks_mae.py` | `scripts/checks/checks_mae.py` |
| `scripts/constant_windows.py` | `scripts/checks/constant_windows.py` |
| `scripts/gd_early_stop.py` | `scripts/checks/gd_early_stop.py` |
| `scripts/irls_dtype_check.py` | `scripts/checks/irls_dtype_check.py` |
| `scripts/mae_filtered.py` | `scripts/checks/mae_filtered.py` |
| `scripts/recheck_paths.py` | `scripts/checks/recheck_paths.py` |
| `scripts/ridge_w0.py` | `scripts/checks/ridge_w0.py` |
| `scripts/stopping_check.py` | `scripts/checks/stopping_check.py` |
| `scripts/report_etth2.py` | `scripts/report/report_etth2.py` |
| `scripts/report_etth2_2.py` | `scripts/report/report_etth2_2.py` |
| `scripts/report_nv7.py` | `scripts/report/report_nv7.py` |
| `scripts/report_outliers.py` | `scripts/report/report_outliers.py` |
| `scripts/report_sgd_b1_b4.py` | `scripts/report/report_sgd_b1_b4.py` |
| `docs/BAO_CAO_5060TI.md` | `docs/reports/BAO_CAO_5060TI.md` |
| `docs/BAO_CAO_ETTH2_2.md` | `docs/reports/BAO_CAO_ETTH2_2.md` |
| `docs/BAO_CAO_ETTH2_SGD.md` | `docs/reports/BAO_CAO_ETTH2_SGD.md` |
| `docs/BAO_CAO_IRLS_DTYPE.md` | `docs/reports/BAO_CAO_IRLS_DTYPE.md` |
| `docs/BAO_CAO_NV7.md` | `docs/reports/BAO_CAO_NV7.md` |
| `docs/BAO_CAO_OUTLIER.md` | `docs/reports/BAO_CAO_OUTLIER.md` |
| `docs/BAO_CAO_SGD_B1_B4.md` | `docs/reports/BAO_CAO_SGD_B1_B4.md` |
| `docs/BAO_CAO_TIEN_DO.md` | `docs/reports/BAO_CAO_TIEN_DO.md` |
| `docs/BAO_CAO_TIEU_CHI_DUNG.md` | `docs/reports/BAO_CAO_TIEU_CHI_DUNG.md` |
| `docs/NHIEM_VU_2.md` | `docs/tasks/NHIEM_VU_2.md` |
| `docs/NHIEM_VU_3.md` | `docs/tasks/NHIEM_VU_3.md` |
| `docs/NHIEM_VU_4.md` | `docs/tasks/NHIEM_VU_4.md` |
| `docs/NHIEM_VU_5.md` | `docs/tasks/NHIEM_VU_5.md` |
| `docs/NHIEM_VU_6.md` | `docs/tasks/NHIEM_VU_6.md` |
| `docs/NHIEM_VU_7.md` | `docs/tasks/NHIEM_VU_7.md` |
| `docs/NHIEM_VU_8.md` | `docs/tasks/NHIEM_VU_8.md` |
| `NHIEM_VU_5060TI.md` | `docs/tasks/NHIEM_VU_5060TI.md` |
| `docs/CHECKLIST_CON_LAI.md` | `docs/plan/CHECKLIST_CON_LAI.md` |
| `TEMPLATE/` (chưa track) | `report/template/` |

Không dời: `src/`, `tests/`, `notebooks/`, `data/`, `checkpoints/`, `results/` (giữ nguyên nội dung), `third_party/LTSF-Linear` (submodule), `docs/outline.docx`, `docs/NHAT_KY_THAY_DOI.md`.
