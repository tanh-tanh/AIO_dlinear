#!/usr/bin/env bash
# Tái lập dlinear-objective-lab bằng một lệnh (NHIEM_VU_8, Phần D).
#
#   ./run_all.sh            # = quick
#   ./run_all.sh quick      # môi trường, test, chạy lại một ô grid để kiểm, sinh bảng/hình/số, biên dịch báo cáo
#   ./run_all.sh full       # chạy lại toàn bộ thí nghiệm theo thứ tự các nhiệm vụ (nhiều giờ GPU + CPU)
#   ./run_all.sh full --dry-run   # chỉ in các lệnh của full (và chạy --dry-run của runner, sgd_lr_tune)
#
# Biến môi trường:
#   VENV=<thư mục>        venv dùng/tạo (mặc định .venv-run). Dùng venv có sẵn: VENV=.venv-gpu ./run_all.sh
#   TORCH_INDEX=<url>     nơi cài torch nếu venv chưa có. Mặc định bản CUDA 13.0 (RTX 50xx, Blackwell):
#                         https://download.pytorch.org/whl/cu130 ; máy không có GPU NVIDIA:
#                         TORCH_INDEX=https://download.pytorch.org/whl/cpu
#   PYTHON=<lệnh>         Python dùng để tạo venv (mặc định python3, rồi python). Đã kiểm với 3.13.5.
#   SKIP_CELL=1           bỏ bước 3 (chạy lại ô ETTh1 H = 96, khoảng 15 phút trên RTX 5060 Ti).
#
# Chạy được trên Linux, macOS (bash) và Windows qua Git Bash. Grid (bước 3 và phần lớn chế độ full) cần GPU CUDA;
# không có GPU thì bước 3 tự bỏ qua (README). Mọi script chạy từ gốc repo.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"
DRY=0
[[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]] && DRY=1
[[ "$MODE" == "--dry-run" ]] && MODE=quick
export PYTHONIOENCODING=utf-8

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
STEP_T0=0
step() { STEP_T0=$SECONDS; say "$*"; }
done_step() { printf '   (%d s)\n' $((SECONDS - STEP_T0)); }

# ---------------------------------------------------------------------------
# 1. Môi trường
# ---------------------------------------------------------------------------
setup_env() {
    step "1. Môi trường"
    VENV="${VENV:-.venv-run}"
    if [[ -x "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"
    elif [[ -x "$VENV/bin/python" ]]; then PY="$VENV/bin/python"
    else
        local base="${PYTHON:-}"
        if [[ -z "$base" ]]; then
            # python3 trên Windows có thể là lối tắt của Microsoft Store: kiểm bằng cách chạy thật
            for c in python3 python; do
                if "$c" -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>/dev/null; then base="$c"; break; fi
            done
        fi
        [[ -n "$base" ]] || { echo "Không tìm thấy Python >= 3.10"; exit 1; }
        echo "tạo venv $VENV bằng $base ($("$base" --version))"
        "$base" -m venv "$VENV"
        if [[ -x "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi
        "$PY" -m pip install -q --upgrade pip
    fi
    if ! "$PY" -c "import torch" 2>/dev/null; then
        # torch phải cài TRƯỚC requirements.txt, đúng bản CUDA (xem chú thích trong requirements.txt)
        local idx="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
        echo "cài torch==2.14.0 từ $idx"
        "$PY" -m pip install -q torch==2.14.0 --index-url "$idx"
    fi
    "$PY" -m pip install -q -r requirements.txt
    "$PY" -c "import torch, numpy; print('python', __import__('sys').version.split()[0], '| torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| numpy', numpy.__version__)"
    if [[ -d .git ]] && [[ ! -e third_party/LTSF-Linear/run_longExp.py ]]; then
        git submodule update --init third_party/LTSF-Linear || echo "(không lấy được submodule LTSF-Linear; chỉ notebook 01 và official_ltsf.py cần)"
    fi
    done_step
}

# ---------------------------------------------------------------------------
# quick
# ---------------------------------------------------------------------------
quick() {
    setup_env

    step "2. Test (kỳ vọng 66 test, tất cả đạt; test cần results/weights/ tự bỏ qua nếu thiếu)"
    "$PY" -m unittest discover -s tests -t .
    done_step

    step "3. Chạy lại grid ETTh1 H = 96 (3 mô hình × 3 mục tiêu) vào thư mục tạm, so với results/lambda_path.csv"
    if [[ "${SKIP_CELL:-0}" == "1" ]]; then
        echo "bỏ qua (SKIP_CELL=1)"
    elif ! "$PY" -c "import sys, torch; sys.exit(not torch.cuda.is_available())"; then
        echo "bỏ qua: không có GPU CUDA (runner.py chạy IRLS trên GPU)"
    else
        local tmp; tmp="$(mktemp -d)"
        local out="$tmp"
        command -v cygpath >/dev/null && out="$(cygpath -w "$tmp")"
        "$PY" scripts/grid/runner.py --data ETTh1 --H 96 --out "$out" > "$tmp/run.log" 2>&1 \
            || { tail -20 "$tmp/run.log"; exit 1; }
        "$PY" scripts/checks/compare_cell.py --new "$out/lambda_path.csv" --data ETTh1 --H 96
        rm -rf "$tmp"
    fi
    done_step

    step "4. Sinh lại số, bảng, hình của báo cáo từ results/"
    "$PY" scripts/report/make_numbers.py
    "$PY" scripts/report/make_tables.py
    "$PY" scripts/report/make_figures.py
    "$PY" scripts/report/audit_numbers.py | head -1
    "$PY" scripts/report/check_latex.py
    done_step

    step "5. Biên dịch báo cáo"
    if command -v latexmk >/dev/null && command -v pdflatex >/dev/null; then
        (cd report && latexmk main.tex) && echo "→ report/build/main.pdf"
    else
        echo "bỏ qua: không có latexmk/pdflatex (xem report/README.md)"
    fi
    done_step
    say "quick xong trong $SECONDS s"
}

# ---------------------------------------------------------------------------
# full: theo thứ tự các nhiệm vụ. Thời gian đo trên RTX 5060 Ti 16 GB + CPU Intel 12 luồng (Phụ lục D của báo cáo).
# ---------------------------------------------------------------------------
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

full() {
    if [[ $DRY == 1 ]]; then
        PY="${VENV:+$VENV/Scripts/python.exe}"; [[ -x "${PY:-}" ]] || PY="$(command -v python3 || command -v python)"
        say "full --dry-run: chỉ in lệnh (python: $PY)"
    else
        setup_env
    fi
    local R="scripts/grid/runner.py" S="scripts/grid/select_lambda.py" G="scripts/sgd/sgd_etth2.py"
    local sel=(--selections val_mse val_mae val_objective)

    say "A. Grid: 3 dataset × 4 H × 3 mô hình × 3 mục tiêu (Huber δ = 1). GPU, khoảng 15 giờ (MAE chiếm gần hết)"
    [[ $DRY == 1 ]] && "$PY" "$R" --dry-run || true
    run "$PY" "$R"
    run "$PY" "$S"
    run "$PY" "$S" "${sel[@]}" --out results/results_v2.csv

    say "B. Huber δ_MAD (nhiệm vụ 7 A). CPU 10 phút + GPU khoảng 50 phút"
    run "$PY" scripts/grid/huber_deltas.py
    run "$PY" "$R" --objective Huber --skip-mse --huber-delta mad --out huber_mad
    run "$PY" "$S" --in results/lambda_path.csv results/huber_mad/lambda_path.csv "${sel[@]}" --out results/huber_mad/results_v2.csv
    for s in 0.5 2 4; do
        run "$PY" "$R" --data ETTh1 --H 96 720 --model Linear NLinear --objective Huber --skip-mse --huber-delta mad --delta-scale $s --out huber_mad/sens_x$s
        run "$PY" "$R" --data ETTh2 --H 720 --model Linear NLinear --objective Huber --skip-mse --huber-delta mad --delta-scale $s --out huber_mad/sens_x$s
        run "$PY" "$S" --in results/huber_mad/sens_x$s/lambda_path.csv "${sel[@]}" --out results/huber_mad/sens_x$s/results_v2.csv
    done

    say "C. Outlier nhân tạo, ETTh1 (nhiệm vụ 6, 7). GPU, khoảng 8 giờ (độ lớn 3σ, 5σ chiếm gần 6 giờ)"
    for rate in 0.1 0.01 0.05; do
        run "$PY" "$R" --data ETTh1 --model Linear NLinear --out outliers --outlier-mode series --outlier-rate $rate --outlier-seed 2026
        run "$PY" "$R" --data ETTh1 --model Linear NLinear --objective Huber --skip-mse --huber-delta mad --out huber_mad/outliers --outlier-mode series --outlier-rate $rate --outlier-seed 2026
    done
    for rate in 0.1 0.01; do
        run "$PY" "$R" --data ETTh1 --H 96 720 --model Linear NLinear --out outliers --outlier-mode target --outlier-rate $rate --outlier-seed 2026
        run "$PY" "$R" --data ETTh1 --H 96 720 --model Linear NLinear --objective Huber --skip-mse --huber-delta mad --out huber_mad/outliers --outlier-mode target --outlier-rate $rate --outlier-seed 2026
    done
    for seed in 2027 2028; do for rate in 0.1 0.01; do
        run "$PY" "$R" --data ETTh1 --H 96 --model Linear --out outliers --outlier-mode series --outlier-rate $rate --outlier-seed $seed
        run "$PY" "$R" --data ETTh1 --H 96 --model Linear --objective Huber --skip-mse --huber-delta mad --out huber_mad/outliers --outlier-mode series --outlier-rate $rate --outlier-seed $seed
    done; done
    for mag in 3 5; do for mode in series target; do for rate in 0.1 0.01; do
        run "$PY" "$R" --data ETTh1 --H 96 720 --model Linear --objective MSE MAE Huber --huber-delta mad --out outliers_mag --outlier-mode $mode --outlier-rate $rate --outlier-seed 2026 --outlier-magnitude $mag
    done; done; done
    run "$PY" "$S" --in results/outliers/lambda_path.csv "${sel[@]}" --out results/outliers/results_v2.csv
    run "$PY" "$S" --in results/outliers/lambda_path.csv results/huber_mad/outliers/lambda_path.csv "${sel[@]}" --out results/huber_mad/outliers/results_v2.csv
    run "$PY" "$S" --in results/outliers_mag/lambda_path.csv "${sel[@]}" --out results/outliers_mag/results_v2.csv

    say "D. SGD với lr gốc, 20 seed (nhiệm vụ 4, 5). CPU, khoảng 6 giờ CPU (ETTm1 chiếm phần lớn)"
    for ds in ETTh1 ETTm1; do for m in DLinear Linear NLinear; do
        run "$PY" "$G" --step run --dataset $ds --out sgd_b1_b4/$ds --model $m --seed $(seq 2021 2040)
    done; done
    run "$PY" "$G" --step run --out sgd_etth2_20seed --model Linear DLinear --seed $(seq 2021 2040)
    run "$PY" "$G" --step run --out sgd_etth2_20seed --model NLinear --seed $(seq 2021 2030)
    run "$PY" scripts/sgd/optim_etth2.py

    say "E. SGD tinh chỉnh lr (nhiệm vụ 7 D). CPU, khoảng 2,5 giờ với 5 tiến trình"
    [[ $DRY == 1 ]] && "$PY" scripts/sgd/sgd_lr_tune.py --dry-run --workers 5 --threads 2 | tail -5 || true
    run "$PY" scripts/sgd/sgd_lr_tune.py --workers 5 --threads 2

    say "F. Nhiệm vụ 9: mở rộng lưới lr ở 15 ô biên (CPU, khoảng 0,5 giờ với 5 tiến trình); code gốc với drop_last của
   test loader ở 9 ô (GPU, khoảng 1,5 giờ: ETTm1 chiếm phần lớn); Huber với δ theo bước h ở 2 ô (GPU, khoảng 5 phút)"
    run "$PY" scripts/sgd/sgd_lr_extend.py --workers 5 --threads 2
    [[ $DRY == 1 ]] || [[ -e scratch/ltsf_official/run_longExp.py ]] || "$PY" scripts/sgd/official_ltsf.py --prepare
    for c in ETTh1:DLinear:192 ETTh1:DLinear:720 ETTm1:DLinear:96 ETTm1:DLinear:192 ETTm1:DLinear:336 \
             ETTh2:NLinear:336 ETTh2:NLinear:720 ETTm1:NLinear:336 ETTh2:DLinear:720; do
        IFS=: read -r ds m H <<< "$c"
        for dl in 1 0; do
            run "$PY" scripts/sgd/official_ltsf.py --dataset "$ds" --model "$m" --H "$H" --seed 2021 2022 2023 2024 2025 \
                --test-drop-last $dl --out "official_ltsf_droplast/$ds"
        done
    done
    run "$PY" scripts/checks/huber_delta_h.py

    say "G. Bảng tổng hợp, báo cáo"
    run "$PY" scripts/report/report_sgd_b1_b4.py
    run "$PY" scripts/report/report_nv7.py --part all
    run "$PY" scripts/report/report_nv9.py
    run "$PY" scripts/report/make_numbers.py
    run "$PY" scripts/report/make_tables.py
    run "$PY" scripts/report/make_figures.py
    echo
    echo "Các phân tích phụ (ETTh2: cửa sổ hằng, train đã lọc, code gốc LTSF-Linear; kiểm IRLS) có lệnh riêng ở mục"
    echo "\"Tái lập\" của docs/reports/BAO_CAO_ETTH2_SGD.md, BAO_CAO_ETTH2_2.md, BAO_CAO_TIEU_CHI_DUNG.md (đường dẫn mới: docs/DUONG_DAN_CU.md)."
}

case "$MODE" in
    quick) quick ;;
    full) full ;;
    *) echo "dùng: $0 [quick|full] [--dry-run]"; exit 2 ;;
esac
