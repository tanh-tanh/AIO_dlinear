# Phần B: đổi tiêu chí chọn λ* (Huber δ = 1, chờ Phần A)

Kiểm: 432/432 dòng (val_objective, val_mse) của results.csv có trong results_v2.csv với cùng λ, MSE test, MAE test.

Kiểm B4: λ* của mục tiêu MSE theo val_mse trùng λ* mà report_sgd_b1_b4.py đọc (val_objective của MSE) ở 36/36 ô, nên B4 không đổi.

Theo định nghĩa: với mục tiêu MSE, MSE test giữ nguyên (val_objective = val_mse), MAE test đổi sang λ*_MAE; với MAE, MAE test giữ nguyên, MSE test đổi sang λ*_MSE; với Huber, cả hai đổi (val_huber không còn dùng).

Số ô (trên 36) có λ* mới khác λ* cũ: MSE (MSE test) 0, MSE (MAE test) 30, MAE (MSE test) 24, MAE (MAE test) 0, Huber (MSE test) 13, Huber (MAE test) 16

## B3 theo tiêu chí mới (Huber δ = 1)

MSE test ở λ*_MSE (argmin val MSE), MAE test ở λ*_MAE (argmin val MAE). λ = 0: MSE / MAE test. **Đậm**: thấp nhất trong ô (dataset, H, mô hình) giữa ba mục tiêu.

| Dataset | H | Mô hình | Mục tiêu | λ = 0: MSE / MAE test | λ*_MSE | MSE test | λ*_MAE | MAE test |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | Linear | MSE | 0,3702 / 0,3915 | 0 | 0,3702 | 141,3 | 0,3915 |
| ETTh1 | 96 | Linear | MAE | 0,3653 / 0,3833 | 0 | **0,3653** | 0 | **0,3833** |
| ETTh1 | 96 | Linear | Huber | 0,3678 / 0,3880 | 1 | 0,3678 | 30 | 0,3879 |
| ETTh1 | 96 | DLinear | MSE | 0,3702 / 0,3915 | 446,7 | 0,3698 | 446,7 | 0,3914 |
| ETTh1 | 96 | DLinear | MAE | 0,3653 / 0,3833 | 300 | **0,3647** | 100 | **0,3832** |
| ETTh1 | 96 | DLinear | Huber | 0,3678 / 0,3880 | 300 | 0,3671 | 100 | 0,3879 |
| ETTh1 | 96 | NLinear | MSE | 0,3702 / 0,3915 | 1413 | 0,3695 | 794,3 | 0,3913 |
| ETTh1 | 96 | NLinear | MAE | 0,3662 / 0,3841 | 3000 | **0,3638** | 300 | **0,3839** |
| ETTh1 | 96 | NLinear | Huber | 0,3685 / 0,3887 | 1000 | 0,3674 | 300 | 0,3886 |
| ETTh1 | 192 | Linear | MSE | 0,4042 / 0,4127 | 223,9 | 0,4040 | 199,5 | 0,4126 |
| ETTh1 | 192 | Linear | MAE | 0,4047 / 0,4077 | 1000 | 0,4031 | 300 | **0,4074** |
| ETTh1 | 192 | Linear | Huber | 0,4028 / 0,4095 | 100 | **0,4026** | 100 | 0,4094 |
| ETTh1 | 192 | DLinear | MSE | 0,4042 / 0,4127 | 1000 | 0,4036 | 562,3 | 0,4125 |
| ETTh1 | 192 | DLinear | MAE | 0,4047 / 0,4077 | 3000 | **0,4014** | 300 | **0,4074** |
| ETTh1 | 192 | DLinear | Huber | 0,4028 / 0,4095 | 300 | 0,4022 | 300 | 0,4093 |
| ETTh1 | 192 | NLinear | MSE | 0,4034 / 0,4112 | 2512 | 0,4027 | 891,3 | 0,4110 |
| ETTh1 | 192 | NLinear | MAE | 0,4064 / 0,4080 | 1e+04 | 0,4026 | 1000 | **0,4075** |
| ETTh1 | 192 | NLinear | Huber | 0,4036 / 0,4096 | 1000 | **0,4025** | 300 | 0,4094 |
| ETTh1 | 336 | Linear | MSE | 0,4334 / 0,4342 | 1995 | 0,4324 | 891,3 | 0,4340 |
| ETTh1 | 336 | Linear | MAE | 0,4372 / 0,4283 | 3e+04 | **0,4302** | 1000 | **0,4275** |
| ETTh1 | 336 | Linear | Huber | 0,4332 / 0,4302 | 3000 | 0,4304 | 300 | 0,4300 |
| ETTh1 | 336 | DLinear | MSE | 0,4334 / 0,4342 | 4467 | 0,4317 | 1585 | 0,4340 |
| ETTh1 | 336 | DLinear | MAE | 0,4372 / 0,4283 | 3e+04 | **0,4296** | 3000 | **0,4272** |
| ETTh1 | 336 | DLinear | Huber | 0,4332 / 0,4302 | 3000 | 0,4300 | 1000 | 0,4299 |
| ETTh1 | 336 | NLinear | MSE | 0,4277 / 0,4262 | 8913 | **0,4261** | 1995 | 0,4260 |
| ETTh1 | 336 | NLinear | MAE | 0,4361 / 0,4247 | 3e+04 | 0,4290 | 3000 | **0,4237** |
| ETTh1 | 336 | NLinear | Huber | 0,4308 / 0,4255 | 1e+04 | 0,4280 | 1000 | 0,4253 |
| ETTh1 | 720 | Linear | MSE | 0,4714 / 0,4878 | 1,778e+04 | 0,4701 | 4467 | 0,4878 |
| ETTh1 | 720 | Linear | MAE | 0,4568 / 0,4737 | 3e+04 | **0,4553** | 1e+04 | **0,4747** |
| ETTh1 | 720 | Linear | Huber | 0,4616 / 0,4792 | 1e+04 | 0,4603 | 3000 | 0,4798 |
| ETTh1 | 720 | DLinear | MSE | 0,4714 / 0,4878 | 1,778e+04 | 0,4697 | 5623 | 0,4879 |
| ETTh1 | 720 | DLinear | MAE | 0,4568 / 0,4737 | 3e+04 | **0,4549** | 1e+04 | **0,4746** |
| ETTh1 | 720 | DLinear | Huber | 0,4616 / 0,4792 | 1e+04 | 0,4597 | 3000 | 0,4797 |
| ETTh1 | 720 | NLinear | MSE | 0,4344 / 0,4505 | 1,995e+04 | 0,4329 | 3981 | 0,4505 |
| ETTh1 | 720 | NLinear | MAE | 0,4318 / 0,4445 | 3e+04 | **0,4282** | 1e+04 | **0,4454** |
| ETTh1 | 720 | NLinear | Huber | 0,4318 / 0,4474 | 1e+04 | 0,4295 | 3000 | 0,4479 |
| ETTh2 | 96 | Linear | MSE | 0,3019 / 0,3655 | 1585 | 0,3009 | 1413 | 0,3649 |
| ETTh2 | 96 | Linear | MAE | 0,2789 / 0,3342 | 1000 | **0,2779** | 1000 | **0,3334** |
| ETTh2 | 96 | Linear | Huber | 0,2873 / 0,3491 | 1000 | 0,2864 | 300 | 0,3487 |
| ETTh2 | 96 | DLinear | MSE | 0,3019 / 0,3655 | 1995 | 0,3008 | 1585 | 0,3649 |
| ETTh2 | 96 | DLinear | MAE | 0,2789 / 0,3342 | 1000 | **0,2778** | 1000 | **0,3333** |
| ETTh2 | 96 | DLinear | Huber | 0,2873 / 0,3491 | 1000 | 0,2863 | 300 | 0,3487 |
| ETTh2 | 96 | NLinear | MSE | 0,2726 / 0,3352 | 1778 | **0,2717** | 1585 | 0,3342 |
| ETTh2 | 96 | NLinear | MAE | 0,2781 / 0,3317 | 3000 | 0,2764 | 1000 | **0,3307** |
| ETTh2 | 96 | NLinear | Huber | 0,2746 / 0,3325 | 1000 | 0,2736 | 300 | 0,3319 |
| ETTh2 | 192 | Linear | MSE | 0,3972 / 0,4276 | 1122 | 0,3966 | 1122 | 0,4273 |
| ETTh2 | 192 | Linear | MAE | 0,3541 / 0,3861 | 1000 | **0,3531** | 1000 | **0,3855** |
| ETTh2 | 192 | Linear | Huber | 0,3717 / 0,4055 | 300 | 0,3712 | 300 | 0,4052 |
| ETTh2 | 192 | DLinear | MSE | 0,3972 / 0,4276 | 1259 | 0,3966 | 1259 | 0,4273 |
| ETTh2 | 192 | DLinear | MAE | 0,3541 / 0,3861 | 1000 | **0,3531** | 1000 | **0,3855** |
| ETTh2 | 192 | DLinear | Huber | 0,3717 / 0,4055 | 300 | 0,3713 | 300 | 0,4052 |
| ETTh2 | 192 | NLinear | MSE | 0,3343 / 0,3762 | 1585 | **0,3336** | 1585 | 0,3754 |
| ETTh2 | 192 | NLinear | MAE | 0,3464 / 0,3765 | 3000 | 0,3447 | 1000 | 0,3756 |
| ETTh2 | 192 | NLinear | Huber | 0,3405 / 0,3752 | 1000 | 0,3395 | 1000 | **0,3742** |
| ETTh2 | 336 | Linear | MSE | 0,4858 / 0,4837 | 1000 | 0,4853 | 1000 | 0,4835 |
| ETTh2 | 336 | Linear | MAE | 0,4116 / 0,4301 | 1000 | **0,4106** | 1000 | **0,4296** |
| ETTh2 | 336 | Linear | Huber | 0,4407 / 0,4531 | 300 | 0,4401 | 300 | 0,4529 |
| ETTh2 | 336 | DLinear | MSE | 0,4858 / 0,4837 | 1122 | 0,4853 | 1122 | 0,4835 |
| ETTh2 | 336 | DLinear | MAE | 0,4116 / 0,4301 | 1000 | **0,4106** | 1000 | **0,4296** |
| ETTh2 | 336 | DLinear | Huber | 0,4407 / 0,4531 | 300 | 0,4401 | 300 | 0,4529 |
| ETTh2 | 336 | NLinear | MSE | 0,3588 / 0,4004 | 1585 | **0,3581** | 1413 | 0,3997 |
| ETTh2 | 336 | NLinear | MAE | 0,3779 / 0,4039 | 3000 | 0,3760 | 1000 | 0,4028 |
| ETTh2 | 336 | NLinear | Huber | 0,3687 / 0,4006 | 1000 | 0,3677 | 1000 | **0,3996** |
| ETTh2 | 720 | Linear | MSE | 0,7407 / 0,6116 | 1000 | 0,7404 | 1122 | 0,6115 |
| ETTh2 | 720 | Linear | MAE | 0,5898 / 0,5337 | 1000 | **0,5893** | 1000 | **0,5335** |
| ETTh2 | 720 | Linear | Huber | 0,6665 / 0,5725 | 300 | 0,6662 | 300 | 0,5723 |
| ETTh2 | 720 | DLinear | MSE | 0,7407 / 0,6116 | 1122 | 0,7403 | 1259 | 0,6115 |
| ETTh2 | 720 | DLinear | MAE | 0,5898 / 0,5337 | 1000 | **0,5892** | 1000 | **0,5335** |
| ETTh2 | 720 | DLinear | Huber | 0,6665 / 0,5725 | 300 | 0,6662 | 300 | 0,5723 |
| ETTh2 | 720 | NLinear | MSE | 0,3930 / 0,4346 | 1995 | 0,3924 | 1585 | 0,4341 |
| ETTh2 | 720 | NLinear | MAE | 0,3945 / 0,4255 | 3000 | 0,3926 | 1000 | **0,4242** |
| ETTh2 | 720 | NLinear | Huber | 0,3920 / 0,4280 | 1000 | **0,3912** | 1000 | 0,4273 |
| ETTm1 | 96 | Linear | MSE | 0,2993 / 0,3428 | 631 | 0,2992 | 891,3 | 0,3427 |
| ETTm1 | 96 | Linear | MAE | 0,2931 / 0,3332 | 1000 | **0,2929** | 1000 | **0,3331** |
| ETTm1 | 96 | Linear | Huber | 0,2991 / 0,3404 | 300 | 0,2991 | 300 | 0,3403 |
| ETTm1 | 96 | DLinear | MSE | 0,2993 / 0,3428 | 1413 | 0,2992 | 1259 | 0,3427 |
| ETTm1 | 96 | DLinear | MAE | 0,2931 / 0,3332 | 1000 | **0,2929** | 1000 | **0,3331** |
| ETTm1 | 96 | DLinear | Huber | 0,2991 / 0,3404 | 1000 | 0,2990 | 1000 | 0,3403 |
| ETTm1 | 96 | NLinear | MSE | 0,3005 / 0,3426 | 5012 | 0,3004 | 2512 | 0,3426 |
| ETTm1 | 96 | NLinear | MAE | 0,2941 / 0,3337 | 1e+04 | **0,2937** | 3000 | **0,3337** |
| ETTm1 | 96 | NLinear | Huber | 0,3004 / 0,3407 | 3000 | 0,3002 | 1000 | 0,3407 |
| ETTm1 | 192 | Linear | MSE | 0,3340 / 0,3637 | 354,8 | 0,3340 | 501,2 | 0,3637 |
| ETTm1 | 192 | Linear | MAE | 0,3324 / 0,3554 | 1000 | **0,3322** | 300 | **0,3553** |
| ETTm1 | 192 | Linear | Huber | 0,3341 / 0,3608 | 100 | 0,3341 | 100 | 0,3608 |
| ETTm1 | 192 | DLinear | MSE | 0,3340 / 0,3637 | 1122 | 0,3339 | 631 | 0,3637 |
| ETTm1 | 192 | DLinear | MAE | 0,3324 / 0,3554 | 3000 | **0,3319** | 300 | **0,3553** |
| ETTm1 | 192 | DLinear | Huber | 0,3341 / 0,3608 | 1000 | 0,3339 | 300 | 0,3608 |
| ETTm1 | 192 | NLinear | MSE | 0,3357 / 0,3634 | 5623 | 0,3356 | 1995 | 0,3634 |
| ETTm1 | 192 | NLinear | MAE | 0,3343 / 0,3563 | 1e+04 | **0,3335** | 3000 | **0,3563** |
| ETTm1 | 192 | NLinear | Huber | 0,3361 / 0,3613 | 3000 | 0,3359 | 1000 | 0,3613 |
| ETTm1 | 336 | Linear | MSE | 0,3684 / 0,3844 | 446,7 | 0,3683 | 354,8 | 0,3843 |
| ETTm1 | 336 | Linear | MAE | 0,3659 / 0,3759 | 1000 | **0,3658** | 300 | **0,3759** |
| ETTm1 | 336 | Linear | Huber | 0,3673 / 0,3809 | 100 | 0,3672 | 100 | 0,3808 |
| ETTm1 | 336 | DLinear | MSE | 0,3684 / 0,3844 | 1259 | 0,3683 | 446,7 | 0,3843 |
| ETTm1 | 336 | DLinear | MAE | 0,3659 / 0,3759 | 3000 | **0,3655** | 300 | **0,3759** |
| ETTm1 | 336 | DLinear | Huber | 0,3673 / 0,3809 | 300 | 0,3672 | 100 | 0,3809 |
| ETTm1 | 336 | NLinear | MSE | 0,3704 / 0,3834 | 6310 | 0,3703 | 1585 | 0,3834 |
| ETTm1 | 336 | NLinear | MAE | 0,3691 / 0,3770 | 1e+04 | **0,3685** | 1000 | **0,3770** |
| ETTm1 | 336 | NLinear | Huber | 0,3701 / 0,3812 | 3000 | 0,3699 | 300 | 0,3812 |
| ETTm1 | 720 | Linear | MSE | 0,4233 / 0,4181 | 1259 | 0,4232 | 631 | 0,4181 |
| ETTm1 | 720 | Linear | MAE | 0,4218 / 0,4102 | 1000 | 0,4216 | 1000 | **0,4101** |
| ETTm1 | 720 | Linear | Huber | 0,4214 / 0,4140 | 300 | **0,4214** | 100 | 0,4140 |
| ETTm1 | 720 | DLinear | MSE | 0,4233 / 0,4181 | 2512 | 0,4232 | 794,3 | 0,4181 |
| ETTm1 | 720 | DLinear | MAE | 0,4218 / 0,4102 | 3000 | 0,4213 | 1000 | **0,4101** |
| ETTm1 | 720 | DLinear | Huber | 0,4214 / 0,4140 | 1000 | **0,4213** | 100 | 0,4140 |
| ETTm1 | 720 | NLinear | MSE | 0,4263 / 0,4157 | 7079 | 0,4262 | 1585 | 0,4157 |
| ETTm1 | 720 | NLinear | MAE | 0,4281 / 0,4114 | 3e+04 | 0,4270 | 1000 | **0,4113** |
| ETTm1 | 720 | NLinear | Huber | 0,4263 / 0,4139 | 3000 | **0,4261** | 300 | 0,4138 |

## Thứ hạng mục tiêu trong từng ô B3: cũ (val_objective) so với mới

Thứ hạng từ tốt đến kém theo MSE test và theo MAE test. Chỉ liệt kê ô có thứ hạng đổi.

| Dataset | H | Mô hình | metric | cũ | mới | giá trị mới (MSE / MAE / Huber) |
|---|---|---|---|---|---|---|
| ETTh1 | 192 | Linear | MSE test | Huber < MSE < MAE | Huber < MAE < MSE | 0,4040 / 0,4031 / 0,4026 |
| ETTh1 | 192 | DLinear | MSE test | Huber < MSE < MAE | MAE < Huber < MSE | 0,4036 / 0,4014 / 0,4022 |
| ETTh1 | 192 | NLinear | MSE test | Huber < MSE < MAE | Huber < MAE < MSE | 0,4027 / 0,4026 / 0,4025 |
| ETTh1 | 336 | Linear | MSE test | Huber < MSE < MAE | MAE < Huber < MSE | 0,4324 / 0,4302 / 0,4304 |
| ETTh1 | 336 | DLinear | MSE test | Huber < MSE < MAE | MAE < Huber < MSE | 0,4317 / 0,4296 / 0,4300 |
| ETTm1 | 192 | DLinear | MSE test | MAE < MSE < Huber | MAE < Huber < MSE | 0,3339 / 0,3319 / 0,3339 |
| ETTm1 | 720 | NLinear | MSE test | MSE < Huber < MAE | Huber < MSE < MAE | 0,4262 / 0,4270 / 0,4261 |

Thứ hạng đổi ở 7/36 ô theo MSE test (mục tiêu tốt nhất đổi ở 4) và 0/36 ô theo MAE test (mục tiêu tốt nhất đổi ở 0). Vị trí của Huber trong các thứ hạng này **chờ Phần A** (δ_MAD).

## Mục tiêu tốt nhất trong ô (số ô trên 12 mỗi dataset × mô hình)

| Dataset | Mô hình | MSE test, cũ: MSE / MAE / Huber | MSE test, mới | MAE test, cũ | MAE test, mới |
|---|---|---|---|---|---|
| ETTh1 | Linear | 0 / 2 / 2 | 0 / 3 / 1 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTh1 | DLinear | 0 / 2 / 2 | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTh1 | NLinear | 1 / 2 / 1 | 1 / 2 / 1 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTh2 | Linear | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTh2 | DLinear | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTh2 | NLinear | 3 / 0 / 1 | 3 / 0 / 1 | 0 / 2 / 2 | 0 / 2 / 2 |
| ETTm1 | Linear | 0 / 3 / 1 | 0 / 3 / 1 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTm1 | DLinear | 0 / 3 / 1 | 0 / 3 / 1 | 0 / 4 / 0 | 0 / 4 / 0 |
| ETTm1 | NLinear | 1 / 3 / 0 | 0 / 3 / 1 | 0 / 4 / 0 | 0 / 4 / 0 |

## "Huấn luyện bằng MAE cho MSE test tốt hơn huấn luyện bằng MSE": đếm lại

Số H (trên 4) có MSE test(MAE) < MSE test(MSE). Cũ: mỗi mục tiêu ở λ* theo metric val của chính nó (MAE theo val MAE). Mới: cả hai ở λ*_MSE. Chênh = MSE test(MSE) − MSE test(MAE) theo từng H (dương là MAE tốt hơn).

| Dataset | Mô hình | cũ: số H | cũ: chênh theo H = 96/192/336/720 | mới: số H | mới: chênh |
|---|---|---|---|---|---|
| ETTh1 | Linear | 2 | 0,0049 / −0,0001 / −0,0026 / 0,0181 | 4 | 0,0049 / 0,0009 / 0,0022 / 0,0147 |
| ETTh1 | DLinear | 2 | 0,0047 / −0,0005 / −0,0008 / 0,0181 | 4 | 0,0051 / 0,0021 / 0,0022 / 0,0148 |
| ETTh1 | NLinear | 2 | 0,0038 / −0,0021 / −0,0056 / 0,0067 | 3 | 0,0057 / 0,0002 / −0,0028 / 0,0047 |
| ETTh2 | Linear | 4 | 0,0230 / 0,0435 / 0,0748 / 0,1511 | 4 | 0,0230 / 0,0435 / 0,0748 / 0,1511 |
| ETTh2 | DLinear | 4 | 0,0230 / 0,0435 / 0,0747 / 0,1511 | 4 | 0,0230 / 0,0435 / 0,0747 / 0,1511 |
| ETTh2 | NLinear | 0 | −0,0054 / −0,0117 / −0,0185 / −0,0007 | 0 | −0,0047 / −0,0110 / −0,0178 / −0,0002 |
| ETTm1 | Linear | 4 | 0,0063 / 0,0017 / 0,0024 / 0,0016 | 4 | 0,0063 / 0,0018 / 0,0026 / 0,0016 |
| ETTm1 | DLinear | 4 | 0,0063 / 0,0016 / 0,0024 / 0,0016 | 4 | 0,0063 / 0,0020 / 0,0028 / 0,0018 |
| ETTm1 | NLinear | 3 | 0,0066 / 0,0016 / 0,0013 / −0,0017 | 3 | 0,0067 / 0,0020 / 0,0018 / −0,0009 |

## Thí nghiệm outlier: mục tiêu tốt nhất theo tiêu chí cũ và mới (Huber δ = 1)

Chèn vào chuỗi (seed 2026; rate 0 là grid gốc) và chèn chỉ vào đích (H = 96, 720). Ô ghi: tốt nhất theo MSE test / theo MAE test, cũ → mới. In đậm khi đổi.

| Mô hình | H | chế độ | rate 0% | rate 1% | rate 5% | rate 10% |
|---|---|---|---|---|---|---|
| Linear | 96 | series | MAE / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| Linear | 96 | target | MAE / MAE | MAE / MAE | — | MAE / MAE |
| Linear | 192 | series | Huber / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| Linear | 192 | target | — | — | — | — |
| Linear | 336 | series | **Huber → MAE** / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| Linear | 336 | target | — | — | — | — |
| Linear | 720 | series | MAE / MAE | MAE / MAE | Huber / MAE | MSE / MSE |
| Linear | 720 | target | MAE / MAE | MAE / MAE | — | MAE / MAE |
| NLinear | 96 | series | MAE / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| NLinear | 96 | target | MAE / MAE | MAE / MAE | — | MAE / MAE |
| NLinear | 192 | series | Huber / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| NLinear | 192 | target | — | — | — | — |
| NLinear | 336 | series | MSE / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| NLinear | 336 | target | — | — | — | — |
| NLinear | 720 | series | MAE / MAE | MAE / MAE | MSE / MSE | MSE / MSE |
| NLinear | 720 | target | MAE / MAE | MAE / MAE | — | MAE / MAE |

Số ô có mục tiêu tốt nhất đổi (rate > 0, cả hai metric): 0.

### Đếm mục tiêu tốt nhất theo MSE test, chèn vào chuỗi (8 ô mô hình × H): MSE / MAE / Huber

| rate | cũ | mới |
|---|---|---|
| 0% | 1 / 4 / 3 | 1 / 5 / 2 |
| 1% | 0 / 8 / 0 | 0 / 8 / 0 |
| 5% | 7 / 0 / 1 | 7 / 0 / 1 |
| 10% | 8 / 0 / 0 | 8 / 0 / 0 |

Theo MAE test:

| rate | cũ | mới |
|---|---|---|
| 0% | 0 / 8 / 0 | 0 / 8 / 0 |
| 1% | 0 / 8 / 0 | 0 / 8 / 0 |
| 5% | 7 / 1 / 0 | 7 / 1 / 0 |
| 10% | 8 / 0 / 0 | 8 / 0 / 0 |

### Suy giảm MSE test 0% → 10% (chuỗi, seed 2026): khoảng trên 8 ô, cũ → mới

| Mục tiêu | cũ | mới |
|---|---|---|
| MSE | 0,0591 đến 0,0769 | 0,0591 đến 0,0769 |
| MAE | 0,0730 đến 0,1136 | 0,0758 đến 0,1136 |
| Huber | 0,0703 đến 0,1004 | 0,0704 đến 0,1004 |

Thứ tự suy giảm MSE < Huber < MAE (mới) ở 8/8 ô.


## λ* của DLinear, ETTh1, H = 96

| Mục tiêu | val_objective (cũ) | λ*_MSE (mới, cho MSE test) | λ*_MAE (mới, cho MAE test) | MSE test cũ → mới | MAE test cũ → mới |
|---|---|---|---|---|---|
| MSE | 446,7 | 446,7 | 446,7 | 0,3698 → 0,3698 | 0,3914 → 0,3914 |
| MAE | 100 | 300 | 100 | 0,3651 → 0,3647 | 0,3832 → 0,3832 |
| Huber | 100 | 300 | 100 | 0,3675 → 0,3671 | 0,3879 → 0,3879 |
