# Phần D: SGD đã tinh chỉnh learning rate

Chọn lr ∈ {0,5; 1; 2; 4} × lr gốc (thêm ×8 nếu ×4 thắng) theo trung bình val MSE của seed 2021–2025; chạy cuối seed 2026–2040 (15 seed mới). Dạng đóng: MSE λ* theo val MSE (results/results.csv), như B4.

## Bảng lr đã chọn

| Dataset | Mô hình | H | lr gốc | val MSE TB (seed 2021–2025): ×0,5 / ×1 / ×2 / ×4 / ×8 | chọn | lr chọn |
|---|---|---|---|---|---|---|
| ETTh2 | Linear | 96 | 0,05 | 0,21544 / **0,21510** / 0,21651 / 0,21582 / — | ×1 | 0,05 |
| ETTh2 | Linear | 192 | 0,05 | 0,30634 / 0,30376 / 0,30952 / **0,29600** / 0,29778 | ×4 | 0,2 |
| ETTh2 | Linear | 336 | 0,05 | 0,40058 / **0,39459** / 0,39561 / 0,39674 / — | ×1 | 0,05 |
| ETTh2 | Linear | 720 | 0,05 | **0,65305** / 0,65570 / 0,65493 / 0,65583 / — | ×0,5 | 0,025 |
| ETTh2 | DLinear | 96 | 0,05 | 0,21744 / **0,21560** / 0,21608 / 0,21704 / — | ×1 | 0,05 |
| ETTh2 | DLinear | 192 | 0,05 | **0,29584** / 0,29798 / 0,30046 / 0,30166 / — | ×0,5 | 0,025 |
| ETTh2 | DLinear | 336 | 0,05 | 0,39633 / **0,39625** / 0,39853 / 0,40029 / — | ×1 | 0,05 |
| ETTh2 | DLinear | 720 | 0,05 | 0,66128 / **0,66116** / 0,66134 / 0,66229 / — | ×1 | 0,05 |
| ETTh2 | NLinear | 96 | 0,05 | **0,20546** / 0,20596 / 0,20805 / 0,21472 / — | ×0,5 | 0,025 |
| ETTh2 | NLinear | 192 | 0,05 | **0,27078** / 0,27205 / 0,27723 / 0,29756 / — | ×0,5 | 0,025 |
| ETTh2 | NLinear | 336 | 0,05 | **0,35856** / 0,36058 / 0,36377 / 0,37028 / — | ×0,5 | 0,025 |
| ETTh2 | NLinear | 720 | 0,05 | **0,60372** / 0,60988 / 0,60731 / 0,61500 / — | ×0,5 | 0,025 |
| ETTh1 | Linear | 96 | 0,005 | 0,65027 / 0,64858 / 0,64622 / **0,64476** / 0,65966 | ×4 | 0,02 |
| ETTh1 | Linear | 192 | 0,005 | 0,86546 / 0,86607 / 0,86452 / 0,85931 / **0,85630** | ×8 | 0,04 |
| ETTh1 | Linear | 336 | 0,005 | 1,05258 / **1,04150** / 1,04155 / 1,04447 / — | ×1 | 0,005 |
| ETTh1 | Linear | 720 | 0,005 | **1,19603** / 1,20746 / 1,21314 / 1,20010 / — | ×0,5 | 0,0025 |
| ETTh1 | DLinear | 96 | 0,005 | 0,64868 / **0,64800** / 0,64832 / 0,64869 / — | ×1 | 0,005 |
| ETTh1 | DLinear | 192 | 0,005 | 0,87278 / 0,86432 / **0,85435** / 0,85653 / — | ×2 | 0,01 |
| ETTh1 | DLinear | 336 | 0,005 | **1,04502** / 1,05251 / 1,04932 / 1,07283 / — | ×0,5 | 0,0025 |
| ETTh1 | DLinear | 720 | 0,005 | 1,21328 / 1,22226 / 1,23989 / 1,20324 / **1,20300** | ×8 | 0,04 |
| ETTh1 | NLinear | 96 | 0,005 | 0,66810 / 0,67707 / 0,68218 / **0,66729** / 0,66757 | ×4 | 0,02 |
| ETTh1 | NLinear | 192 | 0,005 | 0,91652 / 0,92104 / 0,91292 / **0,91271** / 0,91301 | ×4 | 0,02 |
| ETTh1 | NLinear | 336 | 0,005 | 1,15682 / 1,15430 / **1,15238** / 1,15314 / — | ×2 | 0,01 |
| ETTh1 | NLinear | 720 | 0,005 | 1,43078 / 1,42915 / **1,42911** / 1,42925 / — | ×2 | 0,01 |
| ETTm1 | Linear | 96 | 0,0001 | 0,39224 / 0,38794 / 0,38459 / 0,38202 / **0,38153** | ×8 | 0,0008 |
| ETTm1 | Linear | 192 | 0,0001 | 0,49933 / 0,49453 / 0,49141 / 0,49127 / **0,48880** | ×8 | 0,0008 |
| ETTm1 | Linear | 336 | 0,0001 | 0,62508 / 0,61983 / 0,61640 / 0,61183 / **0,61064** | ×8 | 0,0008 |
| ETTm1 | Linear | 720 | 0,0001 | 0,87618 / 0,86861 / **0,86117** / 0,86167 / — | ×2 | 0,0002 |
| ETTm1 | DLinear | 96 | 0,0001 | 0,38458 / 0,38265 / **0,38130** / 0,38170 / — | ×2 | 0,0002 |
| ETTm1 | DLinear | 192 | 0,0001 | 0,49220 / 0,48919 / **0,48790** / 0,48791 / — | ×2 | 0,0002 |
| ETTm1 | DLinear | 336 | 0,0001 | 0,61730 / 0,61517 / 0,61419 / **0,61347** / 0,61766 | ×4 | 0,0004 |
| ETTm1 | DLinear | 720 | 0,0001 | 0,86811 / 0,86182 / **0,85238** / 0,85631 / — | ×2 | 0,0002 |
| ETTm1 | NLinear | 96 | 0,0001 | 0,39583 / 0,39280 / 0,39033 / 0,38771 / **0,38447** | ×8 | 0,0008 |
| ETTm1 | NLinear | 192 | 0,0001 | 0,51295 / 0,50957 / 0,50603 / **0,50457** / 0,50508 | ×4 | 0,0004 |
| ETTm1 | NLinear | 336 | 0,0001 | 0,65736 / 0,65391 / 0,65186 / **0,65031** / 0,65129 | ×4 | 0,0004 |
| ETTm1 | NLinear | 720 | 0,0001 | 0,96775 / 0,96488 / 0,96247 / 0,96142 / **0,96073** | ×8 | 0,0008 |

Số ô theo mức chọn: ×0,5: 8, ×1: 7, ×2: 7, ×4: 7, ×8: 7.

## B4-tinh chỉnh: SGD đã tinh chỉnh lr (15 seed 2026–2040) so với dạng đóng λ*

Δ = TB SGD − dạng đóng, CI 95% theo t (14 bậc tự do). Cột "×1, 2026–2040": lr gốc trên đúng 15 seed đó (không chạy mới), để tách phần nhãn đổi do lr với phần do bớt seed. B4 cũ: lr gốc, 20 seed.

| Dataset | H | Mô hình | lr | Δ val [CI] | Δ test [CI] | nhãn (tinh chỉnh) | nhãn ×1, 2026–2040 | nhãn B4 cũ |
|---|---|---|---|---|---|---|---|---|
| ETTh2 | 96 | Linear | ×1 | −0,0029 [−0,0069; 0,0010] | −0,0020 [−0,0136; 0,0095] | không phân biệt được | không phân biệt được | SGD có dấu hiệu khớp val quá mức |
| ETTh2 | 96 | DLinear | ×1 | −0,0042 [−0,0059; −0,0025] | −0,0083 [−0,0111; −0,0054] | SGD tốt hơn thật | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 96 | NLinear | ×0,5 | 0,0015 [0,0010; 0,0020] | 0,0052 [0,0036; 0,0068] | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật (n = 5) | dạng đóng tốt hơn thật |
| ETTh2 | 192 | Linear | ×4 | −0,0004 [−0,0134; 0,0126] | −0,0025 [−0,0280; 0,0230] | không phân biệt được | không phân biệt được | không phân biệt được |
| ETTh2 | 192 | DLinear | ×0,5 | −0,0067 [−0,0097; −0,0037] | −0,0220 [−0,0301; −0,0138] | SGD tốt hơn thật | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 192 | NLinear | ×0,5 | −0,0003 [−0,0016; 0,0009] | 0,0131 [0,0089; 0,0173] | khác biệt do val khác test | khác biệt do val khác test (n = 5) | khác biệt do val khác test |
| ETTh2 | 336 | Linear | ×1 | −0,0129 [−0,0192; −0,0067] | −0,0345 [−0,0476; −0,0215] | SGD tốt hơn thật | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 336 | DLinear | ×1 | −0,0094 [−0,0127; −0,0060] | −0,0338 [−0,0469; −0,0207] | SGD tốt hơn thật | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 336 | NLinear | ×0,5 | 0,0009 [−0,0011; 0,0030] | 0,0191 [0,0106; 0,0275] | khác biệt do val khác test | khác biệt do val khác test (n = 5) | khác biệt do val khác test |
| ETTh2 | 720 | Linear | ×0,5 | −0,0040 [−0,0143; 0,0063] | −0,0314 [−0,0750; 0,0122] | không phân biệt được | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 720 | DLinear | ×1 | −0,0084 [−0,0134; −0,0034] | −0,0507 [−0,0805; −0,0209] | SGD tốt hơn thật | SGD tốt hơn thật | SGD tốt hơn thật |
| ETTh2 | 720 | NLinear | ×0,5 | −0,0021 [−0,0069; 0,0027] | 0,0159 [0,0086; 0,0232] | khác biệt do val khác test | khác biệt do val khác test (n = 5) | khác biệt do val khác test |
| ETTh1 | 96 | Linear | ×4 | −0,0066 [−0,0091; −0,0041] | 0,0060 [0,0029; 0,0091] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 96 | DLinear | ×1 | −0,0059 [−0,0086; −0,0031] | 0,0067 [0,0034; 0,0100] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 96 | NLinear | ×4 | −0,0051 [−0,0072; −0,0031] | 0,0067 [0,0037; 0,0097] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test |
| ETTh1 | 192 | Linear | ×8 | 0,0031 [−0,0076; 0,0139] | 0,0216 [0,0057; 0,0374] | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 192 | DLinear | ×2 | −0,0044 [−0,0090; 0,0003] | 0,0075 [0,0038; 0,0111] | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 192 | NLinear | ×4 | 0,0052 [−0,0086; 0,0190] | 0,0084 [−0,0004; 0,0173] | không phân biệt được | khác biệt do val khác test | khác biệt do val khác test |
| ETTh1 | 336 | Linear | ×1 | −0,0095 [−0,0147; −0,0044] | 0,0259 [0,0142; 0,0375] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 336 | DLinear | ×0,5 | −0,0131 [−0,0216; −0,0046] | 0,0222 [0,0074; 0,0371] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 336 | NLinear | ×2 | −0,0005 [−0,0018; 0,0008] | 0,0039 [0,0024; 0,0055] | khác biệt do val khác test | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTh1 | 720 | Linear | ×0,5 | −0,0095 [−0,0169; −0,0021] | 0,0290 [0,0178; 0,0402] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 720 | DLinear | ×8 | −0,0038 [−0,0084; 0,0009] | 0,0198 [0,0100; 0,0296] | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test |
| ETTh1 | 720 | NLinear | ×2 | 0,0036 [0,0031; 0,0042] | 0,0025 [0,0022; 0,0028] | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 96 | Linear | ×8 | −0,0010 [−0,0017; −0,0002] | 0,0022 [0,0008; 0,0036] | SGD có dấu hiệu khớp val quá mức | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 96 | DLinear | ×2 | −0,0013 [−0,0018; −0,0007] | 0,0025 [0,0011; 0,0039] | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test | khác biệt do val khác test |
| ETTm1 | 96 | NLinear | ×8 | −0,0012 [−0,0019; −0,0005] | 0,0047 [0,0026; 0,0068] | SGD có dấu hiệu khớp val quá mức | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 192 | Linear | ×8 | −0,0017 [−0,0028; −0,0006] | 0,0031 [0,0015; 0,0048] | SGD có dấu hiệu khớp val quá mức | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 192 | DLinear | ×2 | −0,0019 [−0,0024; −0,0013] | 0,0022 [0,0014; 0,0029] | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 192 | NLinear | ×4 | −0,0007 [−0,0024; 0,0011] | 0,0063 [0,0037; 0,0089] | khác biệt do val khác test | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 336 | Linear | ×8 | −0,0028 [−0,0047; −0,0009] | 0,0084 [0,0039; 0,0128] | SGD có dấu hiệu khớp val quá mức | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 336 | DLinear | ×4 | −0,0036 [−0,0059; −0,0014] | 0,0064 [0,0032; 0,0097] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 336 | NLinear | ×4 | −0,0007 [−0,0021; 0,0007] | 0,0070 [0,0031; 0,0109] | khác biệt do val khác test | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |
| ETTm1 | 720 | Linear | ×2 | −0,0096 [−0,0128; −0,0065] | 0,0100 [0,0069; 0,0131] | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 720 | DLinear | ×2 | −0,0111 [−0,0158; −0,0064] | 0,0080 [0,0056; 0,0104] | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 720 | NLinear | ×8 | −0,0019 [−0,0028; −0,0009] | 0,0048 [0,0019; 0,0078] | SGD có dấu hiệu khớp val quá mức | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật |

| nhãn | tinh chỉnh (15 seed) | ×1, 2026–2040 | B4 cũ (×1, 20 seed) |
|---|---|---|---|
| SGD tốt hơn thật | 5 | 6 | 6 |
| dạng đóng tốt hơn thật | 2 | 10 | 10 |
| khác biệt do val khác test | 9 | 7 | 7 |
| SGD có dấu hiệu khớp val quá mức | 16 | 11 | 12 |
| dạng đóng có dấu hiệu khớp val quá mức | 0 | 0 | 0 |
| không phân biệt được | 4 | 2 | 1 |

SGD đã tinh chỉnh vẫn kém dạng đóng trên test: 28/36 ô theo trung bình, 27/36 ô có ý nghĩa (CI nằm trên 0). "SGD tốt hơn thật": 5 ô (ETTh2 H=96 DLinear, ETTh2 H=192 DLinear, ETTh2 H=336 Linear, ETTh2 H=336 DLinear, ETTh2 H=720 DLinear).

Nhãn đổi so với B4 cũ: 15/36 ô; trong đó 13 ô nhãn tinh chỉnh khác nhãn ×1 trên cùng 15 seed (đổi do lr), 2 ô nhãn ×1 trên 15 seed đã khác B4 cũ (đổi do bớt seed).

| Dataset | H | Mô hình | lr | B4 cũ | ×1, 2026–2040 | tinh chỉnh |
|---|---|---|---|---|---|---|
| ETTh2 | 96 | Linear | ×1 | SGD có dấu hiệu khớp val quá mức | không phân biệt được | không phân biệt được |
| ETTh2 | 720 | Linear | ×0,5 | SGD tốt hơn thật | SGD tốt hơn thật | không phân biệt được |
| ETTh1 | 96 | NLinear | ×4 | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức |
| ETTh1 | 192 | Linear | ×8 | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test |
| ETTh1 | 192 | DLinear | ×2 | SGD có dấu hiệu khớp val quá mức | SGD có dấu hiệu khớp val quá mức | khác biệt do val khác test |
| ETTh1 | 192 | NLinear | ×4 | khác biệt do val khác test | khác biệt do val khác test | không phân biệt được |
| ETTh1 | 336 | NLinear | ×2 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | khác biệt do val khác test |
| ETTm1 | 96 | Linear | ×8 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 96 | DLinear | ×2 | khác biệt do val khác test | khác biệt do val khác test | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 96 | NLinear | ×8 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 192 | Linear | ×8 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 192 | NLinear | ×4 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | khác biệt do val khác test |
| ETTm1 | 336 | Linear | ×8 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | SGD có dấu hiệu khớp val quá mức |
| ETTm1 | 336 | NLinear | ×4 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | khác biệt do val khác test |
| ETTm1 | 720 | NLinear | ×8 | dạng đóng tốt hơn thật | dạng đóng tốt hơn thật | SGD có dấu hiệu khớp val quá mức |

Ở 29 ô có lr chọn khác ×1: TB test (tinh chỉnh) − TB test (×1), cùng 15 seed: −0,0223 đến 0,0141 (âm ở 17 ô).

## C.3-tinh chỉnh: DLinear − Linear khi cả hai được tinh chỉnh lr (15 seed mỗi mô hình, CI Welch)

| Dataset | H | lr Linear / DLinear | Δ val [CI] | Δ test [CI] | DLinear tốt hơn có ý nghĩa (val / test) | ×1, 2026–2040: Δ val / Δ test [CI] | C.3 cũ (×1, 20 seed): Δ test [CI] |
|---|---|---|---|---|---|---|---|
| ETTh2 | 96 | ×1 / ×1 | −0,0012 [−0,0054; 0,0030] | −0,0064 [−0,0182; 0,0054] | không / không | −0,0012 [−0,0054; 0,0030] / −0,0064 [−0,0182; 0,0054] | −0,0055 [−0,0148; 0,0037] |
| ETTh2 | 192 | ×4 / ×0,5 | −0,0063 [−0,0196; 0,0069] | −0,0195 [−0,0459; 0,0069] | không / không | −0,0009 [−0,0125; 0,0107] / −0,0096 [−0,0335; 0,0144] | −0,0154 [−0,0383; 0,0075] |
| ETTh2 | 336 | ×1 / ×1 | 0,0036 [−0,0033; 0,0105] | 0,0007 [−0,0170; 0,0184] | không / không | 0,0036 [−0,0033; 0,0105] / 0,0007 [−0,0170; 0,0184] | 0,0033 [−0,0122; 0,0187] |
| ETTh2 | 720 | ×0,5 / ×1 | −0,0044 [−0,0155; 0,0067] | −0,0193 [−0,0700; 0,0314] | không / không | −0,0014 [−0,0076; 0,0049] / −0,0052 [−0,0457; 0,0354] | −0,0098 [−0,0444; 0,0249] |
| ETTh1 | 96 | ×4 / ×1 | 0,0006 [−0,0029; 0,0042] | 0,0003 [−0,0040; 0,0046] | không / không | −0,0019 [−0,0052; 0,0013] / −0,0041 [−0,0112; 0,0029] | −0,0020 [−0,0075; 0,0036] |
| ETTh1 | 192 | ×8 / ×2 | −0,0077 [−0,0191; 0,0037] | −0,0145 [−0,0307; 0,0016] | không / không | 0,0003 [−0,0036; 0,0043] / −0,0094 [−0,0217; 0,0029] | −0,0105 [−0,0227; 0,0017] |
| ETTh1 | 336 | ×1 / ×0,5 | −0,0041 [−0,0137; 0,0055] | −0,0043 [−0,0224; 0,0138] | không / không | −0,0004 [−0,0068; 0,0061] / −0,0081 [−0,0267; 0,0106] | 0,0020 [−0,0150; 0,0190] |
| ETTh1 | 720 | ×0,5 / ×8 | 0,0051 [−0,0032; 0,0135] | −0,0096 [−0,0238; 0,0046] | không / không | 0,0032 [−0,0090; 0,0154] / 0,0099 [−0,0152; 0,0349] | 0,0033 [−0,0210; 0,0276] |
| ETTm1 | 96 | ×8 / ×2 | −0,0003 [−0,0012; 0,0005] | 0,0002 [−0,0017; 0,0022] | không / không | −0,0056 [−0,0063; −0,0049] / −0,0041 [−0,0054; −0,0028] | −0,0040 [−0,0052; −0,0028] |
| ETTm1 | 192 | ×8 / ×2 | −0,0002 [−0,0014; 0,0010] | −0,0010 [−0,0028; 0,0008] | không / không | −0,0054 [−0,0061; −0,0048] / −0,0032 [−0,0040; −0,0023] | −0,0032 [−0,0041; −0,0024] |
| ETTm1 | 336 | ×8 / ×4 | −0,0009 [−0,0037; 0,0019] | −0,0020 [−0,0073; 0,0033] | không / không | −0,0053 [−0,0076; −0,0030] / −0,0034 [−0,0054; −0,0014] | −0,0038 [−0,0054; −0,0021] |
| ETTm1 | 720 | ×2 / ×2 | −0,0015 [−0,0069; 0,0040] | −0,0020 [−0,0058; 0,0017] | không / không | −0,0036 [−0,0074; 0,0002] / −0,0038 [−0,0062; −0,0014] | −0,0047 [−0,0069; −0,0025] |

DLinear tốt hơn có ý nghĩa ở cả val lẫn test: 0/12 ô (×1 trên cùng 15 seed 2026–2040: 3/12; cũ, 20 seed: 4/12, đều ETTm1). Bốn ô ETTm1 cũ: H=96: test 0,0002 [−0,0017; 0,0022] (không còn có ý nghĩa); H=192: test −0,0010 [−0,0028; 0,0008] (không còn có ý nghĩa); H=336: test −0,0020 [−0,0073; 0,0033] (không còn có ý nghĩa); H=720: test −0,0020 [−0,0058; 0,0017] (không còn có ý nghĩa).

