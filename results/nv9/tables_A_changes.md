# Phần A: mở rộng lưới lr ở 15 ô biên

| Ô | biên | mức mới: val MSE TB (5 seed) | lr cũ → mới | Δ test cũ → mới [CI] | nhãn cũ → mới | còn ở biên |
|---|---|---|---|---|---|---|
| ETTm1 Linear H=96 | trên (×8: 0,38153) | ×16: 0,38133; ×32: 0,38123 | ×8 → ×32 | 0,0022 [±0,0014] → 0,0020 [±0,0015] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | có |
| ETTm1 Linear H=192 | trên (×8: 0,48880) | ×16: 0,48898 | ×8 → ×8 | 0,0031 [±0,0017] → 0,0031 [±0,0017] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTm1 Linear H=336 | trên (×8: 0,61064) | ×16: 0,61634 | ×8 → ×8 | 0,0084 [±0,0045] → 0,0084 [±0,0045] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTm1 NLinear H=96 | trên (×8: 0,38447) | ×16: 0,38729 | ×8 → ×8 | 0,0047 [±0,0021] → 0,0047 [±0,0021] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTm1 NLinear H=720 | trên (×8: 0,96073) | ×16: 0,95823; ×32: 0,95855 | ×8 → ×16 | 0,0048 [±0,0029] → 0,0041 [±0,0021] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTh1 DLinear H=336 | dưới (×0,5: 1,04502) | ×0,25: 1,05106 | ×0,5 → ×0,5 | 0,0222 [±0,0149] → 0,0222 [±0,0149] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTh1 DLinear H=720 | trên (×8: 1,20300) | ×16: 1,20653 | ×8 → ×8 | 0,0198 [±0,0098] → 0,0198 [±0,0098] | khác biệt do val khác test → khác biệt do val khác test | không |
| ETTh1 Linear H=192 | trên (×8: 0,85630) | ×16: 0,85694 | ×8 → ×8 | 0,0216 [±0,0158] → 0,0216 [±0,0158] | khác biệt do val khác test → khác biệt do val khác test | không |
| ETTh1 Linear H=720 | dưới (×0,5: 1,19603) | ×0,25: 1,20811 | ×0,5 → ×0,5 | 0,0290 [±0,0112] → 0,0290 [±0,0112] | SGD có dấu hiệu khớp val quá mức → SGD có dấu hiệu khớp val quá mức | không |
| ETTh2 DLinear H=192 | dưới (×0,5: 0,29584) | ×0,25: 0,29631 | ×0,5 → ×0,5 | -0,0220 [±0,0082] → -0,0220 [±0,0082] | SGD tốt hơn thật → SGD tốt hơn thật | không |
| ETTh2 Linear H=720 | dưới (×0,5: 0,65305) | ×0,25: 0,65592 | ×0,5 → ×0,5 | -0,0314 [±0,0436] → -0,0314 [±0,0436] | không phân biệt được → không phân biệt được | không |
| ETTh2 NLinear H=96 | dưới (×0,5: 0,20546) | ×0,25: 0,20553 | ×0,5 → ×0,5 | 0,0052 [±0,0016] → 0,0052 [±0,0016] | dạng đóng tốt hơn thật → dạng đóng tốt hơn thật | không |
| ETTh2 NLinear H=192 | dưới (×0,5: 0,27078) | ×0,25: 0,27016; ×0,125: 0,27106 | ×0,5 → ×0,25 | 0,0131 [±0,0042] → 0,0137 [±0,0036] | khác biệt do val khác test → SGD có dấu hiệu khớp val quá mức | không |
| ETTh2 NLinear H=336 | dưới (×0,5: 0,35856) | ×0,25: 0,36019 | ×0,5 → ×0,5 | 0,0191 [±0,0085] → 0,0191 [±0,0085] | khác biệt do val khác test → khác biệt do val khác test | không |
| ETTh2 NLinear H=720 | dưới (×0,5: 0,60372) | ×0,25: 0,60360; ×0,125: 0,60128 | ×0,5 → ×0,125 | 0,0159 [±0,0073] → 0,0151 [±0,0051] | khác biệt do val khác test → khác biệt do val khác test | có |

SGD kém dạng đóng có ý nghĩa trên test: 27/36 → 27/36; ETTh1 + ETTm1: 23/24 → 23/24. "SGD tốt hơn thật" ngoài ETTh2: không có. Điểm dừng RQ3: không.
C.3 (DLinear − Linear, cả hai tinh chỉnh): DLinear tốt hơn có ý nghĩa 0/12 → 0/12.

| nhãn | nv7 | nv9 |
|---|---|---|
| SGD tốt hơn thật | 5 | 5 |
| dạng đóng tốt hơn thật | 2 | 2 |
| khác biệt do val khác test | 9 | 8 |
| SGD có dấu hiệu khớp val quá mức | 16 | 17 |
| dạng đóng có dấu hiệu khớp val quá mức | 0 | 0 |
| không phân biệt được | 4 | 4 |
