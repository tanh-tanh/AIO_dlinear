# C.3: Huber với δ theo bước h

MSE test ở λ*_MSE (argmin val MSE trên đường λ của chính mục tiêu), MAE test ở λ*_MAE.

| Ô | δ_MAD | δ_h: min–max (TB) | MSE | MAE | Huber δ_MAD | Huber δ_h | vị trí Huber δ_MAD / δ_h (MSE test; MAE test) |
|---|---|---|---|---|---|---|---|
| ETTh1 H=96 Linear | 0,5380 | 0,2426–0,5952 (0,5415) | 0,3702 / 0,3915 | 0,3653 / 0,3833 | 0,3658 / 0,3852 | 0,3656 / 0,3851 | 2 / 2; 2 / 2 |
| ETTh2 H=720 Linear | 0,6096 | 0,1743–0,8010 (0,6134) | 0,7404 / 0,6115 | 0,5893 / 0,5335 | 0,6277 / 0,5541 | 0,6357 / 0,5566 | 2 / 2; 2 / 2 |

Ô có dạng "MSE / MAE test"; vị trí: hạng của Huber giữa ba mục tiêu (1 = thấp nhất). Vị trí đổi khi dùng δ_h: 0/2 ô.
