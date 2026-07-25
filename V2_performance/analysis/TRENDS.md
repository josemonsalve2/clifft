# Progressive V2/SVM kernel-time ratio (lower = better; <1 = V2 wins)

| circuit | 20260725T030000Z_baseline-preopt | 20260725T051154Z_after-P0-P1 | 20260725T082430Z_after-specializer | 20260725T083507Z_specializer-final | 20260725T162524Z_noise-fenced-gated | 20260725T163547Z_noise-specialized |
|---|---|---|---|---|---|---|
| circuit_d3_p0.001 | 15.193 | 1.081 | 1.116 | 1.103 | 2.117 | 0.524 |
| frame_h | 28.258 | 1.215 | 0.612 | 0.594 | 2.859 | 0.614 |
| qv10 | 1.313 | 1.237 | 0.252 | 0.251 | 0.675 | 0.308 |
| surface_d11_t15 | 0.938 | 0.854 | 1.005 | 1.007 | 1.005 | 1.005 |
| surface_d7_t15 | 1.650 | 1.469 | 1.787 | 1.793 | 1.427 | 0.503 |
| surface_d7_t19 | 0.968 | 0.872 | 1.014 | 1.018 | 1.017 | 1.015 |
| surface_d9_t10 | 1.642 | 1.473 | 1.796 | 1.800 | 1.402 | 0.484 |
| surface_d9_t19 | 0.906 | 0.817 | 0.973 | 0.972 | 0.971 | 0.972 |

## Delta: 20260725T163547Z_noise-specialized vs 20260725T162524Z_noise-fenced-gated (V2/SVM ratio; negative = improvement)

| circuit | prev | cur | Δ | Δ% |
|---|---|---|---|---|
| circuit_d3_p0.001 | 2.117 | 0.524 | -1.593 | -75.2% |
| frame_h | 2.859 | 0.614 | -2.245 | -78.5% |
| qv10 | 0.675 | 0.308 | -0.367 | -54.4% |
| surface_d11_t15 | 1.005 | 1.005 | +0.000 | +0.0% |
| surface_d7_t15 | 1.427 | 0.503 | -0.924 | -64.8% |
| surface_d7_t19 | 1.017 | 1.015 | -0.002 | -0.2% |
| surface_d9_t10 | 1.402 | 0.484 | -0.918 | -65.5% |
| surface_d9_t19 | 0.971 | 0.972 | +0.001 | +0.1% |
