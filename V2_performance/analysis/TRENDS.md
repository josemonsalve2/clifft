# Progressive V2/SVM kernel-time ratio (lower = better; <1 = V2 wins)

| circuit | 20260725T030000Z_baseline-preopt | 20260725T051154Z_after-P0-P1 | 20260725T082430Z_after-specializer | 20260725T083507Z_specializer-final |
|---|---|---|---|---|
| circuit_d3_p0.001 | 15.193 | 1.081 | 1.116 | 1.103 |
| frame_h | 28.258 | 1.215 | 0.612 | 0.594 |
| qv10 | 1.313 | 1.237 | 0.252 | 0.251 |
| surface_d11_t15 | 0.938 | 0.854 | 1.005 | 1.007 |
| surface_d7_t15 | 1.650 | 1.469 | 1.787 | 1.793 |
| surface_d7_t19 | 0.968 | 0.872 | 1.014 | 1.018 |
| surface_d9_t10 | 1.642 | 1.473 | 1.796 | 1.800 |
| surface_d9_t19 | 0.906 | 0.817 | 0.973 | 0.972 |

## Delta: 20260725T083507Z_specializer-final vs 20260725T082430Z_after-specializer (V2/SVM ratio; negative = improvement)

| circuit | prev | cur | Δ | Δ% |
|---|---|---|---|---|
| circuit_d3_p0.001 | 1.116 | 1.103 | -0.013 | -1.2% |
| frame_h | 0.612 | 0.594 | -0.018 | -2.9% |
| qv10 | 0.252 | 0.251 | -0.001 | -0.4% |
| surface_d11_t15 | 1.005 | 1.007 | +0.002 | +0.2% |
| surface_d7_t15 | 1.787 | 1.793 | +0.006 | +0.3% |
| surface_d7_t19 | 1.014 | 1.018 | +0.004 | +0.4% |
| surface_d9_t10 | 1.796 | 1.800 | +0.004 | +0.2% |
| surface_d9_t19 | 0.973 | 0.972 | -0.001 | -0.1% |
