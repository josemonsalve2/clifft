# Benchmark run 20260725T165556Z_specializer-verify
label: specializer-verify · commit: e02eeec-dirty · branch: mlir-v2

| circuit | V2 kernel(µs) | SVM kernel(µs) | V2/SVM | V2 LDS | V2 VGPR | V2 VALU | MFMA |
|---|---|---|---|---|---|---|---|
| circuit_d3_p0.001 | 220.7 | 435.7 | 0.507 | 0 | 64 | 3.79e+06 | 0.0 |
| frame_h | 11.0 | 128.2 | 0.085 | 0 | 32 | 4.04e+04 | 0.0 |
| qv10 | 1348.3 | 4340.0 | 0.311 | 13312 | 36 | 5.02e+08 | 0.0 |
| surface_d11_t15 | 75267.0 | 74774.3 | 1.007 | 1024 | 64 | 8.74e+09 | 0.0 |
| surface_d7_t15 | 11112.8 | 21984.7 | 0.505 | 13312 | 64 | 1.15e+09 | 0.0 |
| surface_d7_t19 | 20334.2 | 19910.3 | 1.021 | 1024 | 64 | 2.34e+09 | 0.0 |
| surface_d9_t10 | 21255.4 | 44302.3 | 0.48 | 13312 | 64 | 2.30e+09 | 0.0 |
| surface_d9_t19 | 43485.4 | 44599.4 | 0.975 | 1024 | 64 | 5.04e+09 | 0.0 |
