# Benchmark run index (progressive)

| run_id | label | commit | branch | circuits | mean V2/SVM |
|---|---|---|---|---|---|
| 20260725T030000Z_baseline-preopt | baseline-preopt | 788f23b* | mlir-v2 | 8 | 6.358 |
| 20260725T051154Z_after-P0-P1 | after-P0-P1 | 715f8d0* | mlir-v2 | 8 | 1.127 |
| 20260725T082430Z_after-specializer | after-specializer | 60d5728* | mlir-v2 | 8 | 1.069 |
| 20260725T083507Z_specializer-final | specializer-final | 4b55871* | mlir-v2 | 8 | 1.067 |
| 20260725T162524Z_noise-fenced-gated | noise-fenced-gated | 9d9cc68* | mlir-v2 | 8 | 1.434 |
| 20260725T163547Z_noise-specialized | noise-specialized | bbb5e42* | mlir-v2 | 8 | 0.678 |
| 20260725T165556Z_specializer-verify | specializer-verify | e02eeec* | mlir-v2 | 8 | 0.611 |
| 20260725T171826Z_global-specialized | global-specialized | 5d10409* | mlir-v2 | 8 | 0.410 |
| 20260726T000322Z_fullbench-rank26 | fullbench-rank26 | 6960527* | mlir-v2 | 21 | 0.643 |
| 20260726T011254Z_fullbench-3way | fullbench-3way | 68cc1c6* | mlir-v2 | 21 | 0.664 |
