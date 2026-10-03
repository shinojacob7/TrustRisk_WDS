# Phase 6K: Temporal Window Sweep (E1 Masked Mean)

| Window (W) | PR-AUC | ROC-AUC | F1 Score | Precision | Recall | Threshold | Event Detection |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.5321 | 0.9200 | 0.9551 | 0.9497 | 0.9605 | 0.98 | 3/4 |
| 24 | 0.2376 | 0.6676 | 0.6760 | 0.8818 | 0.5480 | 0.82 | 2/4 |
| 36 | 0.2453 | 0.7345 | 0.6783 | 0.8899 | 0.5480 | 0.98 | 2/4 |
| 48 | 0.5063 | 0.8409 | 0.8395 | 0.7456 | 0.9605 | 0.98 | 3/4 |

## Event-Level Breakdown

### Window = 12
- Event 1 (60 frames): Detected=True, Delay=28
- Event 2 (37 frames): Detected=True, Delay=17
- Event 3 (7 frames): Detected=False, Delay=-1
- Event 4 (73 frames): Detected=True, Delay=34

### Window = 24
- Event 1 (60 frames): Detected=True, Delay=52
- Event 2 (37 frames): Detected=True, Delay=20
- Event 3 (7 frames): Detected=False, Delay=-1
- Event 4 (73 frames): Detected=False, Delay=-1

### Window = 36
- Event 1 (60 frames): Detected=True, Delay=50
- Event 2 (37 frames): Detected=True, Delay=18
- Event 3 (7 frames): Detected=False, Delay=-1
- Event 4 (73 frames): Detected=False, Delay=-1

### Window = 48
- Event 1 (60 frames): Detected=True, Delay=28
- Event 2 (37 frames): Detected=True, Delay=18
- Event 3 (7 frames): Detected=False, Delay=-1
- Event 4 (73 frames): Detected=True, Delay=5
