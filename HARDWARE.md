# Hardware

The full run of all 50 Parquet files was done in a **Kaggle notebook**.

| | |
|---|---|
| GPU | 2× NVIDIA Tesla T4, 15 GB (15,360 MiB) each, driver 580.178.04 |
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPUs (2 cores × 2 threads) |
| RAM | 31 GB |
| Disk | Kaggle cloud persistent disk, reported as non-rotational (SSD class). The exact disk type isn't exposed |
| OS | Ubuntu 24.04.4 LTS (Kaggle container) |
| Software | Python 3.13, torch 2.11.0+cu130, ultralytics 8.4.216, av 19.0.2 |

## How the run was split

Two `run.py` processes ran at the same time, one per GPU:

- GPU 0: `train-00000` to `train-00024`
- GPU 1: `train-00025` to `train-00049`

Both shared the same 4 vCPUs. No CPU thread limits were set. `run.py` is the same file in both processes and the one in this repo.

## How long it took

| | Time |
|---|---|
| GPU 0 (25 files) | 12,123 s (3 h 22 min) |
| GPU 1 (25 files) | 11,342 s (3 h 9 min) |
| **Full run, start to finish** | **about 3 h 22 min** (the two processes ran in parallel, so the longer one sets the total) |
| Total GPU time | 23,465 s (6 h 31 min) |

Times come from each process's own timer (`done in ...` in its log), which starts after the model has loaded. The dataset download was done beforehand and isn't included.

461,757 seconds (128.3 hours) of video were processed. That's about 19× real time per T4, or about 39× with both together.

The hardware details above were read in a new Kaggle session of the same machine type, because the session that did the run had ended.
