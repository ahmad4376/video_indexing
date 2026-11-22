# Video indexing with YOLO11l

`run.py` reads every video in a folder of FineVideo Parquet files and writes one JSON line per second of video. Each line lists which of five objects appear (person, car, dog, laptop, cell phone) and how many people are visible.

## Install

Tested on Python 3.11 with CUDA 12.

```bash
# system libraries that OpenCV (used by Ultralytics) needs on a minimal Ubuntu
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls the CUDA 12.6 build of PyTorch from the PyTorch package index. PyAV ships its own FFmpeg, so no system FFmpeg is needed.

## Preparation command (run once, needs internet)

Run this from the repository folder. It downloads `yolo11l.pt` next to `run.py`, which is where `run.py` looks for it.

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

No internet access is needed while it runs.

Optional flags:

| Flag | Default | Meaning |
|---|---|---|
| `--stride` | `5` | Send every Nth decoded frame to the model |
| `--batch` | `32` | Frames per model call |
| `--resume` | off | Skip Parquet files whose part file already exists (for restarting after a crash) |

`run.py` writes each Parquet file's results to `<output>.parts/<file>.jsonl` first, then joins them into `--output` at the end. Per-file timings go to stderr and to `<output>.parts/timing.jsonl`.

## How it works

- Every frame is decoded with PyAV. Seconds are counted from the first decoded frame using integer timestamp arithmetic.
- Every 5th frame is rotated if the MP4 has a rotation matrix, converted to BGR and sent to YOLO11l in batches of 32. Settings: `imgsz=640`, `conf=0.4`, `iou=0.7`, the 5 classes only, FP16 on GPU.
- Decoding runs in a separate thread, so it overlaps with GPU inference.
- For each second, an object is present if it was found in at least half of the sampled frames. The people count is the median over the sampled frames, rounded down at .5.
- If a video fails, the lines already written for it are kept and one error line is added. The script then moves on to the next video.

## Known limits

- Rotation was tested with display-matrix rotation (90/180/270) against FFmpeg's autorotate. The older `rotate` metadata tag was not tested.
- Videos under about 9 fps get a smaller stride (`round(fps / 2)`), so each second still gets at least about 2 sampled frames. That path was not tested, because all test videos were 24–30 fps.
- The error path was tested only on deliberately broken videos (garbage bytes, a truncated file).
- No test video contained a dog, so dog detection accuracy is unverified.

## Results in this repo

`results.jsonl.gz` covers the first 50 FineVideo Parquet files: 1,650 videos and 461,757 seconds. It was produced on Kaggle (see `HARDWARE.md`) with Python 3.13, torch 2.11.0+cu130 and av 19.0.2. `av 19` does not support Python 3.11, so `requirements.txt` pins `av==18.1.1` and the CUDA 12.6 build of the same torch version. A fresh install from `requirements.txt` on Python 3.11 reproduced 99.93% of the seconds of file `train-00000` exactly (6 of 9,144 differed).
