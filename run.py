"""Per-second object index for FineVideo parquet files."""
import argparse, io, json, math, queue, shutil, struct, sys, threading, time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from ultralytics import YOLO
import logging
from ultralytics.utils import LOGGER


class _DropHalf(logging.Filter):
    def filter(self, record):
        return "'half' is deprecated" not in record.getMessage()


LOGGER.addFilter(_DropHalf())

NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
ORDER = [0, 2, 16, 63, 67]
TIMERS = ("decode_s", "convert_s", "blocked_s", "wait_s", "infer_s")


# ---------- rotation: read the tkhd display matrix from the mp4 bytes ----------
def _boxes(buf, start, end):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, pos)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, pos + 8)[0]; hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield typ, pos + hdr, pos + size
        pos += size


def get_rotation(buf):
    """k for np.rot90 (CCW quarter turns); 0 if the matrix can't be read."""
    try:
        for t, s, e in _boxes(buf, 0, len(buf)):
            if t != b"moov":
                continue
            for t2, s2, e2 in _boxes(buf, s, e):
                if t2 != b"trak":
                    continue
                tkhd, is_video = None, False
                for t3, s3, e3 in _boxes(buf, s2, e2):
                    if t3 == b"tkhd":
                        tkhd = s3
                    elif t3 == b"mdia":
                        for t4, s4, e4 in _boxes(buf, s3, e3):
                            if t4 == b"hdlr":
                                is_video = buf[s4 + 8:s4 + 12] == b"vide"
                if is_video and tkhd is not None:
                    off = tkhd + (40 if buf[tkhd] == 0 else 52)
                    m = struct.unpack_from(">9i", buf, off)
                    ccw = -math.degrees(math.atan2(m[1] / 65536, m[0] / 65536))
                    return int(round(ccw / 90.0)) % 4
    except Exception:
        pass
    return 0


def rotate_frame(img, k):
    return img if k == 0 else np.ascontiguousarray(np.rot90(img, k))


# ---------- decoding thread ----------
def decode_worker(mp4, stride, batch, q, st, stop):
    def put(x):
        t = time.perf_counter()
        while not stop.is_set():
            try:
                q.put(x, timeout=0.5)
                break
            except queue.Full:
                pass
        st["blocked_s"] += time.perf_counter() - t

    try:
        k = get_rotation(mp4)
        c = av.open(io.BytesIO(mp4))
        s = c.streams.video[0]
        tb = s.time_base
        fps = float(s.average_rate) if s.average_rate else 25.0
        # low-fps videos: keep at least ~2 sampled frames per second
        stride_eff = max(1, min(stride, round(fps / 2)))
        pts0, buf, bsec = None, [], []
        it = iter(c.decode(s))
        idx = -1
        while not stop.is_set():
            t = time.perf_counter()
            try:
                fr = next(it)
            except StopIteration:
                st["decode_s"] += time.perf_counter() - t
                break
            st["decode_s"] += time.perf_counter() - t
            idx += 1
            if fr.pts is None:
                sec = int(idx / fps)
            else:
                if pts0 is None:
                    pts0 = fr.pts
                sec = max(0, ((fr.pts - pts0) * tb.numerator) // tb.denominator)
            st["last_sec"] = max(st["last_sec"], sec)
            if idx % stride_eff == 0:           # skipped frames are never converted
                t = time.perf_counter()
                buf.append(rotate_frame(fr.to_ndarray(format="bgr24"), k))
                st["convert_s"] += time.perf_counter() - t
                bsec.append(sec)
                if len(buf) == batch:
                    put((buf, bsec)); buf, bsec = [], []
        if buf:
            put((buf, bsec))
        c.close()
    except Exception as e:
        st["err"] = f"decode failed: {type(e).__name__}: {e}"[:120]
    finally:
        put(None)


# ---------- per-second aggregation (section 5 rules) ----------
def aggregate(secs, people, flags, n_secs):
    secs = np.asarray(secs, dtype=np.int64)
    people = np.asarray(people, dtype=np.int64)
    flags = np.asarray(flags, dtype=np.int64).reshape(-1, len(ORDER))
    keep = secs < n_secs
    secs, people, flags = secs[keep], people[keep], flags[keep]
    order = np.argsort(secs, kind="stable")
    secs, people, flags = secs[order], people[order], flags[order]
    n = np.bincount(secs, minlength=n_secs)
    hits = np.stack([np.bincount(secs, weights=flags[:, j], minlength=n_secs)
                     for j in range(len(ORDER))], axis=1)
    bounds = np.concatenate([[0], np.cumsum(n)])
    out = []
    for s in range(n_secs):
        cnt = int(n[s])
        if cnt == 0:
            out.append(([], 0)); continue
        p = np.sort(people[bounds[s]:bounds[s + 1]])
        med = int((p[(cnt - 1) // 2] + p[cnt // 2]) // 2)      # median, halves floored
        objs = [NAMES[k] for j, k in enumerate(ORDER) if hits[s, j] * 2 >= cnt]
        out.append((objs, med))
    return out


def process_video(model, mp4, args, half):
    q = queue.Queue(maxsize=4)
    stop = threading.Event()
    st = {"last_sec": -1, "err": None, "decode_s": 0.0, "convert_s": 0.0, "blocked_s": 0.0}
    th = threading.Thread(target=decode_worker,
                          args=(mp4, args.stride, args.batch, q, st, stop), daemon=True)
    th.start()
    secs, people, flags, err = [], [], [], None
    wait_s = infer_s = 0.0
    try:
        while True:
            t = time.perf_counter()
            item = q.get()
            wait_s += time.perf_counter() - t
            if item is None:
                break
            frames, fsecs = item
            t = time.perf_counter()
            res = model.predict(frames, imgsz=640, conf=0.4, iou=0.7,
                                classes=ORDER, half=half, verbose=False)
            for r, sec in zip(res, fsecs):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                secs.append(sec)
                people.append(int((cls == 0).sum()))
                flags.append([int((cls == j).any()) for j in ORDER])
            infer_s += time.perf_counter() - t
    except Exception as e:
        err = f"inference failed: {type(e).__name__}: {e}"[:120]
    finally:
        stop.set()
        th.join(timeout=10)
    tm = {"decode_s": st["decode_s"], "convert_s": st["convert_s"],
          "blocked_s": st["blocked_s"], "wait_s": wait_s, "infer_s": infer_s}
    err = err or st["err"]
    last = st["last_sec"]
    if last < 0:
        return [], err or "no frames decoded", tm
    if err is None:
        n_secs = last + 1
    else:                                  # failure: drop the possibly partial last second
        n_secs = max(secs) if secs else 0
    return aggregate(secs, people, flags, n_secs), err, tm


def process_file(model, path, args, half, part_dir):
    part = part_dir / (path.name + ".jsonl")
    if args.resume and part.exists():
        print(f"skip {path.name} (already done)", file=sys.stderr); return
    tmp = part.with_suffix(".tmp")
    t0 = time.perf_counter()
    tot = {k: 0.0 for k in TIMERS}
    nvid = nsec = 0
    with open(tmp, "w") as f:
        def err_line(row, msg):
            f.write(json.dumps({"video": f"{path.name}:{row}", "error": msg}) + "\n")

        it, n_rows = None, 0
        try:
            pf = pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
            it = pf.iter_batches(batch_size=1, columns=["mp4"])
        except Exception as e:
            err_line(0, f"parquet open failed: {type(e).__name__}")

        row = 0                                    # index of the row about to be read
        while it is not None:
            try:
                b = next(it)
            except StopIteration:
                break
            except Exception as e:                 # this row (and the rest) can't be read
                for r in range(row, max(n_rows, row + 1)):
                    err_line(r, f"parquet read failed: {type(e).__name__}")
                break
            try:
                mp4 = b.column(0)[0].as_py()
                if not mp4:
                    lines, err, tm = [], "empty video", None
                else:
                    lines, err, tm = process_video(model, mp4, args, half)
            except Exception as e:
                lines, err, tm = [], f"failed: {type(e).__name__}", None
            vid = f"{path.name}:{row}"
            for sec, (objs, ppl) in enumerate(lines):
                f.write(json.dumps({"video": vid, "second": sec,
                                    "objects": objs, "people": ppl}) + "\n")
            if err:
                err_line(row, err)
            if tm:
                for k in TIMERS:
                    tot[k] += tm[k]
            nvid += 1; nsec += len(lines)
            row += 1
    tmp.replace(part)                              # a part file only exists once complete
    dt = time.perf_counter() - t0
    other = dt - tot["wait_s"] - tot["infer_s"]
    print(f"{path.name}: {nvid} videos, {nsec} s of video, wall {dt:.0f}s | "
          f"main thread: GPU inference {tot['infer_s']:.0f}s, GPU waiting for frames "
          f"{tot['wait_s']:.0f}s, other {other:.0f}s | decode thread: decode "
          f"{tot['decode_s']:.0f}s, convert {tot['convert_s']:.0f}s, blocked on full queue "
          f"{tot['blocked_s']:.0f}s", file=sys.stderr)
    with open(part_dir / "timing.jsonl", "a") as tf:
        tf.write(json.dumps({"file": path.name, "videos": nvid, "video_seconds": nsec,
                             "wall_s": round(dt, 1), "other_s": round(other, 1),
                             **{k: round(v, 1) for k, v in tot.items()}}) + "\n")


def find_weights():
    for d in (Path(__file__).resolve().parent, Path.cwd()):
        if (d / "yolo11l.pt").exists():
            return str(d / "yolo11l.pt")
    return "yolo11l.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                    help="skip parquet files whose part file already exists")
    args = ap.parse_args()

    files = sorted(Path(args.input).rglob("*.parquet"))
    if not files:
        sys.exit(f"no .parquet files under {args.input}")
    part_dir = Path(str(args.output) + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(find_weights())
    half = torch.cuda.is_available()
    t0 = time.time()
    for p in files:
        process_file(model, p, args, half, part_dir)
    with open(args.output, "w") as out:
        for p in files:
            with open(part_dir / (p.name + ".jsonl")) as f:
                shutil.copyfileobj(f, out)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
