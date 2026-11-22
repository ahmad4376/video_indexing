# Write-up
 
 
## 1. Your timings
 
My first version ran YOLO on every frame in FP32. I never ran it on all 50 files: I measured about 35 frames per second on a T4, and with roughly 12 million frames that's around 95 hours. My final version finished all 50 files in 3 h 22 min on Kaggle, with two T4s running side by side on 25 files each (6 h 31 min of GPU time in total). The change that saved the most was only sending every 5th frame to the model, which is about 5x less GPU work. FP16 was next, at about 2.2x.
 
## 2. Your pipeline
 
I open each Parquet file in the input folder with pyarrow and read one row at a time; the row number is the video id. For each video I read the rotation from the MP4's display matrix, then start a decode thread. That thread decodes every frame with PyAV and works out each frame's second from its timestamp minus the first frame's, using integer maths. Every 5th frame gets converted to BGR, rotated if needed, and pushed onto a queue in batches of 32. The main thread runs YOLO11l on each batch (640, conf 0.4, IoU 0.7, only the 5 classes, FP16) and stores the people count and objects for each frame. Once the video is done, an object counts for a second if it shows up in at least half of that second's frames, and people is the median, rounded down. Then I write one line per second.
 
## 3. Where the time goes
 
The GPU inference step. I added timers with time.perf_counter around each stage in run.py and logged them for every file. In the full run, GPU 0 spent 11,929 s out of 12,123 s on inference and only 98 s waiting for frames. Decoding added up to 5,932 s, but it runs in its own thread alongside the GPU, and that thread was often stuck waiting because the queue was full. So the GPU was the slow part, not decoding. I also checked Ultralytics' own per-image timings on a T4: about 1 ms preprocessing, 6.4 ms inference and 0.3 to 1.6 ms postprocessing. On a 4090 I think this flips: inference gets much faster, and single-thread decoding plus CPU preprocessing probably become the bottleneck.
 
## 4. Checking correctness
 
I built my own reference the same way you do: I ran the model on every frame in FP32 and saved the result for each frame. Then I scored every change against it with the three thresholds. I picked 16 videos with lots of people (1,621 seconds), since crowds are the hardest case for skipping frames. At stride 5 the real run.py got F1 0.989, exact 0.924 and within-1 0.998. I also scored my final results on 9 random videos from a file I never tuned on: F1 0.985, exact 0.976. A clean install reproduced 99.93% of file 0.
 
## 5. Working with AI
 
When I tested rotation, the AI gave me an ffmpeg command to make rotated copies of a video so I could compare my frames with ffmpeg's autorotate. Every copy came back with rotation 0 and the same shape, so the test "passed" but didn't actually test anything. I caught it because the tagged files should have shown a non-zero rotation. It turned out the command used the old -metadata rotate= syntax, which ffmpeg 6.1 just ignores. With -display_rotation the tags worked and my frames matched ffmpeg's exactly. Since then I always make sure a test can actually fail.
 
## 6. Next steps
 
I'd focus on the CPU side. On a 4090 inference should be several times faster, but decoding is still one thread per video and preprocessing still happens in the main thread, so the GPU would end up waiting. I'd decode several videos at the same time and move the letterbox into the worker threads, then measure it on a real 4090. I'd also try adaptive sampling: look at a few frames per second and only check more when they disagree, since crowded videos were my weakest (one got 0.705 exact). Last, I'd test real damaged files.
 
