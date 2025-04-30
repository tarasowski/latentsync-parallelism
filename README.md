
```
  ./multi-gpu-sequstion.sh assets/demo1_video.mp4 assets/demo1_audio.wav output.mp4 2
```

 Just install the bc command using:
  apt-get update && apt-get install -y bc

  Then run:
  ```
  ./overlap_multi_gpu.sh assets/demo1_video.mp4 assets/demo1_audio.wav output.mp4 2
```
  This script:
  1. Uses the original inference.py
  2. Creates overlapping video segments for better face detection
  3. Uses CUDA_VISIBLE_DEVICES to assign different GPUs to each segment
  4. Processes segments in parallel
  5. Uses the first successfully processed segment for the final output

  No other modifications are needed to your existing codebase.

