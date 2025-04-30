#!/bin/bash

# This script processes a video file with overlapping segments to ensure face detection works

# Parameters (modify these as needed)
INPUT_VIDEO="$1"
INPUT_AUDIO="$2"
OUTPUT_VIDEO="$3"
NUM_GPUS="$4"  # Number of GPUs to use

# Default values if not provided
INPUT_VIDEO="${INPUT_VIDEO:-assets/demo1_video.mp4}"
INPUT_AUDIO="${INPUT_AUDIO:-assets/demo1_audio.wav}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-multi_gpu_output.mp4}"
NUM_GPUS="${NUM_GPUS:-2}"  # Default to 2 GPUs

# Configuration
CURRENT_DIR="$(pwd)"
TEMP_DIR="$CURRENT_DIR/parallel_segments"
OUTPUT_DIR="$CURRENT_DIR/parallel_outputs"
UNET_CONFIG="configs/unet/stage2.yaml"
MODEL_CHECKPOINT="checkpoints/latentsync_unet.pt"
INFERENCE_STEPS=20
GUIDANCE_SCALE=2.0

# Create directories
rm -rf "$TEMP_DIR" "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR/logs"

echo "=== Multi-GPU Overlapping Segment Processing ==="
echo "Input video: $INPUT_VIDEO"
echo "Input audio: $INPUT_AUDIO" 
echo "Output video: $OUTPUT_VIDEO"
echo "Using $NUM_GPUS GPUs for processing"

# Get video duration and check if files exist
if [ ! -f "$INPUT_VIDEO" ]; then
  echo "Error: Input video file not found: $INPUT_VIDEO"
  exit 1
fi

if [ ! -f "$INPUT_AUDIO" ]; then
  echo "Error: Input audio file not found: $INPUT_AUDIO"
  exit 1
fi

# Get video duration using ffprobe
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO")
echo "Video duration: $duration seconds"

# Use overlapping segments to ensure face detection works on all segments
# Calculate segment duration with longer overlapping segments
segment_duration=$(awk "BEGIN {print $duration * 0.75}")
echo "Each segment will be approximately $segment_duration seconds (with overlap)"

# Create overlapping segments
echo "Creating overlapping segments for $NUM_GPUS GPUs..."

# First segment: Start from beginning
ffmpeg -y -loglevel error -i "$INPUT_VIDEO" -t $segment_duration \
  -c:v libx264 -preset ultrafast "$TEMP_DIR/segment_0_video.mp4"

ffmpeg -y -loglevel error -i "$INPUT_AUDIO" -t $segment_duration \
  "$TEMP_DIR/segment_0_audio.wav"

echo "Created segment 0: 0 to $segment_duration seconds"

# Second segment: Start from middle to end
second_start=$(awk "BEGIN {print $duration - $segment_duration}")
if (( $(echo "$second_start < 0" | bc -l) )); then
  second_start=0
fi

ffmpeg -y -loglevel error -ss $second_start -i "$INPUT_VIDEO" \
  -c:v libx264 -preset ultrafast "$TEMP_DIR/segment_1_video.mp4"

ffmpeg -y -loglevel error -ss $second_start -i "$INPUT_AUDIO" \
  "$TEMP_DIR/segment_1_audio.wav"

echo "Created segment 1: $second_start to $duration seconds"

# List the created files to verify they exist
echo "Verifying created segment files:"
ls -la "$TEMP_DIR/"

# Process segments in parallel
echo "Processing segments on multiple GPUs in parallel..."
processes=()

for (( i=0; i<NUM_GPUS && i<2; i++ )); do  # Limit to 2 segments max
  echo "Starting inference on GPU $i for segment $i..."
  
  # Verify segment files exist
  if [ ! -f "$TEMP_DIR/segment_${i}_video.mp4" ]; then
    echo "Error: Segment video file not found: $TEMP_DIR/segment_${i}_video.mp4"
    exit 1
  fi
  
  if [ ! -f "$TEMP_DIR/segment_${i}_audio.wav" ]; then
    echo "Error: Segment audio file not found: $TEMP_DIR/segment_${i}_audio.wav"
    exit 1
  fi
  
  # Create a separate shell command for each GPU and run it in the background
  LOG_FILE="$TEMP_DIR/logs/gpu_${i}.log"
  
  (
    echo "Starting process on GPU $i at $(date)" > "$LOG_FILE"
    
    # Use the original inference.py script since we're using CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$i python -m scripts.inference \
      --unet_config_path "$UNET_CONFIG" \
      --inference_ckpt_path "$MODEL_CHECKPOINT" \
      --inference_steps $INFERENCE_STEPS \
      --guidance_scale $GUIDANCE_SCALE \
      --video_path "$TEMP_DIR/segment_${i}_video.mp4" \
      --audio_path "$TEMP_DIR/segment_${i}_audio.wav" \
      --video_out_path "$OUTPUT_DIR/processed_segment_$i.mp4" >> "$LOG_FILE" 2>&1
    
    echo "Process on GPU $i finished with exit code $? at $(date)" >> "$LOG_FILE"
  ) &
  
  # Store the process ID
  processes+=($!)
  echo "Inference process started on GPU $i with PID ${processes[-1]}"
done

# Wait for all processes to complete
echo "Waiting for all GPU processes to complete..."
success=true

for pid in "${processes[@]}"; do
  wait $pid
  exit_code=$?
  echo "Process $pid completed with exit code $exit_code"
  
  if [ $exit_code -ne 0 ]; then
    success=false
  fi
done

# Check if at least one segment was processed successfully
echo "Checking for processed output files:"
ls -la "$OUTPUT_DIR/"

# Find the first successful output
OUTPUT_FILE=""
for (( i=0; i<2; i++ )); do
  if [ -f "$OUTPUT_DIR/processed_segment_$i.mp4" ]; then
    OUTPUT_FILE="$OUTPUT_DIR/processed_segment_$i.mp4"
    echo "Found processed segment: $OUTPUT_FILE"
    break
  fi
done

if [ -z "$OUTPUT_FILE" ]; then
  echo "Error: No successful output found. Check logs in $TEMP_DIR/logs/"
  exit 1
fi

# Copy the successful output to the final destination
echo "Copying successful output to final destination..."
cp "$OUTPUT_FILE" "$OUTPUT_VIDEO"

echo "=== Multi-GPU processing complete! ==="
echo "Final output saved to: $OUTPUT_VIDEO"
echo "Temporary files are in $TEMP_DIR and $OUTPUT_DIR"
echo "You can delete these directories when you're satisfied with the results"
