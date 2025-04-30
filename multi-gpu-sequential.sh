#!/bin/bash

# This script processes a video file in sequence across multiple GPUs
# without the need to modify the original code.

# Parameters (modify these as needed)
INPUT_VIDEO="$1"
INPUT_AUDIO="$2"
OUTPUT_VIDEO="$3"
NUM_GPUS="$4"  # Number of GPUs to use (this will determine segments)

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

echo "=== Multi-GPU Video Processing ==="
echo "Input video: $INPUT_VIDEO"
echo "Input audio: $INPUT_AUDIO"
echo "Output video: $OUTPUT_VIDEO"
echo "Using $NUM_GPUS GPUs - processing $NUM_GPUS segments sequentially"

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

# Calculate segment duration
segment_duration=$(awk "BEGIN {print $duration / $NUM_GPUS}")
echo "Each segment will be approximately $segment_duration seconds"

# Split the video into segments
echo "Splitting video into $NUM_GPUS segments..."
for (( i=0; i<$NUM_GPUS; i++ )); do
  # Calculate start time
  start_time=$(awk "BEGIN {print $i * $segment_duration}")
  
  # Extract video segment
  ffmpeg -y -loglevel error -ss $start_time -i "$INPUT_VIDEO" -t $segment_duration \
    -c:v libx264 -preset ultrafast "$TEMP_DIR/segment_${i}_video.mp4"
  
  # Extract corresponding audio segment
  ffmpeg -y -loglevel error -ss $start_time -i "$INPUT_AUDIO" -t $segment_duration \
    "$TEMP_DIR/segment_${i}_audio.wav"
  
  # Calculate end time 
  end_time=$(awk "BEGIN {print $start_time + $segment_duration}")
  echo "Created segment $i: $start_time to $end_time seconds"
done

# List the created files to verify they exist
echo "Verifying created segment files:"
ls -la "$TEMP_DIR/"

# Process each segment on different GPU, but one at a time to avoid conflicts
echo "Processing segments on different GPUs one at a time..."
all_successful=true

for (( i=0; i<$NUM_GPUS; i++ )); do
  gpu_id=$((i % NUM_GPUS))  # Ensure we stay within the available GPU range
  echo "Processing segment $i on GPU $gpu_id..."
  
  # Verify segment files exist
  if [ ! -f "$TEMP_DIR/segment_${i}_video.mp4" ]; then
    echo "Error: Segment video file not found: $TEMP_DIR/segment_${i}_video.mp4"
    exit 1
  fi
  
  if [ ! -f "$TEMP_DIR/segment_${i}_audio.wav" ]; then
    echo "Error: Segment audio file not found: $TEMP_DIR/segment_${i}_audio.wav"
    exit 1
  fi
  
  # Process this segment on the specified GPU
  echo "Starting process on GPU $gpu_id for segment $i..."
  
  # Set GPU context and process segment
  CUDA_VISIBLE_DEVICES=$gpu_id python -m scripts.inference \
    --unet_config_path "$UNET_CONFIG" \
    --inference_ckpt_path "$MODEL_CHECKPOINT" \
    --inference_steps $INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --video_path "$TEMP_DIR/segment_${i}_video.mp4" \
    --audio_path "$TEMP_DIR/segment_${i}_audio.wav" \
    --video_out_path "$OUTPUT_DIR/processed_segment_$i.mp4"
  
  # Check if processing was successful
  exit_code=$?
  
  if [ $exit_code -ne 0 ]; then
    echo "Error processing segment $i on GPU $gpu_id."
    all_successful=false
    break
  else
    echo "Successfully processed segment $i on GPU $gpu_id"
  fi
done

if [ "$all_successful" = false ]; then
  echo "Error: Some segments failed to process."
  exit 1
fi

# Verify output files exist
echo "Checking for processed output files:"
ls -la "$OUTPUT_DIR/"

# Create a file list for concatenation
echo "Creating file list for concatenation..."
FILE_LIST="$TEMP_DIR/file_list.txt"
rm -f "$FILE_LIST"

for (( i=0; i<$NUM_GPUS; i++ )); do
  output_file="$OUTPUT_DIR/processed_segment_$i.mp4"
  if [ -f "$output_file" ]; then
    echo "file '$output_file'" >> "$FILE_LIST"
  else
    echo "Error: Output file not found: $output_file"
    exit 1
  fi
done

# Show the file list
echo "File list contents:"
cat "$FILE_LIST"

# Concatenate all segments
echo "Concatenating processed segments..."
ffmpeg -y -f concat -safe 0 -i "$FILE_LIST" -c copy "$TEMP_DIR/concat_output.mp4"

if [ ! -f "$TEMP_DIR/concat_output.mp4" ]; then
  echo "Error: Failed to concatenate segments."
  exit 1
fi

# Add the full audio to the final video
echo "Adding original audio to final video..."
ffmpeg -y -i "$TEMP_DIR/concat_output.mp4" -i "$INPUT_AUDIO" -c:v copy -c:a aac -shortest "$OUTPUT_VIDEO"

if [ ! -f "$OUTPUT_VIDEO" ]; then
  echo "Error: Failed to create final output video."
  exit 1
fi

echo "=== Multi-GPU processing complete! ==="
echo "Final output saved to: $OUTPUT_VIDEO"
echo "Temporary files are in $TEMP_DIR and $OUTPUT_DIR"
echo "You can delete these directories when you're satisfied with the results"
