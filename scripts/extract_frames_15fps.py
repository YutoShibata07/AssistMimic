#!/usr/bin/env python
"""
Extract frames from video at 15 FPS with high quality (lossless PNG)

Usage:
    python extract_frames_15fps.py <video_path>

Example:
    python extract_frames_15fps.py /path/to/video.mp4

Output:
    Frames will be saved to a directory named: <video_name>_frames/
    in the same directory as the video file.
"""

import cv2
import os
import sys

def extract_frames_15fps(video_path, target_fps=15):
    """
    Extract frames from video at specified FPS

    Args:
        video_path: Path to the input video file
        target_fps: Target frames per second (default: 15)
    """
    # Generate output directory name
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, f"{video_name}_frames")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Original FPS: {original_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/original_fps:.2f} seconds")

    # Calculate frame interval
    frame_interval = int(original_fps / target_fps)
    print(f"Frame interval: {frame_interval} (extracting every {frame_interval}th frame)")
    print(f"Output directory: {output_dir}")
    print()

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at target FPS intervals
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
            # Use maximum quality for PNG (lossless compression level 0)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"\nDone! Extracted {saved_count} frames at {target_fps} FPS")
    print(f"Output directory: {output_dir}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_frames_15fps.py <video_path>")
        print("Example: python extract_frames_15fps.py /path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    success = extract_frames_15fps(video_path)
    sys.exit(0 if success else 1)
