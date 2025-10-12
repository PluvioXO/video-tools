#!/usr/bin/env python3
"""
Video Edge Detection Tool
Detects sharp edges in a video and draws dots on them.
"""

import cv2
import numpy as np
import argparse
import os


def detect_edges_and_draw(input_video_path, output_video_path, 
                          canny_threshold1=50, canny_threshold2=150,
                          dot_color=(0, 255, 0), dot_radius=2):
    """
    Process a video to detect sharp edges and draw dots on them.
    
    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the output video
        canny_threshold1 (int): First threshold for Canny edge detection
        canny_threshold2 (int): Second threshold for Canny edge detection
        dot_color (tuple): BGR color for the dots (default: green)
        dot_radius (int): Radius of the dots to draw
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_video_path}")
        cap.release()
        return False
    
    frame_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detection
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
        
        # Find coordinates of edge pixels
        edge_points = np.column_stack(np.where(edges > 0))
        
        # Draw dots on edge points
        # Note: edge_points are in (row, col) format, need to convert to (x, y)
        for point in edge_points:
            y, x = point
            cv2.circle(frame, (x, y), dot_radius, dot_color, -1)
        
        # Write the frame to output video
        out.write(frame)
        
        # Print progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\nProcessing complete!")
    print(f"Output saved to: {output_video_path}")
    
    return True


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect sharp edges in a video and draw dots on them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_video",
        help="Path to the input video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to the output video file (default: input_edges.mp4)",
        default=None
    )
    
    parser.add_argument(
        "-t1", "--threshold1",
        type=int,
        default=50,
        help="First threshold for Canny edge detection (lower threshold)"
    )
    
    parser.add_argument(
        "-t2", "--threshold2",
        type=int,
        default=150,
        help="Second threshold for Canny edge detection (upper threshold)"
    )
    
    parser.add_argument(
        "-r", "--radius",
        type=int,
        default=2,
        help="Radius of the dots to draw on edges"
    )
    
    parser.add_argument(
        "-c", "--color",
        nargs=3,
        type=int,
        default=[0, 255, 0],
        metavar=("B", "G", "R"),
        help="Color of the dots in BGR format (e.g., 0 255 0 for green)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input file does not exist: {args.input_video}")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input_video)[0]
        args.output = f"{base_name}_edges.mp4"
    
    # Process the video
    success = detect_edges_and_draw(
        args.input_video,
        args.output,
        canny_threshold1=args.threshold1,
        canny_threshold2=args.threshold2,
        dot_color=tuple(args.color),
        dot_radius=args.radius
    )
    
    if not success:
        print("Failed to process video.")
        exit(1)


if __name__ == "__main__":
    main()
