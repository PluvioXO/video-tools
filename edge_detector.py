#!/usr/bin/env python3
"""
Video Edge Detection Tool
Detects sharp edges in a video and draws lines between randomly selected edge points.
"""

import cv2
import numpy as np
import argparse
import os
import random


def detect_edges_and_draw(input_video_path, output_video_path, 
                          canny_threshold1=50, canny_threshold2=150,
                          line_color=(255, 255, 255), line_width=1,
                          num_graph_points=50, line_probability=0.05,
                          reset_interval=30, grain_intensity=0, grain_color=None):
    """
    Process a video to detect sharp edges and draw lines between random edge points.
    
    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the output video
        canny_threshold1 (int): First threshold for Canny edge detection
        canny_threshold2 (int): Second threshold for Canny edge detection
        line_color (tuple): BGR color for the lines (default: white)
        line_width (int): Width of the lines to draw
        num_graph_points (int): Number of random points to select for graph
        line_probability (float): Probability of drawing line between any two points
        reset_interval (int): Number of frames before resetting line selection (0 = every frame)
        grain_intensity (int): Intensity of grain effect (0-255, 0 = no grain)
        grain_color (tuple): BGR color for grain (None = use line_color)
    
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
    
    # Use line color for grain if not specified
    if grain_color is None:
        grain_color = line_color
    
    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Reset interval: {reset_interval} frames ({reset_interval/fps:.2f} seconds)" if reset_interval > 0 else "Reset interval: Every frame")
    print(f"Grain intensity: {grain_intensity}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_video_path}")
        cap.release()
        return False
    
    frame_count = 0
    selected_lines = []  # Store line coordinates (x1, y1, x2, y2)
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Check if we need to reset/recalculate lines
        should_recalculate = (reset_interval == 0) or (frame_count % reset_interval == 1)
        
        if should_recalculate:
            # Convert frame to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges using Canny edge detection
            edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
            
            # Find coordinates of edge pixels
            edge_points = np.column_stack(np.where(edges > 0))
            
            # Clear previous lines and randomly select new points for the graph
            selected_lines = []
            
            if len(edge_points) > 0:
                num_points = min(num_graph_points, len(edge_points))
                selected_indices = random.sample(range(len(edge_points)), num_points)
                selected_points = edge_points[selected_indices]
                
                # Generate lines between random pairs of selected points
                for i in range(num_points):
                    for j in range(i + 1, num_points):
                        if random.random() < line_probability:
                            y1, x1 = selected_points[i]
                            y2, x2 = selected_points[j]
                            selected_lines.append((x1, y1, x2, y2))
        
        # Apply grain effect if enabled
        if grain_intensity > 0:
            # Create a mask for areas between lines (gaps in the graph)
            line_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Draw lines on mask
            for x1, y1, x2, y2 in selected_lines:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, line_width + 2)  # Slightly thicker to account for anti-aliasing
            
            # Invert mask to get gaps (areas without lines)
            gap_mask = cv2.bitwise_not(line_mask)
            
            # Generate random noise
            noise = np.random.randint(-grain_intensity, grain_intensity, (height, width, 3), dtype=np.int16)
            
            # Apply noise only to gaps
            for c in range(3):
                # Add noise to each color channel
                channel = frame[:, :, c].astype(np.int16)
                channel_noise = noise[:, :, c]
                
                # Apply noise where gap_mask is white (255)
                mask_3d = (gap_mask > 0).astype(np.float32) / 255.0
                channel = channel + (channel_noise * mask_3d).astype(np.int16)
                
                # Color the grain with the specified color
                color_influence = 0.3  # How much the grain color affects the result
                channel = channel + (grain_color[c] - 128) * mask_3d * color_influence
                
                # Clip values to valid range
                channel = np.clip(channel, 0, 255)
                frame[:, :, c] = channel.astype(np.uint8)
        
        # Draw the stored lines on this frame
        for x1, y1, x2, y2 in selected_lines:
            cv2.line(frame, (x1, y1), (x2, y2), line_color, line_width)
        
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
        description="Detect sharp edges in a video and draw lines between randomly selected edge points.",
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
        "-w", "--line-width",
        type=int,
        default=1,
        help="Width of the lines to draw"
    )
    
    parser.add_argument(
        "-c", "--color",
        nargs=3,
        type=int,
        default=[255, 255, 255],
        metavar=("B", "G", "R"),
        help="Color of the lines in BGR format (e.g., 255 255 255 for white)"
    )
    
    parser.add_argument(
        "-n", "--num-points",
        type=int,
        default=50,
        help="Number of random edge points to select for graph connections"
    )
    
    parser.add_argument(
        "-p", "--line-probability",
        type=float,
        default=0.05,
        help="Probability (0-1) of drawing a line between any two selected points"
    )
    
    parser.add_argument(
        "-r", "--reset-interval",
        type=int,
        default=30,
        help="Number of frames before resetting line selection (0 = reset every frame, 30 = reset every second at 30fps)"
    )
    
    parser.add_argument(
        "-g", "--grain-intensity",
        type=int,
        default=0,
        help="Intensity of grain effect in gaps between lines (0-255, 0 = no grain)"
    )
    
    parser.add_argument(
        "-gc", "--grain-color",
        nargs=3,
        type=int,
        default=None,
        metavar=("B", "G", "R"),
        help="Color tint for grain in BGR format (default: uses line color)"
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
        line_color=tuple(args.color),
        line_width=args.line_width,
        num_graph_points=args.num_points,
        line_probability=args.line_probability,
        reset_interval=args.reset_interval,
        grain_intensity=args.grain_intensity,
        grain_color=tuple(args.grain_color) if args.grain_color else None
    )
    
    if not success:
        print("Failed to process video.")
        exit(1)


if __name__ == "__main__":
    main()