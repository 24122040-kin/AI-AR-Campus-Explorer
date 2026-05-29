import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps=2):
    """
    Extracts frames from a video file at a specified frame rate.
    Ideal for preparing dataset for DUSt3R 3D Reconstruction.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > fps else 1
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_name = f"{os.path.basename(video_path).split('.')[0]}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1
            
        count += 1
        
    cap.release()
    print(f"Extraction complete! Saved {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for 3D Mapping")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract (default: 2)")
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.fps)

