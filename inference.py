from ultralytics import YOLO
import cv2
import time
import torch
import os
from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def setup_output_paths(video_num):
    Path("./videos").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    return {
        'video': f"./videos/pred_OTV{video_num}.mp4",
        'log': f'./logs/pred_OTV{video_num}_log.txt'
    }

def process_video(model, source_path, output_paths, conf_threshold=0.75):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {source_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_paths['video'], fourcc, fps, (width, height))

    COLORS = {
        'high_conf': (0, 255, 0),  # Green
        'low_conf': (0, 0, 255)    # Red
    }

    with open(output_paths['log'], 'w') as log_file:
        log_file.write("Frame,Inference Time (s),Speed (FPS)\n")
        
        frame_count = 0
        start_time = time.time()

        try:
            for result in model(source_path, stream=True):
                frame_count += 1
                frame = result.orig_img.copy()
                
                inference_start = time.time()
                
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        # Determine color based on confidence
                        color = COLORS['high_conf'] if conf >= conf_threshold else COLORS['low_conf']
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f'{result.names[cls]}: {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Calculate and log performance metrics
                inference_time = time.time() - inference_start
                speed_fps = 1 / inference_time if inference_time > 0 else 0
                log_file.write(f"{frame_count},{inference_time:.4f},{speed_fps:.2f}\n")

                # Write frame and show preview
                out.write(frame)
                cv2.imshow('Inference', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            total_time = time.time() - start_time
            average_fps = frame_count / total_time if total_time > 0 else 0
            
            log_file.write(f"\nTotal Frames: {frame_count}\n")
            log_file.write(f"Total Processing Time: {total_time:.2f} seconds\n")
            log_file.write(f"Average Speed: {average_fps:.2f} FPS\n")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

def main():
    video_num = 3
    model_path = "./models/runs/detect/extra_large/weights/best.pt"
    source_path = f"./videos/OTV{video_num}.mp4"
    
    device = get_device()
    print(f"Using device: {device}")
    
    model = YOLO(model_path).to(device)
    output_paths = setup_output_paths(video_num)
    
    try:
        process_video(model, source_path, output_paths)
        print(f"Log saved to {output_paths['log']}")
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()