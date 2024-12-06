from ultralytics import YOLO
import cv2
import time
import torch
import os
from pathlib import Path
import numpy as np

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def get_frame_annotations(frame_number, labels_dir, img_width, img_height):
    """Get ground truth annotations for specific frame"""
    # Format frame number to match file naming, ignoring anything after the frame number
    frame_base = f"HTV_mp4-{frame_number:04d}"
    
    # Find matching label file
    label_files = list(Path(labels_dir).glob(f"{frame_base}*.txt"))
    
    boxes = []
    if label_files:  # If any matching files found
        label_path = label_files[0]  # Take the first matching file
        with open(label_path, 'r') as f:
            for line in f:
                # YOLO format: class x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                boxes.append([x1, y1, x2, y2])
    
    return boxes

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
        'video': f"./videos/pred_HTV{video_num}_nano.mp4",
        'log': f'./logs/pred_HTV{video_num}_log_nano.txt'
    }

def process_video(model, source_path, labels_dir, output_paths, conf_threshold=0.6, iou_threshold=0.5):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_paths['video'], fourcc, fps, (width, height))

    COLORS = {
        'match': (0, 255, 0),     # Green (IoU > threshold & in GT frame)
        'no_match': (0, 0, 255),  # Red (IoU < threshold but in GT frame)
        'no_gt': (255, 0, 0)      # Blue (No GT for this frame)
    }

    with open(output_paths['log'], 'w') as log_file:
        log_file.write("Frame,Inference Time (s),Speed (FPS),Matches,No Matches\n")
        
        frame_count = 0
        start_time = time.time()

        try:
            for result in model(source_path, stream=True):
                frame_count += 1
                frame = result.orig_img.copy()
                
                inference_start = time.time()
                matches = 0
                no_matches = 0
                
                # Get ground truth annotations for current frame
                gt_boxes = get_frame_annotations(frame_count-1, labels_dir, width, height)
                
                if len(result.boxes) > 0:
                    has_gt_boxes = len(gt_boxes) > 0
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        if conf < conf_threshold:
                            continue
                            
                        color = COLORS['no_gt']
                        
                        if has_gt_boxes:
                            max_iou = 0
                            for gt_box in gt_boxes:
                                iou = calculate_iou([x1, y1, x2, y2], gt_box)
                                max_iou = max(max_iou, iou)
                            
                            if max_iou > iou_threshold:
                                color = COLORS['match']
                                matches += 1
                            else:
                                color = COLORS['no_match']
                                no_matches += 1
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f'{result.names[cls]}: {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                inference_time = time.time() - inference_start
                speed_fps = 1 / inference_time if inference_time > 0 else 0
                log_file.write(f"{frame_count},{inference_time:.4f},{speed_fps:.2f},{matches},{no_matches}\n")

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
    model_path = "./models/runs/detect/nano/weights/best.pt"
    source_path = f"./videos/HTV.mp4"
    labels_dir = "./videos/MarioHTV.v1i.yolov11/test/labels"
    
    device = get_device()
    print(f"Using device: {device}")
    
    model = YOLO(model_path).to(device)
    output_paths = setup_output_paths(video_num)
    
    try:
        process_video(model, source_path, labels_dir, output_paths)
        print(f"Log saved to {output_paths['log']}")
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()