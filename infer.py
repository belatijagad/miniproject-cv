from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

def get_device():
    """Determine the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU")
    return device

def main():
    # Get the best available device
    device = get_device()
    
    # Define base directory for the project
    project_dir = Path(__file__).parent
    
    # Load the data.yaml
    data_yaml_path = project_dir / "videos/MarioHTV.v1i.yolov11/data.yaml"
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # Get test path
    test_path = str(project_dir / "videos/MarioHTV.v1i.yolov11/test/images")

    # Create a new config - using test path for all splits
    new_config = {
        'path': str(project_dir / "videos/MarioHTV.v1i.yolov11"),
        'train': test_path,  # Required by YOLO but won't be used
        'val': test_path,    # Required by YOLO but won't be used
        'test': test_path,   # This is what we'll actually use
        'nc': data_config['nc'],
        'names': data_config['names']
    }

    # Write updated yaml
    temp_yaml_path = project_dir / "temp_data.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(new_config, f)

    print("Testing path:", new_config['test'])

    # Load model with appropriate device
    model = YOLO("./models/runs/detect/nano/weights/best.pt").to(device)

    # Run validation on test set
    try:
        results = model.val(data=str(temp_yaml_path), split='test', device=device)
        
        # Print metrics
        print(f"\nTest Set Results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
    except Exception as e:
        print(f"Error during validation: {e}")
        print("\nContent of new data.yaml:")
        print(new_config)

if __name__ == "__main__":
    main()