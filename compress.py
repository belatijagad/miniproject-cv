import cv2

# Define the input and output video paths
input_video_path = './videos/pred_OTV5.mp4'
output_video_path = './videos/OTV5_compressed.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
# 'mp4v' codec is used for MP4, and 'XVID' can be used for AVI files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Reduce frame size by 50% for compression
compressed_width = width // 8
compressed_height = height // 8

# Create VideoWriter object with reduced resolution
out = cv2.VideoWriter(output_video_path, fourcc, fps, (compressed_width, compressed_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to compress the video
    compressed_frame = cv2.resize(frame, (compressed_width, compressed_height))

    # Write the frame to the output video
    out.write(compressed_frame)

# Release resources
cap.release()
out.release()

print(f"Video saved to {output_video_path}")
