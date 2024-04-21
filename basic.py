import cv2
import numpy as np

# Load video file
video_path = "/home/kalki/Downloads/Alan Walker - Sing Me To Sleep.mp4"  # Update with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Initialize video writer for saving processed video
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (960, 540))  # Output video resolution matches resized frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame using CPU
    resized_frame = cv2.resize(frame, (960, 540))

    # Detect humans in the frame
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    # Convert indices to layer names
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Only consider 'person' class
                # Object detected is a person
                center_x = int(detection[0] * resized_frame.shape[1])
                center_y = int(detection[1] * resized_frame.shape[0])
                w = int(detection[2] * resized_frame.shape[1])
                h = int(detection[3] * resized_frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box around detected human
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display resized frame
    cv2.imshow('Preview', resized_frame)

    # Write resized frame to output video file
    out.write(resized_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
