import cv2
import telepot
import numpy as np
import time

# Initialize the Telegram bot
bot = telepot.Bot('YOUR_TELEGRAM_BOT_TOKEN')

# Load YOLO model
net = cv2.dnn.readNet("yolo.weights", "yolo.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to send a notification with a video clip to Telegram
def send_notification(video_path):
    bot.sendVideo('TELEGRAM_CHAT_ID', video=open(video_path, 'rb'))

# Function to detect persons in a frame
def detect_persons(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # If the object detected is a person and confidence is high enough
            if class_id == 0 and confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw bounding box around the person
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)

                # Write 'Person' text
                cv2.putText(frame, 'Person', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Main function to detect motion, perform person detection, and notify on Telegram
def main():
    # Initialize video capture object
    cap = cv2.VideoCapture("YOUR_RTSP_STREAM_URL")

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Threshold the foreground mask
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours (motion) are detected
        if contours:
            # Perform person detection
            detect_persons(frame)

            # Save the video clip for notification
            timestamp = int(time.time())
            video_path = f'video_{timestamp}.avi'
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame.shape[1], frame.shape[0]))
            out.write(frame)

            # Send notification with the video clip
            send_notification(video_path)

            # Release the video writer
            out.release()

            # Sleep to avoid continuous notifications for the same event
            time.sleep(30)

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
