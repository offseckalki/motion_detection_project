import cv2
import telepot
import numpy as np
import time

# Initialize the Telegram bot (Replace '' with your bot token)
bot = telepot.Bot('7186251726:AAFjsNe7pVz-r_GbndNqoMIpCJ06fzow6mA')

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Convert indices to layer names
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Function to send a notification with a video clip to Telegram
def send_notification(video_path):
    print("Sending notification...")
    try:
        print("Sending video:", video_path)
        with open(video_path, 'rb') as video_file:
            # Replace 'YOUR_TELEGRAM_CHAT_ID' with your Telegram chat ID 
            bot.sendVideo('891381553', video=video_file)
        print("Notification sent successfully")
    except Exception as e:
        print("Error sending notification:", e)

# Define the RTSP URL with authentication and port
username = 'admin'
password = 'cameraman123'
ip_address = '192.168.1.6'
port = '10554'
rtsp_url = f'rtsp://{username}:{password}@{ip_address}:{port}/Streaming/channels/101'

# Initialize video capture object
cap = cv2.VideoCapture(rtsp_url)

# Main function to detect human motion and notify on Telegram
def main():
    motion_detected = False
    max_retries = 3  # Maximum number of retries for decoding frames
    retries = 0

    while True:
        ret = False
        frame = None

        # Retry decoding frame if failed
        while not ret and retries < max_retries:
            ret, frame = cap.read()
            retries += 1
            if not ret:
                print("Error decoding frame. Retrying...")

        if not ret:
            print("Failed to decode frame after retries. Exiting...")
            break

        retries = 0  # Reset retries after successful frame decoding

        # Detecting objects
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Only consider 'person' class
                    # Object detected is a person
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            motion_detected = True

            # Initialize video writer object if not already initialized
            if not motion_detected:
                out = cv2.VideoWriter('motion_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))

            # Write frames to the video file
            out.write(frame)

        # If motion detected and no longer detected, send notification with the video clip
        if motion_detected and len(indexes) == 0:
            # Release the video writer
            out.release()

            # Send notification with the video clip
            send_notification('motion_detected.mp4')

            # Reset motion detection flag
            motion_detected = False

    # Release the camera and video capture objects
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
