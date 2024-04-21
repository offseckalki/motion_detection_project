import cv2
import telepot
import numpy as np
import time
import os

# Initialize the Telegram bot (Replace '' with your bot token)
bot = telepot.Bot('7186251726:AAasdasdasdbndNqoMIpCJ06fzow6mA')

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Convert indices to layer names
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Define the RTSP URL with authentication and port
username = 'admin'
password = 'caasdasan123'
ip_address = '192.168.1.6'
port = '10554'
rtsp_url = f'rtsp://{username}:{password}@{ip_address}:{port}/Streaming/channels/101'

# Initialize video capture object
cap = cv2.VideoCapture(rtsp_url)

# Create directories to store media
media_dir = "media"
photos_dir = os.path.join(media_dir, "photos")
videos_dir = os.path.join(media_dir, "videos")
os.makedirs(photos_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

# Initialize flag to keep track of notification status
notification_sent = False

# Function to send a notification with a photo to Telegram
def send_notification_photo(photo_path):
    print("Sending notification with photo...")
    try:
        print("Sending photo:", photo_path)
        with open(photo_path, 'rb') as photo_file:
            # Replace 'YOUR_TELEGRAM_CHAT_ID' with your Telegram chat ID
            bot.sendPhoto('832349138asdas1553', photo=photo_file, caption="Someone is on the door.")
        print("Notification with photo sent successfully")
    except Exception as e:
        print("Error sending notification with photo:", e)

# Function to send a notification with a video to Telegram
def send_notification_video(video_path):
    print("Sending notification with video...")
    try:
        print("Sending video:", video_path)
        with open(video_path, 'rb') as video_file:
            # Replace 'YOUR_TELEGRAM_CHAT_ID' with your Telegram chat ID
            bot.sendVideo('8913243815asdas53', video=video_file, caption="Someone is on the door.")
        print("Notification with video sent successfully")
    except Exception as e:
        print("Error sending notification with video:", e)

# Function to capture photo
def capture_photo():
    print("Capturing photo...")
    try:
        _, frame = cap.read()
        # Save photo with current date and time as filename
        current_time = time.strftime("%Y%m%d-%H%M%S")
        photo_path = os.path.join(photos_dir, f"{current_time}.jpg")
        cv2.imwrite(photo_path, frame)
        print("Photo captured and saved successfully")
        # Send notification with the captured photo
        send_notification_photo(photo_path)
    except Exception as e:
        print("Error capturing photo:", e)

# Function to record video
def record_video(start_time):
    print("Recording video...")
    try:
        # Initialize video writer object
        video_path = os.path.join(videos_dir, f"{start_time}.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 1080))  # Adjust resolution and FPS as needed

        # Record video for 15 seconds
        while time.time() - start_time < 15:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        # Release video writer
        out.release()

        print("Video recorded successfully")
        # Send notification with the recorded video
        send_notification_video(video_path)
    except Exception as e:
        print("Error recording video:", e)

# Main function for motion detection
def main():
    global notification_sent
    motion_detected = False
    start_time = 0

    print("Human motion detection started...")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

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

            # Draw bounding box around detected humans
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame with bounding boxes
            cv2.imshow('Motion Detection', frame)
            cv2.waitKey(1)

            # If motion detected and notification is not sent, capture photo and record video
            if len(indexes) > 0 and not notification_sent:
                print("Motion detected!")
                capture_photo()
                record_video(time.time())
                notification_sent = True

        except Exception as e:
            print("Error:", e)
            time.sleep(120)  # Wait for 2 minutes before restarting detection process

# Start motion detection
main()
