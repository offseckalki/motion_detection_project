import cv2
import telepot
import numpy as np
import time
import os

# Initialize the Telegram bot (Replace '' with your bot token)
bot = telepot.Bot('Your Bot Token')

# Function to send a notification with a video clip to Telegram
def send_notification(video_path):
    print("Sending notification...")
    try:
        print("Sending video:", video_path)
        with open(video_path,'rb') as video_file:
            # Replace 'YOUR_TELEGRAM_CHAT_ID' with your Telegram chat ID 
            bot.sendVideo('Your telegram chat ID', video=video_file)
        print("Notification sent successfully")
        
    except Exception as e:
        print("Error sending notification:", e)

# Main function to detect motion, record video, and notify on Telegram
def main():
    # Create a 'videos' folder if it doesn't exist
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # Define the RTSP URL with authentication and port
    username = 'admin'
    password = 'cpassword'
    ip_address = '192.168.1.6'
    port = '10554'
    rtsp_url = f'rtsp://{username}:{password}@{ip_address}:{port}/Streaming/channels/101'

    # Initialize video capture object
    cap = cv2.VideoCapture(rtsp_url)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize reference frame
    reference_frame = None

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

        # Print number of contours detected
        print("Number of contours detected:", len(contours))

        # Check if any contours (motion) are detected
        if contours:
            print("Motion detected!")

            # Print frame shape
            print("Frame shape:", frame.shape)

            # Create video writer object if not already created
            if 'out' not in locals():
                timestamp = int(time.time())
                video_path = f'videos/video_{timestamp}.mp4'
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))

            # Write frame to the video file
            out.write(frame)

            try:
                # Send notification with the video clip
                send_notification(video_path)
            except Exception as e:
                print("Error sending notification:", e)

            # Sleep to avoid continuous notifications for the same event
            time.sleep(30)

    # Release the camera and video writer
    cap.release()
    if 'out' in locals():
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
