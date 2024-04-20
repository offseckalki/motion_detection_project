<h1>Motion Detection with Telegram Notification</h1>

<p>This Python script performs human motion detection using OpenCV and YOLO (You Only Look Once) object detection model. When motion is detected, it captures a photo and sends a notification with the photo to a Telegram chat. Additionally, it records a video when motion is detected and sends the video as a notification.</p>

<h2>Prerequisites</h2>

<ul>
  <li>Python 3</li>
  <li>OpenCV (cv2)</li>
  <li>telepot</li>
  <li>numpy</li>
</ul>

<h2>Installation</h2>

<p>You can install the required Python packages using pip:</p>

<pre><code>pip install opencv-python telepot numpy
</code></pre>

<h2>Setup</h2>

<ol>
  <li>Replace <code>'YOUR_TELEGRAM_CHAT_ID'</code> with your Telegram chat ID.</li>
  <li>Replace <code>''</code> with your Telegram bot token.</li>
  <li>Make sure you have the YOLOv3 weights (<code>yolov3.weights</code>) and configuration file (<code>yolov3.cfg</code>) in the same directory as the script.</li>
</ol>

<h2>Usage</h2>

<p>Run the script by executing:</p>

<pre><code>python motion_detection.py
</code></pre>

<p>The script will start detecting human motion from an RTSP stream. When motion is detected, it will capture a photo and send it as a notification to the specified Telegram chat. It will also record a video of the detected motion and send it as a notification.</p>

<h2>Configuration</h2>

<p>You can customize the following parameters in the script:</p>

<ul>
  <li><code>username</code>: Username for RTSP authentication.</li>
  <li><code>password</code>: Password for RTSP authentication.</li>
  <li><code>ip_address</code>: IP address of the RTSP camera.</li>
  <li><code>port</code>: Port number for the RTSP stream.</li>
  <li><code>photos_dir</code>: Directory to store captured photos.</li>
  <li><code>videos_dir</code>: Directory to store recorded videos.</li>
  <li><code>notification_sent</code>: Flag to keep track of notification status.</li>
  <li><code>start_time</code>: Variable to store the start time of motion detection.</li>
  <li><code>main()</code>: Main function for motion detection.</li>
</ul>

<h2>Credits</h2>

<p>This script utilizes the YOLO object detection model for detecting humans. YOLO is a state-of-the-art, real-time object detection system.</p>

<h2>License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
