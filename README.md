# StealthMode
Real-Time Player Tracking & Re-Identification in Sports Videos
This project provides a robust solution for tracking football players in a video feed using a combination of object detection and deep learning-based tracking. The system is designed to detect players, assign them unique IDs, and maintain these IDs even when players temporarily leave and re-enter the camera's view, simulating a real-time re-identification scenario.

(Replace the link above with a GIF of your final output video for a great visual demo!)

📋 Objective
The primary goal, as outlined in the project description, is to process a 15-second video clip (15sec_input_720p.mp4) to:

Detect all players using a provided fine-tuned YOLOv8 model.

Assign a unique and persistent ID to each player.

Maintain the correct ID for a player even if they go out of frame and reappear later.

✨ Features
Accurate Player Detection: Utilizes a custom-trained YOLOv8 model to detect players and balls.

Multi-Object Tracking: Implements the DeepSORT algorithm to track multiple players simultaneously across frames.

Robust ID Persistence: Handles occlusions and temporary exits from the frame by tuning tracker parameters (max_age, n_init), ensuring players retain their original ID upon reappearance.

Real-Time Simulation: Processes the video frame-by-frame to simulate a live tracking feed.

Annotated Video Output: Generates an output video file with bounding boxes and player IDs drawn on each frame.

🛠️ Technologies Used
Python 3.8+

Ultralytics YOLOv8: For state-of-the-art object detection.

DeepSORT (via deep-sort-realtime): For tracking using Kalman filtering and a deep association metric.

OpenCV: For video processing and drawing annotations.

NumPy: For numerical operations.

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
Bash

git clone https://github.com/DJier/StealthMode.git
cd player-tracking-project
2. Set Up a Virtual Environment (Recommended)
Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install all the required Python packages using the provided requirements.txt file (or install them manually).

Bash

pip install ultralytics deep-sort-realtime opencv-python numpy
4. Download Project Files
YOLOv8 Model: Download the best.pt model file from the link provided in the project description.

Link: Google Drive Model Download

Place the best.pt file in the root directory of the project.

Input Video: Ensure you have the 15sec_input_720p.mp4 video file in the root directory.

5. Run the Tracking Script
Execute the main Python script to start the tracking process.

Bash

python tracking_script.py
(Make sure to name your Python file stealthMode.py or update the command accordingly)

The script will process the video and save the result as output_stable_tracking_stable.mp4 in the same directory.

⚙️ How It Works
The tracking pipeline operates in a sequential manner for each frame of the video:

Detection: The YOLOv8 model (best.pt) analyzes the frame to detect all instances of the 'player' class, returning their bounding box coordinates and a confidence score.

Formatting: These detections are formatted into a list that DeepSORT can understand: ([x1, y1, x2, y2], confidence, class_name).

Tracker Update: The list of current detections is passed to the tracker.update_tracks() method. DeepSORT performs two key steps:

It uses a Kalman Filter to predict the next position of already-tracked players.

It calculates the similarity between new detections and existing tracks using IoU (Intersection over Union) for positional matching and cosine distance on appearance features (learned by its re-ID model).

Association: Based on this similarity, it associates new detections with existing tracks. If a detection doesn't match any existing track, a new track ID is created.

ID Persistence: By setting a high max_age (e.g., 90 frames), the tracker keeps a "memory" of a track for a short period even if it's not detected. If the player reappears within this window, the tracker re-associates them with their original ID instead of creating a new one.

🔧 Configuration
You can fine-tune the tracker's performance by adjusting the parameters in the stable_params dictionary within the script:

conf_threshold: The minimum confidence score for a YOLO detection to be considered.

max_age: The maximum number of frames to keep a track alive without a new detection. Crucial for re-identification.

n_init: The number of consecutive frames a track must be detected before it's confirmed. Helps prevent false positives.

max_cosine_distance: The maximum cosine distance for appearance matching.

max_iou_distance: The maximum IoU distance for positional matching.
