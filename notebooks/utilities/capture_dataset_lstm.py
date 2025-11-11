import cv2
import numpy as np
import os
import dotenv
import time
import mediapipe as mp

# Add the holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Define the detection and tracking confidence
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results


# Draw the landmarks on the image
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


# Extract keypoints from the results
def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])

# Ask the user for the gesture name
def get_gesture_name():
    gesture_name = input("Enter the gesture name to capture: ").strip().lower()
    if not gesture_name:
        print("Gesture name cannot be empty. Exiting.")
        exit()
    return gesture_name

# Set up constants
# Get the lakeFS data path from environment variable. Note: a local `.env` file
dotenv.load_dotenv()
DATA_PATH = os.getenv("LAKEFS_DATA_PATH")  # Path to store data
action = get_gesture_name() # Get gesture name to make directory
no_sequences_to_record = 30  # Number of sequences to record
timestamp = str(int(time.time()))  # Current timestamp for uniqueness

# Create directories for storing data
def create_gesture_directory():
    # Create directories for storing data
    # Directory for the gesture
    gesture_dir = os.path.join(DATA_PATH, action)
    # Create the gesture directory if it doesn't exist
    os.makedirs(
        gesture_dir, exist_ok=True
    )  
    # List existing sequence directories
    existing = [
        d for d in os.listdir(gesture_dir) if d.isdigit()
    ]  
    # Determine the starting sequence index
    start_sequence = (
        max(map(int, existing), default=-1) + 1
    )  
    return gesture_dir, start_sequence

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    
    # Create gesture directory and get starting sequence index
    gesture_dir, start_sequence = create_gesture_directory()
    # Print starting info
    print(f"Starting collection for '{action}' at sequence {start_sequence}")
    quite_flag = False

    # Loop through sequences
    for seq in range(start_sequence, start_sequence + no_sequences_to_record):

        # Create sequence directory
        seq_dir = os.path.join(gesture_dir, str(seq), timestamp)
        os.makedirs(seq_dir, exist_ok=True)
        print(f"Get ready for sequence {seq}.")

        # Buffer to store frames for video and keypoints
        frame_buffer = []
        keypoints_buffer = []

        # State variables for recording
        is_recording = False

        # Loop to capture loop while recording
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Make a copy of the image to display
            copy_image = image.copy()
            draw_landmarks(copy_image, results)

            # Catch the keyboard input
            key = cv2.waitKey(10) & 0xFF

            # If the "q" key is pressed, exit
            if key == ord("q"):
                print("Exiting...")
                quite_flag = True
                break
            # If the "space" key is pressed, start/stop recording
            if key == ord(" "):
                if not is_recording:
                    is_recording = True
                    print(f"[Sequence {seq}] Recording started.")
                    # Clear buffers
                    frame_buffer.clear()
                    keypoints_buffer.clear()
                else:
                    is_recording = False
                    print(f"[Sequence {seq}] Recording stopped.")
                    break  # Exit the loop to save the sequence
            
            # If recording, store the frame and keypoints
            if is_recording:
                keypoints = extract_keypoints(results)
                frame_buffer.append(image)
                keypoints_buffer.append(keypoints)

                # Show the frame to the user 
                cv2.putText(copy_image, "OPNEMEN...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(copy_image, f"Frames: {len(frame_buffer)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
            else:
                cv2.putText(copy_image, f"Ready for sequence {seq}. Press SPACE to start recording.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                
            cv2.imshow("OpenCV Feed", copy_image)

        # If the user chose to quit
        if quite_flag: 
            print("Quitting the program.")
            break

        # After recording, collect frames until we reach the desired sequence length
        if len(frame_buffer) > 0:
            print(f"Saving sequence {seq} with {len(frame_buffer)} frames")

        for frame_num in range(len(frame_buffer)):
            # Get the data out of the buffers
            image = frame_buffer[frame_num]
            keypoints = keypoints_buffer[frame_num]

            # Define the paths to save
            image_path = os.path.join(seq_dir, f"frame_{frame_num}.png")
            keypoints_path = os.path.join(seq_dir, f"keypoints_{frame_num}.npy")

            # Save the image and keypoints
            cv2.imwrite(image_path, image)
            np.save(keypoints_path, keypoints)


print("Data collection complete.")
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
