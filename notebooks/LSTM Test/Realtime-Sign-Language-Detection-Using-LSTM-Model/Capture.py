import os
import cv2
import numpy as np
from cv2 import VideoWriter_fourcc# type: ignore
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ---------------- CONFIG ----------------
DATA_PATH = 'MP_Data'
NO_SEQUENCES_TO_RECORD = 20    # How many video sequences to capture
SEQUENCE_LENGTH = 30           # Frames per sequence
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # Webcam resolution
# ----------------------------------------

# --- Ask user for gesture name ---
gesture_name = input("Enter the gesture name to capture: ").strip().lower()
if not gesture_name:
    print("❌ Gesture name cannot be empty. Exiting.")
    exit()

# --- Prepare directories ---
gesture_dir = os.path.join(DATA_PATH, gesture_name)
os.makedirs(gesture_dir, exist_ok=True)

# Find next sequence index
existing = [d for d in os.listdir(gesture_dir) if d.isdigit()]
start_sequence = max(map(int, existing), default=-1) + 1
print(f"📹 Starting collection for '{gesture_name}' at sequence {start_sequence}")

# --- Helper: extract Mediapipe keypoints ---
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- Video capture setup ---
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    for seq in range(start_sequence, start_sequence + NO_SEQUENCES_TO_RECORD):
        seq_dir = os.path.join(gesture_dir, str(seq))
        os.makedirs(seq_dir, exist_ok=True)

        # Setup video writer
        video_path = os.path.join(seq_dir, "video.avi")
        fourcc = VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

        print(f"Recording sequence {seq}...")

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Process frame with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, list(mp_holistic.POSE_CONNECTIONS))# type: ignore
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, list(mp_holistic.HAND_CONNECTIONS))# type: ignore
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, list(mp_holistic.HAND_CONNECTIONS))# type: ignore

            # Display info text
            if frame_num == 0:
                cv2.putText(image, f'STARTING {gesture_name.upper()} - SEQ {seq}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Capture', image)
                cv2.waitKey(1500)  # Short pause before recording
            else:
                cv2.putText(image, f'{gesture_name.upper()} | Seq {seq} Frame {frame_num}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save keypoints
            keypoints = extract_keypoints(results)
            np.save(os.path.join(seq_dir, f"{frame_num}.npy"), keypoints)

            # Save raw frame
            out.write(frame)

            # Show live view
            cv2.imshow('Capture', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Stopping early.")
                break

        out.release()  # Save video to disk

        # Break outer loop if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
