import cv2
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'MP_Data'
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences_to_record = 20  # Number of sequences (samples) to record in this single session
sequence_length = 30         # Number of frames per sequence

# --- NEW: User Input for Action ---
print("Available actions:", actions)
selected_action = input("Enter the sign you want to capture (e.g., 'hello'): ").strip().lower()

if selected_action not in actions:
    print(f"Error: '{selected_action}' is not a valid action. Exiting.")
    exit()

# --- NEW: Determine the starting sequence number for this action ---
action_dir = os.path.join(DATA_PATH, selected_action)
if not os.path.exists(action_dir):
    start_sequence = 0
else:
    # Find existing sequence folders and determine the next one
    existing_sequences = [d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d)) and d.isdigit()]
    if existing_sequences:
        # Start recording after the highest existing sequence number
        start_sequence = max([int(d) for d in existing_sequences]) + 1
    else:
        start_sequence = 0

print(f"Starting collection for action: '{selected_action}' from sequence number: {start_sequence}")
# -----------------------------------------------------------------

# The folder creation loop is adjusted to create folders for the sequences we are about to record.
for sequence in range(start_sequence, start_sequence + no_sequences_to_record):
    os.makedirs(os.path.join(DATA_PATH, selected_action, str(sequence)), exist_ok=True)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # --- CHANGED: Loop only over the sequences for the selected action ---
    for sequence in range(start_sequence, start_sequence + no_sequences_to_record):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Display status
            display_action = selected_action  # Use the selected action
            
            if frame_num == 0:
                cv2.putText(image, f'STARTING COLLECTION FOR {display_action}', (20,40),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)  # 2-sec pause before recording
            else:
                cv2.putText(image, f'Collecting {display_action} | Video {sequence}', (10,40),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Extract and save keypoints
            keypoints = extract_keypoints(results)
            # Use selected_action in the save path
            np.save(os.path.join(DATA_PATH, selected_action, str(sequence), f"{frame_num}.npy"), keypoints)

            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Break the outer sequence loop if 'q' was pressed in the inner frame loop
        if cv2.waitKey(1) & 0xFF == ord('q'): 
             break

cap.release()
cv2.destroyAllWindows()