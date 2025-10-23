import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import csv, os, string, time
from typing import Optional

# ==== SETTINGS ====
SHOW_BOUNDING_BOX = True   # Draw bounding box around detected hand
CROP_IMAGES = True         # Save cropped images around the hand
PADDING = 40               # Extra pixels around bounding box when cropping
MIN_DETECTION_CONFIDENCE = 0.7  # Higher confidence for cleaner training data
SAMPLES_PER_LETTER = 100   # Target samples per letter for balanced dataset

# ==== INITIALIZE MEDIAPIPE ====
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand for alphabet (cleaner training)
    model_complexity=1,
    min_detection_confidence=0.7,  # Higher threshold for quality data
    min_tracking_confidence=0.7,   # Maintain high quality across frames
)

# ==== CAMERA SETTINGS ====
width, height = 1280, 720
cam = cv.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

# ==== DRAWING STYLES ====
landmark_style = drawing.DrawingSpec(color=(3, 252, 236), thickness=5, circle_radius=3)
connection_style = drawing.DrawingSpec(color=(235, 252, 3), thickness=2)

# ==== DATA STORAGE ====
# Consolidate all samples into ONE CSV compatible with asl_model/data_utils.py expectations.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASL_MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "asl_model"))
IMAGES_ROOT = os.path.join(ASL_MODEL_DIR, "images")
CSV_PATH = os.path.join(IMAGES_ROOT, "hand_landmarks.csv")

# Ensure base directories exist
os.makedirs(IMAGES_ROOT, exist_ok=True)

# Manual toggle to mirror left-hand captures into a right-hand canonical frame (optional)
LEFT_HANDED_MODE = False  # starts in right-handed mode

def get_bbox_from_landmarks(landmarks_list):
    xs = [lm['x'] for lm in landmarks_list]
    ys = [lm['y'] for lm in landmarks_list]
    return {
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
        "y_min": float(min(ys)),
        "y_max": float(max(ys))
    }

def save_landmark_data(letter, landmarks_list, frame, bbox_norm=None, left_mode=False, handedness_detected: Optional[str] = None):
    """
    Save cropped image and append landmarks for the given label.
    Appends ALL samples to one cumulative CSV file at asl_model/images/hand_landmarks.csv.
    Images are stored under asl_model/images/<LETTER>/.
    """
    # Organize images per letter under the shared images root
    letter_img_dir = os.path.join(IMAGES_ROOT, letter.upper())
    os.makedirs(letter_img_dir, exist_ok=True)

    
    count_images = len([f for f in os.listdir(letter_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    timestamp = int(time.time() * 1000)
    base_name = f"{letter.upper()}_{count_images}_{timestamp}"
    img_name = f"{base_name}.jpg"
    img_path = os.path.join(letter_img_dir, img_name)

    # Mirror landmarks and image if left-handed mode is active
    if left_mode:
        mirrored_landmarks = [{"x": 1.0 - lm["x"], "y": lm["y"], "z": lm["z"]} for lm in landmarks_list]
        if bbox_norm:
            mirrored_bbox = {
                "x_min": float(max(0.0, 1.0 - bbox_norm["x_max"])),
                "x_max": float(min(1.0, 1.0 - bbox_norm["x_min"])),
                "y_min": float(bbox_norm["y_min"]),
                "y_max": float(bbox_norm["y_max"]),
            }
        else:
            mirrored_bbox = None
        frame_to_save = cv.flip(frame.copy(), 1)
        landmarks_for_saving = mirrored_landmarks
        bbox_for_saving = mirrored_bbox
    else:
        frame_to_save = frame.copy()
        landmarks_for_saving = landmarks_list
        bbox_for_saving = bbox_norm

    # Crop around bounding box if enabled
    if CROP_IMAGES and bbox_for_saving:
        h, w = frame_to_save.shape[:2]
        x_min_px = max(int(bbox_for_saving["x_min"] * w) - PADDING, 0)
        x_max_px = min(int(bbox_for_saving["x_max"] * w) + PADDING, w)
        y_min_px = max(int(bbox_for_saving["y_min"] * h) - PADDING, 0)
        y_max_px = min(int(bbox_for_saving["y_max"] * h) + PADDING, h)
        if x_max_px > x_min_px and y_max_px > y_min_px:
            frame_to_save = frame_to_save[y_min_px:y_max_px, x_min_px:x_max_px]
        else:
            print("[WARN] Invalid crop; saving full frame.")

    # Save image
    cv.imwrite(img_path, frame_to_save)

    # Append to the shared CSV at asl_model/images/hand_landmarks.csv
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='') as csvfile:
        # Keep 'class' and 'landmarks' columns for compatibility with asl_model/data_utils.py
        fieldnames = ['image_id', 'timestamp', 'class', 'image_file', 'handedness', 'left_handed_mode', 'landmarks', 'bounding_box']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'image_id': count_images + 1,
            'timestamp': timestamp,
            'class': letter.upper(),
            # Store path relative to images root for portability
            'image_file': f"{letter.upper()}/{img_name}",
            'handedness': handedness_detected if handedness_detected in ("Left", "Right") else "Unknown",
            'left_handed_mode': bool(left_mode),
            'landmarks': landmarks_for_saving,
            'bounding_box': bbox_for_saving
        })

    # # Load existing JSON or initialize
    # if os.path.exists(json_path):
    #     with open(json_path, "r") as f:
    #         try:
    #             all_data = json.load(f)
    #         except json.JSONDecodeError:
    #             all_data = []
    # else:
    #     all_data = []

    # # Append new entry
    # all_data.append({
    #     "image_id": count_images + 1,
    #     "timestamp": timestamp,
    #     "label": letter.upper(),
    #     "image_file": f"images/{img_name}",
    #     "left_handed_mode": bool(left_mode),
    #     "bounding_box": bbox_for_saving,
    #     "landmarks": landmarks_for_saving
    # })

    # # Write back
    # with open(json_path, "w") as f:
    #     json.dump(all_data, f, indent=2)

    print(f"[SAVED] {letter.upper()}{' (mirrored)' if left_mode else ''} → {img_name}  | CSV: {os.path.relpath(CSV_PATH, ASL_MODEL_DIR)}")

def get_dataset_progress():
    """Display current sample counts per letter to help balance the dataset."""
    if not os.path.isfile(CSV_PATH):
        return {}
    
    with open(CSV_PATH, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counts = {}
        for row in reader:
            letter = row['class']
            counts[letter] = counts.get(letter, 0) + 1
    return counts

def display_progress_overlay(img, letter_counts):
    """Draw a compact progress table on the image."""
    alphabet = string.ascii_uppercase
    y_offset = 90
    col_width = 80
    for i, letter in enumerate(alphabet):
        count = letter_counts.get(letter, 0)
        x = 30 + (i % 13) * col_width
        y = y_offset + (i // 13) * 30
        color = (0, 255, 0) if count >= SAMPLES_PER_LETTER else (0, 165, 255) if count > 0 else (180, 180, 180)
        text = f"{letter}:{count}"
        cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

print("Press any letter (A–Z) to save data. Press SPACE to toggle left/right-hand mode. Close the window to exit.")

# Track dataset progress
letter_counts = get_dataset_progress()
total_samples = sum(letter_counts.values())
print(f"\n[DATASET] Current samples: {total_samples} total")
if letter_counts:
    print(f"          Letters captured: {', '.join(sorted(letter_counts.keys()))}")
print()

# ==== MAIN LOOP ====
while True:
    success, img = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    bbox_norm = None
    landmarks_list = None

    handedness_detected = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing.draw_landmarks(
                img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_style, connection_style,
            )

        first_hand = result.multi_hand_landmarks[0]
        landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in first_hand.landmark]
        bbox_norm = get_bbox_from_landmarks(landmarks_list)

        # If available, record handedness detected by MediaPipe for the first hand
        try:
            if result.multi_handedness:
                handedness_detected = result.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
        except Exception:
            handedness_detected = None

        if SHOW_BOUNDING_BOX and bbox_norm:
            x_min_px = int(bbox_norm["x_min"] * width)
            x_max_px = int(bbox_norm["x_max"] * width)
            y_min_px = int(bbox_norm["y_min"] * height)
            y_max_px = int(bbox_norm["y_max"] * height)
            cv.rectangle(
                img_bgr,
                (max(x_min_px - PADDING, 0), max(y_min_px - PADDING, 0)),
                (min(x_max_px + PADDING, width), min(y_max_px + PADDING, height)),
                (0, 255, 0),
                2
            )

    display = cv.flip(img_bgr, 1)
    label_text = "LEFT-HANDED MODE" if LEFT_HANDED_MODE else "RIGHT-HANDED MODE"
    cv.putText(display, label_text, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Show dataset progress
    display = display_progress_overlay(display, letter_counts)
    
    cv.imshow("Sign Capture", display)

    key = cv.waitKey(20) & 0xFF
    if cv.getWindowProperty("Sign Capture", cv.WND_PROP_VISIBLE) < 1:
        break

    if key == 32:  # Spacebar toggles mode
        LEFT_HANDED_MODE = not LEFT_HANDED_MODE
        print(f"[INFO] {'Left' if LEFT_HANDED_MODE else 'Right'}-handed mode activated.")

    elif key in [ord(c.lower()) for c in string.ascii_lowercase]:
        letter = chr(key).upper()
        if landmarks_list and bbox_norm:
            save_landmark_data(letter, landmarks_list, img_bgr, bbox_norm, left_mode=LEFT_HANDED_MODE, handedness_detected=handedness_detected)
            letter_counts[letter] = letter_counts.get(letter, 0) + 1  # Update live counter
        else:
            print("[WARN] No hand detected when capturing.")

# ==== CLEANUP ====
cam.release()
cv.destroyAllWindows()
