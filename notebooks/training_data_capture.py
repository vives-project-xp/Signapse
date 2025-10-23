import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import json, os, string, time

# ==== SETTINGS ====
SHOW_BOUNDING_BOX = True   # Draw bounding box around detected hand
CROP_IMAGES = True         # Save cropped images around the hand
PADDING = 40               # Extra pixels around bounding box when cropping

# ==== INITIALIZE MEDIAPIPE ====
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
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
os.makedirs("data", exist_ok=True)
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

def save_landmark_data(letter, landmarks_list, frame, bbox_norm=None, left_mode=False):
    """
    Save cropped image and append landmarks for the given label.
    Stores all positional data in one cumulative JSON file per letter.
    """
    letter_folder = os.path.join("data", letter.upper())
    img_folder = os.path.join(letter_folder, "images")
    os.makedirs(img_folder, exist_ok=True)

    json_path = os.path.join(letter_folder, f"{letter.upper()}_landmark_data.json")

    count_images = len(os.listdir(img_folder))
    timestamp = int(time.time() * 1000)
    base_name = f"{letter.upper()}_{count_images}_{timestamp}"
    img_name = f"{base_name}.jpg"
    img_path = os.path.join(img_folder, img_name)

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

    # Load existing JSON or initialize
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []

    # Append new entry
    all_data.append({
        "image_id": count_images + 1,
        "timestamp": timestamp,
        "label": letter.upper(),
        "image_file": f"images/{img_name}",
        "left_handed_mode": bool(left_mode),
        "bounding_box": bbox_for_saving,
        "landmarks": landmarks_for_saving
    })

    # Write back
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"[SAVED] {letter.upper()}{' (mirrored)' if left_mode else ''} → {img_name}")

print("Press any letter (A–Z) to save data. Press SPACE to toggle left/right-hand mode. Close the window to exit.")

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

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing.draw_landmarks(
                img_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_style, connection_style,
            )

        first_hand = result.multi_hand_landmarks[0]
        landmarks_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in first_hand.landmark]
        bbox_norm = get_bbox_from_landmarks(landmarks_list)

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
            save_landmark_data(letter, landmarks_list, img_bgr, bbox_norm, left_mode=LEFT_HANDED_MODE)
        else:
            print("[WARN] No hand detected when capturing.")

# ==== CLEANUP ====
cam.release()
cv.destroyAllWindows()
