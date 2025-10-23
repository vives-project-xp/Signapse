import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import numpy as np

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    )

# List of image files to process
IMAGE_FILES = ['./images/test.jpg']

# Add padding to the bottom of the image
def pad_bottom(image, pad):
    return cv.copyMakeBorder(image, 0, pad, 0, 0, cv.BORDER_REPLICATE)

# Enlarge the image if its smallest side is less than min_side
def ensure_min_size(image, min_side):
    h, w = image.shape[:2]
    scale = max(1.0, min_side / max(h, w))
    if scale == 1.0: return image
    return cv.resize(image, (int(w*scale), int(h*scale)), interpolation=cv.INTER_LINEAR)

# Indices for Metacarpophalangeal joints and fingertip landmarks 
MCP = [5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]

# Function to check if the wrist position is plausible
def wrist_plausible(lms, W, H, tol=0.55):
    pts = np.array([[lm.x*W, lm.y*H] for lm in lms])
    wrist = pts[0]
    mcp = pts[MCP]
    tips = pts[TIPS]
    hand_span = np.linalg.norm(pts[5]-pts[17])
    
    # Space between MCP index and pinky
    if hand_span < 1e-6: 
        return False
    # Wrist should have some distance from MCPs
    if np.mean(np.linalg.norm(mcp - wrist, axis=1))/hand_span < tol:
        return False
    # Wrist should be below the MCPs
    if not (wrist[1] > np.percentile(tips[:,1], 60)):
        return False
    
    return True

for idx, file in enumerate(IMAGE_FILES):
    img = cv.imread(file)
    if img is None:
        print(f"[ERR] cannot read {file}")
        continue

    img = ensure_min_size(pad_bottom(img, 40), 512) # adjust if needed

    picked = None

    # Toggle variable to enable/disable variants
    USE_VARIANTS = False
    def variants_selector(image):
        if USE_VARIANTS:
            yield image, "orig"
            yield cv.flip(image, 1), "flip"
            for ang in (-15, 15):
                M = cv.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), ang, 1.0)
                yield cv.warpAffine(image, M, (image.shape[1], image.shape[0])), f"rot{ang}"
        else:
            yield image, "orig"

    for var, tag in variants_selector(img):
        rgb = cv.cvtColor(var, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        h, w = var.shape[:2]
        lms = res.multi_hand_landmarks[0].landmark
        if wrist_plausible(lms, w, h):
            picked = (var, res, tag)
            break

    if not picked:
        print(f"[SKIP] wrist failed: {file}")
        continue

    var, res, tag = picked
    annotated = var.copy()
    for hand_landmarks in res.multi_hand_landmarks:
        drawing.draw_landmarks(
            annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

    cv.imshow(f"ok:{tag}:{file}", annotated)
    cv.waitKey(0)
    cv.imwrite(f"/tmp/annotated_{idx}_{tag}.png", annotated)
cv.destroyAllWindows()