import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Resolution of the webcam feed
width, height = 1280, 720
cam = cv.VideoCapture(0)
cam.set(3, width)
cam.set(4, height) 

# Style for drawing the landmarks and connections
landmark_style = drawing.DrawingSpec(color=(3, 252, 236), thickness=5, circle_radius=3)
connection_style = drawing.DrawingSpec(color=(235, 252, 3), thickness=2)

while cam.isOpened():
    success, img_rgb = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    # Convert image to RGB format for MediaPipe
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    # Process the image and detect hands
    hands_detected = hands.process(img_rgb)
    # Convert back to BGR format for OpenCV
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    if hands_detected.multi_hand_landmarks:
        # Draw hand landmarks and connections
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            drawing.draw_landmarks(
                img_rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

            # Draw bounding box around the hands
            margin = 30
            img_h, img_w, _ = img_rgb.shape
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            y_min, y_max = int(min(y_coords) * img_h), int(max(y_coords) * img_h)
            x_min, x_max = int(min(x_coords) * img_w), int(max(x_coords) * img_w)

            cv.rectangle(
                img_rgb,
                (x_min - margin, y_min - margin),
                (x_max + margin, y_max + margin),
                (0, 255, 0),
                2,
            )

    cv.imshow("Show Video", cv.flip(img_rgb, 1))

    # Exit on pressing 'q'
    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()