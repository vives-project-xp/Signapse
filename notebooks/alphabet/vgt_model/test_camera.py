import time
import argparse
import os
from typing import Tuple

import cv2
import torch
import numpy as np
import mediapipe as mp

from model_utils import create_model, DEVICE, MODEL_DIR
from data_utils import normalize_landmarks, get_classes


def find_latest_model(path: str) -> str:
    # Look in MODEL_DIR for the newest .pth file
    if not os.path.exists(path):
        return None
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def prepare_input(landmarks) -> np.ndarray:
    """Normalize landmarks and flatten to (63,) as expected by the model."""
    arr = normalize_landmarks(landmarks, root_idx=0, scale_method="wrist_to_middle")
    return arr.reshape(-1)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default=None, help='Path to .pth weights (optional)')
    parser.add_argument('--interval', type=float, default=5.0, help='Seconds between predictions')
    parsed = parser.parse_args(args)

    classes = get_classes()
    num_classes = len(classes)
    in_dim = 63
    model = create_model(num_classes, in_dim)

    model_path = parsed.model or find_latest_model(MODEL_DIR)
    if model_path is None:
        print('No model weights found in', MODEL_DIR, 'and --model not provided. Exiting.')
        return

    print('Loading weights from', model_path)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(parsed.camera)
    if not cap.isOpened():
        print('Cannot open camera', parsed.camera)
        return

    last_time = 0.0
    last_pred = ('', 0.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            h, w, _ = frame.shape
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                # Convert mediapipe landmarks to simple list of dicts like dataset format
                lm_list = []
                for lm in hand_lms.landmark:
                    lm_list.append({'x': lm.x * w, 'y': lm.y * h, 'z': lm.z})

                # draw landmarks
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                if now - last_time >= parsed.interval:
                    inp = prepare_input(lm_list)
                    with torch.no_grad():
                        t = torch.from_numpy(inp).float().to(DEVICE)
                        outputs = model(t.unsqueeze(0))
                        probs = torch.softmax(outputs.cpu(), dim=1).numpy()[0]
                        top_idx = int(np.argmax(probs))
                        conf = float(probs[top_idx])
                        last_pred = (classes[top_idx], conf)
                        last_time = now

            # overlay prediction
            if last_pred[0]:
                text = f"{last_pred[0]} ({last_pred[1]*100:.1f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('Model camera test', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
