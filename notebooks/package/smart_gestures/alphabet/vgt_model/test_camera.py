from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Optional, Tuple
from collections import OrderedDict

import cv2
import torch
import numpy as np
import mediapipe as mp

from .model_utils import create_model, DEVICE, MODEL_DIR
from .data_utils import _normalize_landmarks, get_classes
# TODO: Move to other folder
from notebooks.utilities.spelling_utils import TemporalSpeller, WordAssembler  # type: ignore


def find_latest_model(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    files = [os.path.join(path, f)
             for f in os.listdir(path) if f.endswith(".pth")]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_state_dict_forgiving(model: torch.nn.Module, model_path: str) -> None:
    print("Loading weights from", model_path)
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model_state_dict", "state_dict", "model", "net")):
        state = ckpt.get("model_state_dict") or ckpt.get(
            "state_dict") or ckpt.get("model") or ckpt.get("net")
    else:
        state = ckpt
    if state and len(state) and next(iter(state)).startswith("module."):
        state = OrderedDict((k.replace("module.", "", 1), v)
                            for k, v in state.items())
    missing, unexpected = model.load_state_dict(
        state, strict=False)  # type: ignore
    if missing:
        print("[warn] missing keys:", missing)
    if unexpected:
        print("[warn] unexpected keys:", unexpected)


def prepare_input(landmarks) -> np.ndarray:
    arr = _normalize_landmarks(
        landmarks, root_idx=0, scale_method="wrist_to_middle")
    return arr.reshape(-1)


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--interval", type=float, default=0.30,
                    help="sec tussen voorspellingen")
    ap.add_argument("--idle", type=float, default=1.2,
                    help="sec zonder hand => boundary")
    ap.add_argument("--conf_min", type=float, default=0.70,
                    help="min confidence om frame te accepteren")
    ap.add_argument("--dwell_frames", type=int, default=2,
                    help="# opeenvolgende frames voor letter-emissie")
    ap.add_argument("--min_word_len", type=int, default=2,
                    help="min lengte voor woord-emissie")
    ap.add_argument("--hf_polish", type=str, default=None,
                    help="HF model ID voor polish (optioneel)")
    ap.add_argument("--print_letters", action="store_true")
    ap.add_argument("--print_probs", action="store_true")
    ap.add_argument("--print_words", action="store_true")
    args = ap.parse_args(argv)

    classes = list(get_classes())
    model = create_model(len(classes), 63)

    model_path = args.model or find_latest_model(str(MODEL_DIR))
    if model_path is None:
        print("No model weights found in", MODEL_DIR,
              "and --model not provided. Exiting.")
        return
    load_state_dict_forgiving(model, model_path)
    model.to(DEVICE).eval()

    mp_hands = mp.solutions.hands  # type: ignore
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils  # type: ignore

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera", args.camera)
        return

    # Alleen spelling_utils:
    decoder = TemporalSpeller(conf_min=args.conf_min,
                              dwell_frames=args.dwell_frames,
                              min_word_len=args.min_word_len)
    speller = WordAssembler(hf_model_id=args.hf_polish)

    assembled_text: str = ""
    last_pred: Tuple[str, float] = ("", 0.0)
    last_time = 0.0
    last_hand_seen_ts = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            now = time.time()

            if results.multi_hand_landmarks:
                last_hand_seen_ts = now
                hand_lms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                lm_list = [{"x": lm.x * w, "y": lm.y * h, "z": lm.z}
                           for lm in hand_lms.landmark]

                if (now - last_time) >= args.interval:
                    with torch.no_grad():
                        t_in = torch.from_numpy(prepare_input(
                            lm_list)).float().to(DEVICE).unsqueeze(0)
                        logits = model(t_in)
                        probs = torch.softmax(
                            logits, dim=1).detach().cpu().numpy()[0]
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        label = classes[idx]
                        last_pred = (label, conf)
                        last_time = now

                        letter = decoder.consume(label, conf)
                        if letter is not None and args.print_letters:
                            if args.print_probs:
                                print(
                                    f"[LETTER {now:.2f}s] {letter} p={conf:.3f}", flush=True)
                            else:
                                print(
                                    f"[LETTER {now:.2f}s] {letter}", flush=True)

            # idle-based boundary
            word = decoder.boundary_if_idle(last_hand_seen_ts, args.idle)
            if word:
                pretty = speller.polish(speller.letters_to_words(list(word)))
                assembled_text = pretty
                if args.print_words:
                    print(f"[WORD  {now:.2f}s] {assembled_text}", flush=True)

            # HUD
            if last_pred[0]:
                cv2.putText(frame, f"Pred: {last_pred[0]} ({last_pred[1]*100:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.putText(frame, f"Letters: {decoder.live_buffer}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Output:  {assembled_text}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "[SPACE]=force boundary   [Q/ESC]=quit",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Camera test zonder LetterStream", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                wtxt = decoder.force_boundary()
                if wtxt:
                    pretty = speller.polish(
                        speller.letters_to_words(list(wtxt)))
                    assembled_text = pretty
                    if args.print_words:
                        print(
                            f"[WORD  {time.time():.2f}s] {assembled_text}", flush=True)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
