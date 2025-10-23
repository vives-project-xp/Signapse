import cv2
import threading
import queue
import numpy as np
import time
import tensorflow as tf
import keras 
import os
import mediapipe as mp 

print("Current working directory:", os.getcwd())

# --- Configuration & MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sequence_length = 30 
actions = np.array(['hello', 'thanks', 'iloveyou'])

# --- NEW CONFIGURATION PARAMETERS ---
threshold = 0.85          # Only display predictions with this confidence or higher
smoothing_window = 10     # History buffer size for median filtering
# NEW: MINIMUM LANDMARK THRESHOLD
min_landmarks_detected = 10 # Minimum number of landmarks (out of 258) required to attempt inference
# -----------------------------------

# --- Keypoint Extraction Function (Copied from Data Collection) ---
def extract_keypoints(results):
    """Converts MediaPipe results into a flattened NumPy array (258 features)."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Store the number of non-zero features for later check
    keypoints = np.concatenate([pose, lh, rh])
    num_detected = np.count_nonzero(keypoints)
    
    return keypoints, num_detected

# --- Threaded Video Capture Class ---
class VideoCaptureThread:
    def __init__(self, src=0, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout=1.0):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# --- Inference Worker Class ---
class InferenceWorker:
    def __init__(self):
        self.model = None
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
        self.sequence = [] 
        self.prediction_history = [] 
        self.in_q = queue.Queue(maxsize=2)
        self.out_q = queue.Queue(maxsize=2)
        self.running = False
        self.thread = threading.Thread(target=self._runner, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def load_model(self):
        print("Loading TensorFlow model...")
        
        custom_objects = {
            "Orthogonal": keras.initializers.Orthogonal, 
        }
        
        self.model = tf.keras.models.load_model(
            'model_fixed.keras', 
            custom_objects=custom_objects, 
            compile=False
        ) 
        print("Model loaded successfully.")

    def _runner(self):
        try:
            self.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.running = False
            return

        while self.running:
            try:
                frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # 1. Convert to RGB for MediaPipe (Optimized)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False 
                results = self.holistic.process(image)
                image.flags.writeable = True 
                
                # 2. Extract Keypoints and count detections
                keypoints, num_detected = extract_keypoints(results)
                
                pred = None

                # NEW LOGIC: Only perform inference if enough landmarks are detected
                if num_detected >= min_landmarks_detected:
                    
                    # 3. Update Sequence Buffer
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-sequence_length:] 
                    
                    # 4. Inference only once the sequence buffer is full
                    if len(self.sequence) == sequence_length:
                        input_seq = np.expand_dims(self.sequence, axis=0) 
                        raw_pred = self.model.predict(input_seq, verbose=0)[0] 
                        
                        # Apply smoothing
                        self.prediction_history.append(raw_pred)
                        self.prediction_history = self.prediction_history[-smoothing_window:]
                        pred = np.median(self.prediction_history, axis=0)
                else:
                    # Clear sequence if keypoints are lost, preventing stale data contamination
                    self.sequence = []
                    # Create a "No Sign" vector: low confidence for all classes
                    pred = np.full(len(actions), 0.05) 
                    
                # 5. Output prediction (if one was made)
                if pred is not None:
                    # Clear out stale predictions in the queue
                    while not self.out_q.empty():
                         self.out_q.get_nowait()
                    self.out_q.put(pred)

            except Exception as e:
                print(f"Inference error: {e}")

    def submit(self, frame):
        try:
            self.in_q.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout=0.0):
        try:
            return True, self.out_q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        self.holistic.close()
        self.thread.join(timeout=1.0)

# --- Main Detection Loop ---
def main():
    print("Pre-initializing TensorFlow...")
    _ = tf.zeros(1) 
    print("TensorFlow initialized.")

    cap = VideoCaptureThread(0, width=640, height=480).start() 
    worker = InferenceWorker().start()

    draw_holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

    try:
        last_time = time.time()
        # Initialize a variable to hold the last valid prediction for display continuity
        current_prediction_label = "Buffering..." 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # --- Drawing Keypoints (Main Thread) ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False 
            draw_results = draw_holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, draw_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, draw_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, draw_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # --- Inference Submission and Retrieval ---
            worker.submit(frame)
            has_res, res = worker.get_result()

            now = time.time()
            time_diff = now - last_time
            
            # Calculate FPS
            if time_diff < 1e-6:
                fps = 0.0
            else:
                fps = 1.0 / time_diff
                
            last_time = now

            # Graceful exit check
            if not worker.running and not has_res:
                print("Inference worker failed to start or stopped. Exiting loop.")
                break

            # --- Prediction Logic and Stabilization ---
            if has_res and res.size > 0:
                predicted_index = np.argmax(res)
                confidence = res[predicted_index]
                
                # Check if the result is below the "No Sign" threshold (i.e., we forced a low confidence output)
                if predicted_index == 0 and confidence < 0.2 and np.all(res < 0.2):
                    current_prediction_label = "No Sign Detected"
                
                # Check for high confidence prediction
                elif confidence > threshold:
                    predicted_action = actions[predicted_index]
                    current_prediction_label = f"Sign: {predicted_action} ({confidence:.2f})"
                
                # Maintain the last sign detected if a new one hasn't crossed the threshold
                elif current_prediction_label.startswith("Sign"):
                    pass # Keep the last sign label for smoothness
                
                # Default for low-confidence frames (not buffering)
                elif not current_prediction_label.startswith("Buffering"):
                    current_prediction_label = "..." 

            elif not has_res:
                # Still buffering the 30 frames
                if len(worker.sequence) < sequence_length:
                     current_prediction_label = f"Buffering... ({len(worker.sequence)}/{sequence_length})"
                else:
                    # If buffer is full, but no prediction has returned yet (latency)
                    current_prediction_label = "Processing..." 
            
            # --- Display ---
            cv2.putText(frame, f"{current_prediction_label} | FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Real-Time Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        draw_holistic.close()
        cap.stop()
        worker.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()