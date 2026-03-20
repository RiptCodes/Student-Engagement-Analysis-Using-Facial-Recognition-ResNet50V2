import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import threading
import time
from collections import deque

from config import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # let tf grow gpu memory as needed instead of reserving everything
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_models():
    # load both models once at startup
    face_detector = YOLO('yolov8n-face-lindevs.pt')
    face_detector.to('cuda')
    engagement_model = tf.keras.models.load_model(MODEL_PATH)
    return face_detector, engagement_model


def get_face(frame, yolo_model):
    # detect faces in the current frame
    detection_result = yolo_model(frame, verbose=False, imgsz=320)[0]
    detected_boxes = detection_result.boxes

    if detected_boxes is None or len(detected_boxes) == 0:
        return None, None

    # use the largest face (usually the main person in view)
    face_areas = (detected_boxes.xyxy[:, 2] - detected_boxes.xyxy[:, 0]) * (detected_boxes.xyxy[:, 3] - detected_boxes.xyxy[:, 1])
    biggest_face_idx = int(face_areas.argmax())
    x1, y1, x2, y2 = map(int, detected_boxes.xyxy[biggest_face_idx].tolist())

    frame_height, frame_width = frame.shape[:2]
    # add a little margin so the crop is less tight
    padding = int(0.3 * (x2 - x1))
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame_width, x2 + padding)
    y2 = min(frame_height, y2 + padding)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, IMG_SIZE)
    return face_crop, (x1, y1, x2, y2)


def predict(face_image, classifier):
    # preprocess the crop exactly like training
    model_input = face_image.astype(np.float32)
    model_input = tf.keras.applications.resnet_v2.preprocess_input(model_input)
    model_input = np.expand_dims(model_input, axis=0)

    # pick the top class and confidence from model output
    predictions = classifier(model_input, training=False).numpy()[0]
    top_class_idx = int(np.argmax(predictions))

    return LABELS[top_class_idx], float(predictions[top_class_idx]), float(predictions[1])


def get_colour(prob):
    # red at 0.0, amber at 0.5, green at 1.0
    if prob < 0.5:
        t = prob / 0.5
        return (0, int(165 * t), 220)
    else:
        t = (prob - 0.5) / 0.5
        return (0, int(165 + 35 * t), int(220 * (1 - t)))


def draw_result(frame, label, confidence, engagement_prob, face_box):
    x1, y1, x2, y2 = face_box
    box_color = get_colour(engagement_prob)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label_text = f"{label} {confidence*100:.0f}%"
    text_width, text_height = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 8, y1), box_color, -1)
    cv2.putText(frame, label_text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def draw_engagement_bar(frame, engagement_prob):
    # simple engagement meter at the bottom
    bar_x, bar_y, bar_width, bar_height = 20, frame.shape[0] - 35, 220, 16 # size and position of the bar after some experimentation
    clipped_prob = float(np.clip(engagement_prob, 0.0, 1.0)) # probability cannot go out of bound meaning between 0-1
    filled_width = int(bar_width * clipped_prob) # change colour depending on engagement level and fill the bar accordingly
    bar_color = get_colour(clipped_prob) # red at 0.0, amber at 0.5, green at 1.0

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1) # background of the bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1) # filled portion of the bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (220, 220, 220), 1) # border of the bar
    cv2.putText(frame, f"engagement {clipped_prob*100:.0f}%", (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1) # label above the bar

 # heads up display with fps and controls info
def draw_hud(frame, smoothed_fps, is_paused):
    # small runtime overlay + controls
    cv2.putText(frame, f"fps {smoothed_fps:.1f}", (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(frame, "p pause  s save pic  q quit", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if is_paused:
        cv2.putText(frame, "paused", (frame.shape[1] - 105, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2)

# this thread class allows the main loop to keep the webcam preview smooth while running detection and classification in the background
class DetectionThread(threading.Thread):
    def __init__(self, face_detector, engagement_model):
        super().__init__(daemon=True)
        self.face_detector = face_detector
        self.engagement_model = engagement_model
        self.latest_frame = None
        self.latest_result = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.running = True

    # 
    def run(self):
        while self.running:
            # wait for a new frame from the main loop
            self.new_frame_event.wait()
            self.new_frame_event.clear()

            with self.frame_lock:
                frame_to_process = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame_to_process is None:
                continue

            # run detection on a smaller copy to keep things fast
            small_frame = cv2.resize(frame_to_process, (320, 240))
            face_crop, face_box = get_face(small_frame, self.face_detector)

            if face_crop is not None and face_box is not None:
                # map box coords back to original resolution
                scale_x = frame_to_process.shape[1] / 320
                scale_y = frame_to_process.shape[0] / 240
                x1, y1, x2, y2 = face_box
                face_box = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))

                label, confidence, engagement_prob = predict(face_crop, self.engagement_model)

                with self.frame_lock:
                    self.latest_result = (label, confidence, engagement_prob, face_box)

    def submit(self, frame):
        with self.frame_lock:
            self.latest_frame = frame
        self.new_frame_event.set()

    def get_result(self):
        with self.frame_lock:
            return self.latest_result


def run():
    # load models and keep yolo on gpu for faster inference

    face_detector, engagement_model = load_models()
    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("webcam not found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector_thread = DetectionThread(face_detector, engagement_model)
    detector_thread.start()

    # make sure snapshot folder exists
    snapshot_dir = os.path.join('outputs', 'previews')
    os.makedirs(snapshot_dir, exist_ok=True)

    # smooth predictions a bit so the score doesn't jump around
    recent_probs = deque(maxlen=12)

    is_paused = False
    frozen_frame = None
    smoothed_fps = 0.0
    previous_tick = time.time()

    # run detection every few frames so the preview stays smooth
    frame_count = 0
    while True:
        if not is_paused:
            got_frame, frame = cap.read()
            if not got_frame:
                break
            frozen_frame = frame.copy()
        else:
            if frozen_frame is None:
                continue
            frame = frozen_frame.copy()

        frame_count += 1
        if (not is_paused) and (frame_count % 5 == 0):
            detector_thread.submit(frame)

        prediction = detector_thread.get_result()
        if prediction is not None:
            label, confidence, engagement_prob, face_box = prediction
            recent_probs.append(engagement_prob)
            smooth_prob = float(np.mean(recent_probs))

            frame = draw_result(frame, label, confidence, smooth_prob, face_box)
            draw_engagement_bar(frame, smooth_prob)
        else:
            # fallback when no face has been picked up
            cv2.putText(frame, "no face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # lightly smooth fps so the number is easier to read
        current_tick = time.time()
        frame_dt = max(1e-6, current_tick - previous_tick)
        instant_fps = 1.0 / frame_dt
        smoothed_fps = instant_fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * instant_fps)
        previous_tick = current_tick

        draw_hud(frame, smoothed_fps, is_paused)

        cv2.imshow('engagement detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            is_paused = not is_paused
        elif key == ord('s'):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            snapshot_path = os.path.join(snapshot_dir, f'preview_{timestamp}.jpg')
            cv2.imwrite(snapshot_path, frame)
            print(f"snapshot saved: {snapshot_path}")

    detector_thread.running = False
    detector_thread.new_frame_event.set()
    detector_thread.join(timeout=1.0)

    # clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()