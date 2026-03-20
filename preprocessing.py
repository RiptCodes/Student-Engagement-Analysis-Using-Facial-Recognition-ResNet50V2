import os
import json
import cv2
import signal
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt

from config import *

BATCH_DETECT = 8


def timeout_handler(signum, frame):
    raise TimeoutError()


def find_video(video_dir, clip_id):
    clean_id = str(clip_id).replace('.avi', '').replace('.mp4', '')
    person = clean_id[:6]

    possible = [
        f"{video_dir}/{person}/{clean_id}/{clean_id}.avi",
        f"{video_dir}/{person}/{clean_id}/{clean_id}.mp4",
        f"{video_dir}/{clean_id}.avi",
    ]

    for p in possible:
        if os.path.exists(p):
            return p
    return None


def show_preview(frames, frame_indices, label_text, clip_id):
    figure, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i in range(min(6, len(frames))):
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        axes[i].imshow(frame)
        axes[i].set_title(f"frame {frame_indices[i]} — {label_text}", fontsize=9)
        axes[i].axis('off')

    for i in range(min(6, len(frames)), len(axes)):
        axes[i].axis('off')

    plt.suptitle(clip_id, fontsize=11)
    plt.tight_layout()

    out_path = f"outputs/previews/{label_text}.png"
    counter = 1
    while os.path.exists(out_path):
        out_path = f"outputs/previews/{label_text}_{counter}.png"
        counter += 1

    plt.savefig(out_path)
    plt.close()


def plot_distribution(train_df):
    counts = train_df['Label'].value_counts().sort_index()
    bar_labels = [LABELS[i] for i in counts.index]

    bars = plt.bar(bar_labels, counts.values)
    max_h = max(counts.values)

    for bar in bars:
        h = bar.get_height()
        if h < max_h:
            plt.text(bar.get_x() + bar.get_width()/2, h + max_h*0.01,
                str(int(h)), ha='center', va='bottom', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width()/2, h/2,
                str(int(h)), ha='center', va='center', color='white', fontsize=9)

    plt.title('Training Set Class Distribution')
    plt.xlabel('Engagement Level')
    plt.ylabel('Videos')
    plt.tight_layout()
    plt.savefig('outputs/plots/engagement_distribution.png')
    plt.close()


class FacePreprocessor:
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size
        self.detector = YOLO('yolov8n-face-lindevs.pt')
        self.detector.to('cuda')

    def extract_face(self, result, frame):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best = int(areas.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best].tolist())

        h, w = frame.shape[:2]
        pad = int(0.3 * (x2 - x1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(crop, self.target_size)

    def process_frame(self, frame):
        result = self.detector(frame, verbose=False, imgsz=640)[0]
        return self.extract_face(result, frame)

    def draw_boxes(self, frame):
        result = self.detector(frame, verbose=False, imgsz=640)[0]
        out = frame.copy()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best = int(areas.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best].tolist())
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return out


def process_video(vpath, preprocessor):
    signal.signal(signal.SIGALRM, timeout_handler)

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None

    raw_frames = []
    try:
        signal.alarm(30)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_STEP == 0:
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                raw_frames.append((idx, frame))
            idx += 1
        signal.alarm(0)
    except TimeoutError:
        cap.release()
        return None
    cap.release()

    if len(raw_frames) == 0:
        return None

    faces = []
    indices = []
    try:
        signal.alarm(60)
        for b in range(0, len(raw_frames), BATCH_DETECT):
            batch = [f for _, f in raw_frames[b:b+BATCH_DETECT]]
            results = preprocessor.detector(batch, verbose=False, imgsz=640)
            for j, res in enumerate(results):
                face = preprocessor.extract_face(res, raw_frames[b+j][1])
                if face is not None:
                    faces.append(face)
                    indices.append(raw_frames[b+j][0])
        signal.alarm(0)
    except TimeoutError:
        return None

    if len(faces) < MIN_FRAMES:
        return None

    return np.array(faces, dtype=np.uint8), indices


def write_tfrecord_entry(writer, face, label):
    # encode the face as jpeg bytes and write directly to the tfrecord
    img_bytes = tf.io.encode_jpeg(face).numpy()
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def process_split(video_dir, labels_df, out_dir, split_name, preprocessor,
    max_per_class=None, preview=True):

    os.makedirs(out_dir, exist_ok=True)

    tfrecord_path = os.path.join(out_dir, f'{split_name}.tfrecord')
    progress_path = os.path.join(out_dir, f'{split_name}_progress.json')

    if max_per_class is not None:
        parts = [labels_df[labels_df['Label'] == cls].head(max_per_class)
            for cls in labels_df['Label'].unique()]
        labels_df = pd.concat(parts).sample(frac=1).reset_index(drop=True)

    processed_ids = set()
    failed = []
    shown = set()
    start_idx = 0

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        processed_ids = set(progress['processed_ids'])
        failed = progress['failed']
        shown = set(progress['shown'])
        start_idx = progress['start_idx']
        print(f"\nResuming {split_name} — {len(processed_ids)} done so far")
    else:
        print(f"\nStarting {split_name} ({len(labels_df)} videos)")

    # open in append mode so resuming adds to existing records
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, (_, row) in enumerate(tqdm(labels_df.iterrows(), total=len(labels_df), initial=start_idx)):
            if i < start_idx:
                continue

            clip_id = str(row['ClipID'])
            label = int(row['Label'])

            if clip_id in processed_ids:
                continue

            vpath = find_video(video_dir, clip_id)
            if vpath is None:
                failed.append(clip_id)
                continue

            result = process_video(vpath, preprocessor)
            if result is None:
                failed.append(clip_id)
                continue

            faces, indices = result

            if preview and label not in shown:
                show_preview(faces, indices, LABELS[label], clip_id)
                shown.add(label)

            # write each face frame from this clip directly into the tfrecord
            for face in faces:
                write_tfrecord_entry(writer, face, label)

            processed_ids.add(clip_id)

            if (i + 1) % 50 == 0:
                with open(progress_path, 'w') as f:
                    json.dump({
                        'start_idx': i + 1,
                        'processed_ids': list(processed_ids),
                        'failed': failed,
                        'shown': list(shown)
                    }, f)
                print(f"  checkpoint {i+1}/{len(labels_df)}")

    all_ids = list(processed_ids)
    all_labels = []
    for _, row in labels_df.iterrows():
        if str(row['ClipID']) in processed_ids:
            all_labels.append(int(row['Label']))

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"{split_name} complete — {len(all_ids)} ok, {len(failed)} failed")
    return all_ids, np.array(all_labels, dtype=np.int64)


def load_or_process(video_dir, labels_df, out_dir, split_name, preprocessor,
    max_per_class=None, preview=True, force=False):

    tfrecord_path = os.path.join(out_dir, f'{split_name}.tfrecord')

    if not force and os.path.exists(tfrecord_path):
        print(f"\nLoading {split_name} labels from csv...")
        all_ids = []
        all_labels = []
        for _, row in labels_df.iterrows():
            all_ids.append(str(row['ClipID']))
            all_labels.append(int(row['Label']))
        print(f"  {len(all_ids)} videos")
        return all_ids, np.array(all_labels, dtype=np.int64)

    return process_split(video_dir, labels_df, out_dir, split_name,
        preprocessor, max_per_class=max_per_class, preview=preview)


def load_labels():
    train_df = pd.read_csv(f'{LABELS_PATH}/TrainLabels.csv')
    val_df = pd.read_csv(f'{LABELS_PATH}/ValidationLabels.csv')
    test_df = pd.read_csv(f'{LABELS_PATH}/TestLabels.csv')

    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.strip()
        # scores 0-1 = not engaged, 2-3 = engaged
        df['Label'] = df['Engagement'].apply(lambda x: 0 if int(x) <= 1 else 1)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df