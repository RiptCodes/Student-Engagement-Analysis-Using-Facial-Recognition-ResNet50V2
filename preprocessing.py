#imports
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

BATCH_DETECT = 8 # number of frames to process in a batch for face detection


# this is used to prevent the code from getting stuck on a single video
def timeout_handler(signum, frame):
    raise TimeoutError()

# this function is used to try and find video files based on the Clip Id
def find_video(video_dir, clip_id):
    clean_id = str(clip_id).replace('.avi', '').replace('.mp4', '') # removes file extensions
    person_id = clean_id[:6] 

    possible_paths = [
        f"{video_dir}/{person_id}/{clean_id}/{clean_id}.avi",
        f"{video_dir}/{person_id}/{clean_id}/{clean_id}.mp4",
        f"{video_dir}/{clean_id}.avi",
    ]
    # this is to check possible paths in os
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# displays a preview of the first few frames with detected faces
def show_preview(frames, frame_indices, label_text, clip_id):
    figure, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten() 
    # 6 frames for a sample when preview is enabled
    for i in range(min(6, len(frames))):
        frame = frames[i]
        if frame.dtype != np.uint8: # formatting correction if needed
            frame = (frame * 255).astype(np.uint8)
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {frame_indices[i]}: {label_text}", fontsize=9)
        axes[i].axis('off')

    for i in range(min(6, len(frames)), len(axes)):
        axes[i].axis('off')

    plt.suptitle(clip_id, fontsize=11)
    plt.tight_layout()

    out_path = f"outputs/previews/{label_text}.png"
    counter =1
    while os.path.exists(out_path):
        out_path = f"outputs/previews/{label_text}_{counter}.png"
        counter += 1

    plt.savefig(out_path)
    plt.close()

# distribution plot for the dataset - checks class imbalance
def plot_distribution(train_df):
    counts = train_df['Label'].value_counts().sort_index()
    bar_labels = []
    for i in counts.index:
        bar_labels.append(LABELS[i])

    bars = plt.bar(bar_labels, counts.values)
    max_height = max(counts.values)

    for bar in bars:
        height = bar.get_height()
        if height < max_height:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height + max_height * 0.01,
                     str(int(height)),
                     ha='center', va='bottom', fontsize=9)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height / 2,
                     str(int(height)),
                     ha='center', va='center', color='white', fontsize=9)

    plt.title('Training Set Class Distribution')
    plt.xlabel('Engagement Level')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig('outputs/plots/engagement_distribution.png')
    plt.close()

# this is for bulding the tf.data.Datasets object for TFRecord files
class FacePreprocessor:
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size
        self.detector = YOLO('yolov8n-face-lindevs.pt') # model for face detection (specific copy of face yolov8n model)
        self.detector.to('cuda') # enables GPU acceleration just incase 

    def extract_face(self, result, frame):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        # bounding box coords are created
        areas = ((boxes.xyxy[:,2] - boxes.xyxy[:, 0]) * 
                 (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))
        best_index = int(areas.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_index].tolist())

        # 30% padding to the original box
        frame_height, frame_width = frame.shape[:2]
        padding = int(0.3 * (x2 - x1))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame_width, x2 + padding)
        y2 = min(frame_height, y2 + padding)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        # format correction again
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(face_crop, self.target_size)
    #processes a frame and returns the cropped face
    def process_frame(self, frame):
        result = self.detector(frame, verbose=False, imgsz=640)[0] # [0] for first result in the batch
        return self.extract_face(result, frame)
    # draws the box onto the processed frame
    def draw_boxes(self, frame):
        result = self.detector(frame, verbose=False, imgsz=640)[0]
        output = frame.copy()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) *
                     (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))
            best_index = int(areas.argmax())
            x1,y1, x2, y2 = map(int, boxes.xyxy[best_index].tolist())
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output

# extrats frames from the video and faces from the frame 
def process_video(video_path, preprocessor):
    signal.signal(signal.SIGALRM, timeout_handler)

    cap = cv2.VideoCapture(video_path) # reads frames from file
    # error test
    if not cap.isOpened():
        return None

    raw_frames = []

    signal.alarm(30) # timout for reading a frame (error test)
    frame_index = 0
    # cv2 loop used in many pipelines 
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % FRAME_STEP == 0:
            frame_height, frame_width = frame.shape[:2]
            if frame_width > 640:
                scale = 640 / frame_width
                frame = cv2.resize(frame, (640, int(frame_height * scale)))
            raw_frames.append((frame_index, frame))
        frame_index += 1
    signal.alarm(0)
    cap.release() # made sure to release video file to free space

    if len(raw_frames) == 0:
        return None

    faces = []
    face_indices = []

    signal.alarm(60)
    for batch_start in range(0, len(raw_frames), BATCH_DETECT):

        batch_end = batch_start + BATCH_DETECT # end index for batch
        current_batch = raw_frames[batch_start:batch_end] # current frame batch

        batch_frames = [] # empty list for frames batch
        for frame_index, frame in current_batch:
            batch_frames.append(frame)


        results = preprocessor.detector(batch_frames, verbose=False, imgsz=640)


        for result_index, result in enumerate(results):
            actual_index = batch_start + result_index
            original_frame = raw_frames[actual_index][1]
            original_frame_index = raw_frames[actual_index][0]
            face = preprocessor.extract_face(result, original_frame)
    
            if face is not None:
                faces.append(face)
                face_indices.append(original_frame_index)

    signal.alarm(0)

    if len(faces) < MIN_FRAMES:
        return None

    return np.array(faces, dtype=np.uint8), face_indices

# saves the processed data in TFRecord format (https://keras.io/examples/keras_recipes/creating_tfrecords/)
def write_tfrecord_entry(writer, face, label):
    image_bytes = tf.io.encode_jpeg(face).numpy()
    feature = {
        'image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes])),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label]))
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

# main function
def process_split(video_dir, labels_df, out_dir, split_name, preprocessor,
                  max_per_class=None, preview=True):

    os.makedirs(out_dir, exist_ok=True)

    tfrecord_path = os.path.join(out_dir, f'{split_name}.tfrecord')
    progress_path = os.path.join(out_dir, f'{split_name}_progress.json')

    if max_per_class is not None:
        parts = []
        unique_classes = labels_df['Label'].unique()
        for class_label in unique_classes:
            class_rows = labels_df[labels_df['Label'] == class_label].head(max_per_class)
            parts.append(class_rows)
        labels_df = pd.concat(parts).sample(frac=1).reset_index(drop=True) # shuffles dataset

    ######################
    # checkpoint -  every 50 videos saved incase of crash (WSL crashes randomly)
    processed_ids = set()
    failed_clips = []
    shown_labels = set()
    start_index = 0
    ######################

    if os.path.exists(progress_path):
        with open(progress_path) as progress_file:
            progress = json.load(progress_file)
        processed_ids = set(progress['processed_ids'])
        failed_clips = progress['failed']
        shown_labels = set(progress['shown'])
        start_index = progress['start_idx']
        print(f"\n resuming:{split_name} — {len(processed_ids)} ")
    else:
        print(f"\nstarting {split_name} ({len(labels_df)} videos)")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for row_index, (table_index, row) in enumerate(
                tqdm(labels_df.iterrows(), total=len(labels_df),
                     initial=start_index)):

            if row_index < start_index:
                continue

            clip_id = str(row['ClipID'])
            label = int(row['Label'])

            if clip_id in processed_ids:
                continue

            video_path = find_video(video_dir, clip_id)
            if video_path is None:
                failed_clips.append(clip_id)
                continue

            result = process_video(video_path, preprocessor)
            if result is None:
                failed_clips.append(clip_id)
                continue

            faces, face_indices = result

            if preview and label not in shown_labels:
                show_preview(faces, face_indices, LABELS[label], clip_id)
                shown_labels.add(label)

            for face in faces:
                write_tfrecord_entry(writer, face, label)

            processed_ids.add(clip_id)

            if (row_index + 1) % 50 == 0:
                with open(progress_path, 'w') as progress_file:
                    json.dump({
                        'start_idx': row_index + 1,
                        'processed_ids': list(processed_ids),
                        'failed': failed_clips,
                        'shown': list(shown_labels)
                    }, progress_file)
                print(f"  Checkpoint saved: {row_index + 1}/{len(labels_df)}")

    all_ids = list(processed_ids)
    all_labels = []
    for table_index, row in labels_df.iterrows():
        if str(row['ClipID']) in processed_ids:
            all_labels.append(int(row['Label']))

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"{split_name} complete — {len(all_ids)} successful, "
          f"{len(failed_clips)} failed")
    return all_ids, np.array(all_labels, dtype=np.int64)


def load_or_process(video_dir, labels_df, out_dir, split_name, preprocessor,
                    max_per_class=None, preview=True, force=False):

    tfrecord_path = os.path.join(out_dir, f'{split_name}.tfrecord')

    if not force and os.path.exists(tfrecord_path):
        print(f"\nTFRecord already exists for {split_name}.")
        all_ids = []
        all_labels = []
        for table_index, row in labels_df.iterrows():
            all_ids.append(str(row['ClipID']))
            all_labels.append(int(row['Label']))
        print(f"  {len(all_ids)} videos loaded")
        return all_ids, np.array(all_labels, dtype=np.int64)

    return process_split(video_dir, labels_df, out_dir, split_name,
                         preprocessor, max_per_class=max_per_class,
                         preview=preview)


def load_labels():
    train_df = pd.read_csv(f'{LABELS_PATH}/TrainLabels.csv')
    val_df = pd.read_csv(f'{LABELS_PATH}/ValidationLabels.csv')
    test_df = pd.read_csv(f'{LABELS_PATH}/TestLabels.csv')

    for dataframe in [train_df, val_df, test_df]:
        dataframe.columns = dataframe.columns.str.strip()
        # engagement scores 0-1 map to Not Engaged, scores 2-3 map to Engaged
        dataframe['Label'] = (dataframe['Engagement'] >= 2).astype(int)

    print(f"train data: {len(train_df)} ")
    print(f" validation data: {len(val_df)}")
    print(f" test data: {len(test_df)}")
    return train_df, val_df, test_df