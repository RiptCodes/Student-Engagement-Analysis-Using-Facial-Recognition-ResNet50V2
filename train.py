import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Force TensorFlow on CPU (avoids SIGFPE / crashes on some GPUs, e.g. CC 12.x + TF JIT).
# Must hide CUDA before `import tensorflow` or TF initializes the GPU first.
_tf_force_cpu = os.environ.get('TF_FORCE_CPU', '').lower() in ('1', 'true', 'yes')
if _tf_force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

if _tf_force_cpu:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

train_device = '/CPU:0' if not gpus else '/GPU:0'
print(f"TF: {tf.__version__} | GPUs: {len(gpus)} | train on {train_device}"
      + (" (TF_FORCE_CPU=1)" if _tf_force_cpu else ""))


from config import *
from dataset import DataGenerator
from model import build_model, unfreeze_base_layers


model, base = build_model(freeze_base=True)


def train(train_gen, val_gen, weight_dict):
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    version= time.strftime('%Y%m%d_%H%M')
    save_path = os.path.join(PROJECT_PATH, f'model_{version}.keras')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
              monitor='val_accuracy', factor=0.5,
            patience=2, min_lr=1e-7, verbose=1
        )
    ]

    # stage 1 - head only
    print("\nStage 1: ")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

    train_ds = train_gen.as_tf_dataset(shuffle=True)
    val_ds = val_gen.as_tf_dataset(shuffle=False)

    with tf.device(train_device):
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,
            class_weight=weight_dict,
            verbose=1
        )
    # stage 2
    print("\nStage 2: ")
    unfreeze_base_layers(base, n_layers=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    train_ds = train_gen.as_tf_dataset(shuffle=True)
    val_ds= val_gen.as_tf_dataset(shuffle=False)

    with tf.device(train_device):
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=weight_dict,
            verbose=1
        )

    history = {
        'loss':history1.history['loss']+ history2.history['loss'],
        'acc': history1.history['accuracy']+ history2.history['accuracy'], 
        'val_acc': history1.history['val_accuracy']+ history2.history['val_accuracy'], 'stage':[1]*5 + [2]*len(history2.history['loss'])
    }

    print(f'\nModel saved to: {save_path}')
    return history, save_path


def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs= range(1, len(history['loss']) + 1)
    stage1_end = history['stage'].count(1)

    ax1.plot(epochs, history['loss'], label='Train Loss', color='steelblue')
    ax1.axvline(x=stage1_end, color='gray', linestyle='--', label='Stage 1 → 2')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['acc'],label='Train', color='green')
    ax2.plot(epochs, history['val_acc'], label='Val',color='red', linestyle='--')
    ax2.axvline(x=stage1_end, color='gray', linestyle='--', label='Stage 1 → 2')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('outputs/plots/training_curves.png')
    plt.close()


if __name__ == '__main__':
    from preprocessing import load_labels
    import numpy as np

    train_df, val_df, _ = load_labels() # only need train and val labels for training

    train_labels = train_df['Label'].values
    val_labels = val_df['Label'].values

    train_gen = DataGenerator(f'{SAVE_DIR}/train.tfrecord', train_labels)
    val_gen   = DataGenerator(f'{SAVE_DIR}/val.tfrecord', val_labels)

    # weights computed from tfrecord because frame counts differ from video counts
    print("Computing class weights from TFRecord...")
    actual_labels = np.array([
        int(tf.io.parse_single_example(ex, {'label': tf.io.FixedLenFeature([], tf.int64)})['label'].numpy())
        for ex in tf.data.TFRecordDataset(f'{SAVE_DIR}/train.tfrecord')
    ])
    weights = compute_class_weight('balanced', classes=np.unique(actual_labels), y=actual_labels)
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f'Class weights: {weight_dict}')

    history, save_path = train(train_gen, val_gen, weight_dict)
    plot_training(history)