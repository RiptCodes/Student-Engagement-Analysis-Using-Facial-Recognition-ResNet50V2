import os
import time


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

gpus = tf.config.list_physical_devices('GPU')
# tf
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    

print(f"TF: {tf.__version__} | GPUs: {len(gpus)}")


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

    with tf.device('/GPU:0'):
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
    val_gen = DataGenerator(f'{SAVE_DIR}/val.tfrecord', val_labels)

    weights= compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print(f'Class weights: {weight_dict}')

    history, save_path = train(train_gen, val_gen, weight_dict)
    plot_training(history)