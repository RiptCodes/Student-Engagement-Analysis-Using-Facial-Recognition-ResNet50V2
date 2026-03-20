# imports required for data loading and processing
import tensorflow as tf
import numpy as np
from config import * 

# build the dataset from the tfrecord files created 
def build_dataset(tfrecord_path, batch_size=BATCH, shuffle=True, augment=True):
    
    def parse_record(example):
        # pulls the image bytes and labels out of each tfrecord entry
        feature_map = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        # creates a dict of tensors with keys 'image' and 'label' from the raw tfrecord data 
        # labels become integers and images are JPEG byte strings
        parsed = tf.io.parse_single_example(example, feature_map) 
        img = tf.io.decode_jpeg(parsed['image'], channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
        
        return img, parsed['label']

    def random_augment(img, label):
        # randomly flip and adjust brightness/contrast to help generalisation
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img, label

    # load the tfrecord file
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000) # this has a large buffer to ensure good shuffling without loading everything into memory

    # decode each record into an image and label
    # AUTOTUNE allows tf to decide how many parallel calls to make based on available resources
    # this is important for performance when loading and processing data on the fly during training
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # group into batches
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # normalise pixels for resnet and one-hot encode the labels
    def preprocess_batch(images, labels):
        images = tf.keras.applications.resnet_v2.preprocess_input(images)
        labels = tf.one_hot(labels, N_CLASSES)
        return images, labels

    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    # prefetch so gpu doesnt sit idle waiting for data
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# this class counts the number of samples in the tfrecord file and provides a method to
# get the dataset as a tf.data.Dataset object ready for training or evaluation.
class DataGenerator:
    def __init__(self, tfrecord_path, labels, batch_size=BATCH):
        self.tfrecord_path = tfrecord_path
        self.batch_size    = batch_size

        # count samples
        self.num_samples = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
        self.n_batches   = max(1, self.num_samples // batch_size)
        print(f"  {self.num_samples} samples, {self.n_batches} batches")

    def as_tf_dataset(self, shuffle=True):
        augment = shuffle  # only augment during training (augments are random affects basically)
        return build_dataset(self.tfrecord_path, self.batch_size, shuffle, augment)