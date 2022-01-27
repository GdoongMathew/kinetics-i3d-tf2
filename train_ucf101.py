import tensorflow as tf
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from i3d import InceptionI3d
from preprocess_video import create_augment_layers
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from tensorflow.keras import callbacks
from datetime import datetime

import tensorflow_datasets as tfds
import numpy as np

import os
import math


FRAMES = 16
IMAGE_SIZE = 256
EPOCHS = 2000
BATCH_SIZE = 32
AUGMENTATION = True
AUTOTUNE = tf.data.AUTOTUNE


def select_frames(frames=16, gap=0):

    @tf.function(jit_compile=True)
    def _func(y, x):
        x_frame = tf.shape(x)[0]
        start_frame = tf.random.uniform(shape=(), minval=0, maxval=x_frame - frames, dtype=tf.int32)

        # todo implement gap mechanism
        x = x[start_frame: start_frame + frames]
        tf.assert_equal(tf.shape(x)[0], frames)
        return x, y

    return _func


def video_transpose_reshape(video):
    video = tf.transpose(video, perm=[1, 2, 0, 3])
    video_shape = tf.shape(video)
    video = tf.reshape(video, [video_shape[0], video_shape[1], -1])
    return video


def return_video_dims(video):
    video_shape = tf.shape(video)
    video = tf.reshape(video, [video_shape[0], video_shape[1], -1, 3])
    video = tf.transpose(video, perm=[2, 0, 1, 3])

    return video


data = tfds.load('ucf101', split=['train', 'test'], shuffle_files=True, with_info=True, as_supervised=False)
num_classes = data[1].features['label'].num_classes
train_data, val_data = data[0]

assert isinstance(train_data, tf.data.Dataset) and isinstance(val_data, tf.data.Dataset)

train_data = train_data.map(lambda _x: select_frames(frames=FRAMES)(*_x.values()), num_parallel_calls=AUTOTUNE)
if AUGMENTATION:
    aug_func = create_augment_layers()
    train_data = train_data.map(lambda x, y: (video_transpose_reshape(x), y), num_parallel_calls=AUTOTUNE).\
        map(lambda x, y: (aug_func(x, training=True), y), num_parallel_calls=AUTOTUNE).\
        map(lambda x, y: (return_video_dims(x), y), num_parallel_calls=AUTOTUNE)

train_data = train_data.batch(batch_size=BATCH_SIZE).\
    prefetch(buffer_size=AUTOTUNE)


val_data = val_data.map(lambda _x: select_frames(frames=FRAMES)(*_x.values()), num_parallel_calls=AUTOTUNE).\
    batch(batch_size=BATCH_SIZE).\
    prefetch(buffer_size=AUTOTUNE)

input_shape = (FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = InceptionI3d(input_shape=input_shape, weights='kinetics-600', classes=600, include_top=False)

model = Sequential([
    layers.Input(shape=input_shape),
    base_model,
    layers.GlobalAvgPool3D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax', dtype=tf.float32)
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

today_str = datetime.today().strftime('%Y_%m_%d')

log_dir = os.path.join('logs', today_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def cyclical_step_decay(initial_lr, cycle_step=30, min_lr=1e-8, max_epochs=3000,
                        rate_decay=0.8, policy='exp_range', multiplier=False):
    """
    implementation of CLR
    :param initial_lr:
    :param cycle_step:
    :param min_lr:
    :param max_epochs:
    :param rate_decay:
    :param policy:
    :param multiplier:
    :return:
    """
    if policy not in ['exp_range', 'triangular', 'triangular_2']:
        raise ValueError('Not supported decay policy.')

    def _rate_sch(epoch):
        current_iter = np.floor(1 + epoch / (cycle_step * 2))
        x = np.abs(epoch / cycle_step - 2 * current_iter + 1)
        if policy == 'exp_range':
            max_lr = min_lr + initial_lr * math.pow(1.0 - epoch / max_epochs, rate_decay)
            lr = min_lr + (max_lr - min_lr) * np.maximum(0, x)

        elif policy == 'triangular':
            lr = min_lr + (initial_lr - min_lr) * np.maximum(0, x)

        else:
            lr = min_lr + (initial_lr - min_lr) * np.maximum(0, x / math.pow(2, current_iter))

        lr = max(lr, min_lr)

        return lr if not multiplier else lr / initial_lr

    return _rate_sch


callbacks = [
    callbacks.TensorBoard(log_dir=log_dir, profile_batch=(5, 10)),
    callbacks.LearningRateScheduler(cyclical_step_decay(initial_lr=5e-3,
                                                        cycle_step=200,
                                                        min_lr=1e-9,
                                                        max_epochs=EPOCHS,
                                                        rate_decay=0.98,
                                                        )),
    callbacks.ModelCheckpoint(filepath=f'weights/{base_model.name}_{today_str}.h5',
                              save_best_only=True,
                              save_weights_only=False)
]

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)
