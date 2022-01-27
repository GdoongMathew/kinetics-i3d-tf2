from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras import Sequential


def create_augment_layers():
    aug_seq = Sequential([
        RandomZoom(height_factor=0.2),
        RandomTranslation(0.1, 0.1, fill_mode='constant'),
        RandomFlip('horizontal'),
        RandomRotation(factor=0.08),
        RandomContrast(factor=0.1),
    ])
    return aug_seq
