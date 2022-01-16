import numpy as np
import tensorflow as tf
from i3d import InceptionI3d


official_weights = {}
reader = tf.train.load_checkpoint('../checkpoints/rgb_imagenet/')
dtypes = reader.get_variable_to_dtype_map()
for key in dtypes.keys():
    official_weights[key] = reader.get_tensor(key)

i3d_custom = InceptionI3d(classes=400, include_top=True, weights=None)

custom_official_map = {
    'conv3d/kernel:0':                          'RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w',
    'batch_normalization/beta:0':               'RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/beta',
    'batch_normalization/moving_mean:0':        'RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_mean',
    'batch_normalization/moving_variance:0':    'RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_variance',

    'conv3d_1/kernel:0':                        'RGB/inception_i3d/Conv3d_2b_1x1/conv_3d/w',
    'batch_normalization_1/beta:0':             'RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/beta',
    'batch_normalization_1/moving_mean:0':      'RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/moving_mean',
    'batch_normalization_1/moving_variance:0':  'RGB/inception_i3d/Conv3d_2b_1x1/batch_norm/moving_variance',

    'conv3d_2/kernel:0':                        'RGB/inception_i3d/Conv3d_2c_3x3/conv_3d/w',
    'batch_normalization_2/beta:0':             'RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/beta',
    'batch_normalization_2/moving_mean:0':      'RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/moving_mean',
    'batch_normalization_2/moving_variance:0':  'RGB/inception_i3d/Conv3d_2c_3x3/batch_norm/moving_variance',

    'conv3d_3/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 192, 64)
    'batch_normalization_3/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_3/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_3/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_4/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 192, 96)
    'batch_normalization_4/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_4/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_4/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_5/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 96, 128)
    'batch_normalization_5/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_5/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_5/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_6/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 192, 16)
    'batch_normalization_6/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_6/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_6/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_7/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 16, 32)
    'batch_normalization_7/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_7/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_7/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_8/kernel:0':                        'RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 192, 32)
    'batch_normalization_8/beta:0':             'RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_8/moving_mean:0':      'RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_8/moving_variance:0':  'RGB/inception_i3d/Mixed_3b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_9/kernel:0':                        'RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 256, 128)
    'batch_normalization_9/beta:0':             'RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_9/moving_mean:0':      'RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_9/moving_variance:0':  'RGB/inception_i3d/Mixed_3c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_10/kernel:0':                       'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 256, 128)
    'batch_normalization_10/beta:0':            'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_10/moving_mean:0':     'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_10/moving_variance:0': 'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_11/kernel:0':                       'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 128, 192)
    'batch_normalization_11/beta:0':            'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_11/moving_mean:0':     'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_11/moving_variance:0': 'RGB/inception_i3d/Mixed_3c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_12/kernel:0':                       'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 256, 32)
    'batch_normalization_12/beta:0':            'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_12/moving_mean:0':     'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_12/moving_variance:0': 'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_13/kernel:0':                       'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 32, 96)
    'batch_normalization_13/beta:0':            'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_13/moving_mean:0':     'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_13/moving_variance:0': 'RGB/inception_i3d/Mixed_3c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_14/kernel:0':                       'RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 256, 64)
    'batch_normalization_14/beta:0':            'RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_14/moving_mean:0':     'RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_14/moving_variance:0': 'RGB/inception_i3d/Mixed_3c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_15/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 480, 192)
    'batch_normalization_15/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_15/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_15/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_16/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 480, 96)
    'batch_normalization_16/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_16/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_16/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_17/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 96, 208)
    'batch_normalization_17/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_17/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_17/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_18/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 480, 16)
    'batch_normalization_18/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_18/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_18/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_19/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 16, 48)
    'batch_normalization_19/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_19/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_19/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_20/kernel:0':                       'RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 480, 64)
    'batch_normalization_20/beta:0':            'RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_20/moving_mean:0':     'RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_20/moving_variance:0': 'RGB/inception_i3d/Mixed_4b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_21/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 160)
    'batch_normalization_21/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_21/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_21/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_22/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 112)
    'batch_normalization_22/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_22/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_22/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_23/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 112, 224)
    'batch_normalization_23/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_23/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_23/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_24/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 24)
    'batch_normalization_24/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_24/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_24/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_25/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 24, 64)
    'batch_normalization_25/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_25/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_25/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_26/kernel:0':                       'RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 512, 64)
    'batch_normalization_26/beta:0':            'RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_26/moving_mean:0':     'RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_26/moving_variance:0': 'RGB/inception_i3d/Mixed_4c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_27/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 128)
    'batch_normalization_27/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_27/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_27/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_28/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 128)
    'batch_normalization_28/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_28/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_28/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_29/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 128, 256)
    'batch_normalization_29/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_29/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_29/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_30/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 24)
    'batch_normalization_30/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_30/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_30/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_31/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 24, 64)
    'batch_normalization_31/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_31/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_31/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_32/kernel:0':                       'RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 512, 64)
    'batch_normalization_32/beta:0':            'RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_32/moving_mean:0':     'RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_32/moving_variance:0': 'RGB/inception_i3d/Mixed_4d/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_33/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 112)
    'batch_normalization_33/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_33/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_33/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_34/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 144)
    'batch_normalization_34/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_34/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_34/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_35/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 144, 288)
    'batch_normalization_35/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_35/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_35/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_36/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 512, 32)
    'batch_normalization_36/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_36/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_36/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_37/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 32, 64)
    'batch_normalization_37/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_37/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_37/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_38/kernel:0':                       'RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 512, 64)
    'batch_normalization_38/beta:0':            'RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_38/moving_mean:0':     'RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_38/moving_variance:0': 'RGB/inception_i3d/Mixed_4e/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_39/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 528, 256)
    'batch_normalization_39/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_39/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_39/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_40/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 528, 160)
    'batch_normalization_40/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_40/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_40/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_41/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 160, 320)
    'batch_normalization_41/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_41/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_41/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_42/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 528, 32)
    'batch_normalization_42/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_42/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_42/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_43/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 32, 128)
    'batch_normalization_43/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_43/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_43/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_44/kernel:0':                       'RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 528, 128)
    'batch_normalization_44/beta:0':            'RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_44/moving_mean:0':     'RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_44/moving_variance:0': 'RGB/inception_i3d/Mixed_4f/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_45/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 256)
    'batch_normalization_45/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_45/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_45/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_46/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 160)
    'batch_normalization_46/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_46/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_46/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_47/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 160, 320)
    'batch_normalization_47/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_47/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_47/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_48/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 32)
    'batch_normalization_48/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_48/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_48/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_49/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/conv_3d/w',    # (3, 3, 3, 32, 128)
    'batch_normalization_49/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/beta',
    'batch_normalization_49/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/moving_mean',
    'batch_normalization_49/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_2/Conv3d_0a_3x3/batch_norm/moving_variance',

    'conv3d_50/kernel:0':                       'RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 832, 128)
    'batch_normalization_50/beta:0':            'RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_50/moving_mean:0':     'RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_50/moving_variance:0': 'RGB/inception_i3d/Mixed_5b/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'conv3d_51/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 384)
    'batch_normalization_51/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_51/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_51/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_0/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_52/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 192)
    'batch_normalization_52/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_52/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_52/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_53/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 192, 384)
    'batch_normalization_53/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_53/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_53/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_1/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_54/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/conv_3d/w',    # (1, 1, 1, 832, 48)
    'batch_normalization_54/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/beta',
    'batch_normalization_54/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_mean',
    'batch_normalization_54/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0a_1x1/batch_norm/moving_variance',

    'conv3d_55/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/conv_3d/w',    # (3, 3, 3, 48, 128)
    'batch_normalization_55/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/beta',
    'batch_normalization_55/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_mean',
    'batch_normalization_55/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_2/Conv3d_0b_3x3/batch_norm/moving_variance',

    'conv3d_56/kernel:0':                       'RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/conv_3d/w',    # (1, 1, 1, 832, 128)
    'batch_normalization_56/beta:0':            'RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta',
    'batch_normalization_56/moving_mean:0':     'RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_mean',
    'batch_normalization_56/moving_variance:0': 'RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/moving_variance',

    'dense/kernel:0':                           'RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/w',
    'dense/bias:0':                             'RGB/inception_i3d/Logits/Conv3d_0c_1x1/conv_3d/b'
}

# official_weights = {w.name: wn for w, wn in zip(i3d.weights, i3d.get_weights())}


def copy_weights(layer):
    if 'conv' in layer.name:
        official_weight_name = custom_official_map[layer.weights[0].name]
        w = official_weights[official_weight_name]
        official_weights.pop(official_weight_name)
        layer.set_weights([w])

    if 'batch_norm' in layer.name:
        final_w = []
        for layer_w in layer.weights:
            official_weight_name = custom_official_map[layer_w.name]
            w = official_weights[official_weight_name]
            official_weights.pop(official_weight_name)
            # w = np.reshape(w[0], (w[0].shape[-1], ))
            final_w.append(w)
        layer.set_weights(final_w)

    if 'dense' in layer.name:
        final_w = []
        for layer_w in layer.weights:
            official_weight_name = custom_official_map[layer_w.name]
            w = official_weights[official_weight_name]
            official_weights.pop(official_weight_name)
            if 'kernel' in layer_w.name:
                w = np.squeeze(w[0])
            final_w.append(w)
        layer.set_weights(final_w)


for layer in i3d_custom.layers:
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            copy_weights(l)
    else:
        copy_weights(layer)

i3d_custom.save('imagenet.h5')

for key in official_weights.keys():
    if 'Momentum' not in key:
        print(key)
