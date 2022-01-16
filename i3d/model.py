import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import utils as keras_utils

from typing import Union, Tuple, List
from collections import namedtuple

from .weights import WEIGHTS_MAP, PRETRAINED_WEIGHTS_URL


BranchConfig = namedtuple('BranchConfig', ('Branch_Name', 'Branch_0_Channels', 'Branch_1_Channels', 'Branch_2_Channels', 'Branch_3_Channels'))
config_list = [
    BranchConfig(Branch_Name='Mixed_3b',
                 Branch_0_Channels=64, Branch_1_Channels=(96, 128), Branch_2_Channels=(16, 32), Branch_3_Channels=32),
    BranchConfig(Branch_Name='Mixed_3c',
                 Branch_0_Channels=128, Branch_1_Channels=(128, 192), Branch_2_Channels=(32, 96), Branch_3_Channels=64),
    BranchConfig(Branch_Name='Mixed_4b',
                 Branch_0_Channels=192, Branch_1_Channels=(96, 208), Branch_2_Channels=(16, 48), Branch_3_Channels=64),
    BranchConfig(Branch_Name='Mixed_4c',
                 Branch_0_Channels=160, Branch_1_Channels=(112, 224), Branch_2_Channels=(24, 64), Branch_3_Channels=64),
    BranchConfig(Branch_Name='Mixed_4d',
                 Branch_0_Channels=128, Branch_1_Channels=(128, 256), Branch_2_Channels=(24, 64), Branch_3_Channels=64),
    BranchConfig(Branch_Name='Mixed_4e',
                 Branch_0_Channels=112, Branch_1_Channels=(144, 288), Branch_2_Channels=(32, 64), Branch_3_Channels=64),
    BranchConfig(Branch_Name='Mixed_4f',
                 Branch_0_Channels=256, Branch_1_Channels=(160, 320), Branch_2_Channels=(32, 128), Branch_3_Channels=128),
    BranchConfig(Branch_Name='Mixed_5b',
                 Branch_0_Channels=256, Branch_1_Channels=(160, 320), Branch_2_Channels=(32, 128), Branch_3_Channels=128),
    BranchConfig(Branch_Name='Mixed_5c',
                 Branch_0_Channels=384, Branch_1_Channels=(192, 384), Branch_2_Channels=(48, 128), Branch_3_Channels=128),
]


def conv_block(inputs,
               output_channels: int,
               kernel_size: Union[Tuple, int],
               name: str,
               use_bias: bool = False,
               activation: Union[str, None] = 'relu',
               strides: Union[Tuple, int] = (1, 1, 1),
               bn: Union[layers.Layer, None] = layers.BatchNormalization,
               ):
    with tf.name_scope(name=name):
        x = layers.Conv3D(output_channels,
                          kernel_size=kernel_size,
                          strides=strides,
                          use_bias=use_bias,
                          padding='same')(inputs)
        if bn is not None:
            x = bn(scale=False)(x)
        if activation is not None:
            x = layers.Activation(activation=activation)(x)
    return x


def inception_block(inputs,
                    branch_0_channels=64,
                    branch_1_channels=(96, 128),
                    branch_2_channels=(16, 32),
                    branch_3_channels=32,
                    activation='relu',
                    block_name='Mixed_3b',
                    ):

    input_x = layers.Input(shape=inputs.shape[1:])

    # Branch 0
    with tf.name_scope('Branch_0'):
        branch_0 = conv_block(input_x, branch_0_channels, kernel_size=1, activation=activation, name='Conv3d_0a_1x1')

    # Branch 1
    with tf.name_scope('Branch_1'):
        branch_1 = conv_block(input_x, branch_1_channels[0], kernel_size=1, activation=activation, name='Conv3d_0a_1x1')
        branch_1 = conv_block(branch_1, branch_1_channels[1], kernel_size=3, activation=activation, name='Conv3d_0b_3x3')

    # Branch 2
    with tf.name_scope('Branch_2'):
        branch_2 = conv_block(input_x, branch_2_channels[0], kernel_size=1, activation=activation, name='Conv3d_0a_1x1')
        branch_2 = conv_block(branch_2, branch_2_channels[1], kernel_size=3, activation=activation, name='Conv3d_0b_3x3')

    # Branch 3
    with tf.name_scope('Branch_3'):
        branch_3 = layers.MaxPooling3D(pool_size=3, strides=1, padding='same', name='MaxPool3d_0a_3x3')(input_x)
        branch_3 = conv_block(branch_3, branch_3_channels, kernel_size=1, activation=activation, name='Conv3d_0b_1x1')

    output = layers.Concatenate()([branch_0, branch_1, branch_2, branch_3])
    _model = Model(inputs=input_x, outputs=output, name=block_name)

    return _model(inputs)


def InceptionI3d(input_shape: Tuple = (16, 224, 224, 3),
                 classes: int = 400,
                 activation: str = 'relu',
                 configs: List[BranchConfig] = config_list,
                 drop_out=0.8,
                 include_top=True,
                 input_tensor=None,
                 model_name: str = 'InceptionI3d',
                 weights='kinetics-400',
                 ):

    if len(input_shape) != 4:
        raise ValueError('length of input_shape must be 4 to represent a 3D image.')
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = conv_block(img_input, 64, 7,
                   activation=activation,
                   strides=2,
                   name='Conv3d_1a_7x7'
                   )

    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool3d_2a_3x3')(x)

    x = conv_block(x, 64, 1, activation=activation, name='Conv3d_2b_1x1')
    x = conv_block(x, 192, 3, activation=activation, name='Conv3d_2c_3x3')

    x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool3d_3a_3x3')(x)

    branch_id = '3'
    for branch_config in configs:

        if branch_id != branch_config.Branch_Name.split('_')[1][0]:
            branch_id = branch_config.Branch_Name.split('_')[1][0]
            x = layers.MaxPooling3D(pool_size=7 - int(branch_id),
                                    strides=2,
                                    padding='same',
                                    name=f'MaxPool3d_{branch_id}a_3x3')(x)

        x = inception_block(x,
                            branch_0_channels=branch_config.Branch_0_Channels,
                            branch_1_channels=branch_config.Branch_1_Channels,
                            branch_2_channels=branch_config.Branch_2_Channels,
                            branch_3_channels=branch_config.Branch_3_Channels,
                            block_name=branch_config.Branch_Name,
                            activation=activation
                            )

    if include_top:
        with tf.name_scope('Logits'):
            x = layers.GlobalAvgPool3D()(x)
            if drop_out is not None and drop_out > 0:
                x = layers.Dropout(drop_out)(x)
            x = layers.Dense(classes, use_bias=True, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x, name=model_name)

    if weights in ['kinetics-400', 'kinetics-600', 'imagenet']:
        file_name = WEIGHTS_MAP[weights][include_top]
        weight_path = keras_utils.get_file(
            file_name,
            PRETRAINED_WEIGHTS_URL + file_name,
            cache_subdir='models'
        )
        model.load_weights(weight_path)
    elif weights:
        model.load_weights(weights)

    return model




