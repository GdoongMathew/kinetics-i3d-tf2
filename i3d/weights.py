PRETRAINED_WEIGHTS_URL = 'https://github.com/GdoongMathew/kinetics-i3d-tf2/releases/download/v0.0.1/'

WEIGHTS_MAP = {
    'kinetics-400': {
        True: 'kinetics-400.h5',
        False: 'kinetics-400_no_top.h5'
    },
    'kinetics-600': {
        True: 'kinetics-600.h5',
        False: 'kinetics-600_no_top.h5'
    },
    'imagenet': {
        True: 'imagenet.h5',
        False: 'imagenet_no_top.h5',
    },
}