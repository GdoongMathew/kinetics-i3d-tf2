from i3d import InceptionI3d
import tensorflow_hub as hub
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

i3d = hub.KerasLayer('https://tfhub.dev/deepmind/i3d-kinetics-400/1', trainable=False)

i3d_custom = InceptionI3d(classes=400, weights='k400.h5')

test_data = np.random.random((1, 16, 224, 224, 3))
ori_result = softmax(i3d(test_data))
custom_result = i3d_custom(test_data)

print(ori_result[0][:10])
print(custom_result[0][:10])
# print(ori_result == custom_result)

