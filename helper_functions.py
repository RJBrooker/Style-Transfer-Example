from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K


def preprocess_image(image_path, img_nrows, img_ncols ):
    img = load_img(image_path, target_size=(img_nrows, img_ncols) )
    img = np.expand_dims( img_to_array(img) , axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(x, img_nrows, img_ncols ):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols)).transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')


## Gram Matrix (Feature-wise Outer Product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram