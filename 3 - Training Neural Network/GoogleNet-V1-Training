#!/home/wli/env python3

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
from skimage.filters import threshold_otsu
import glob
#before importing HDFStore, make sure 'tables' is installed by pip3 install tables
from pandas import HDFStore
from openslide.deepzoom import DeepZoomGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import cv2 as cv2
from skimage import io
import math
from keras.utils.np_utils import to_categorical

training_patches_tumor = pd.DataFrame(columns=['patch_path', 'is_tumor'])
tumor_patch_folder = '/home/wli/Downloads/test/augtumor2'
tumor_patch_paths = glob.glob(osp.join(tumor_patch_folder, '*.png'))
tumor_patch = pd.Series(tumor_patch_paths)
training_patches_tumor['patch_path'] = tumor_patch.values
training_patches_tumor['is_tumor'] = 1

training_patches_normal = pd.DataFrame(columns=['patch_path', 'is_tumor'])
normal_patch_folder_i = '/home/wli/Downloads/test/augnormal'
normal_patch_paths_i = glob.glob(osp.join(normal_patch_folder_i, '*.png'))
normal_patch_folder_ii = '/home/li-computer1/augnormal'
normal_patch_paths_ii = glob.glob(osp.join(normal_patch_folder_ii, '*.png'))
normal_patch_paths = normal_patch_paths_i + normal_patch_paths_ii
normal_patch = pd.Series(normal_patch_paths)
training_patches_normal['patch_path'] = normal_patch.values
training_patches_normal['is_tumor'] = 0

training_patches = pd.concat([training_patches_tumor, training_patches_normal])

validation_patches_tumor = pd.DataFrame(columns=['patch_path', 'is_tumor'])
vtumor_patch_folder = '/home/wli/Downloads/test/validation/augtumor'
vtumor_patch_paths = glob.glob(osp.join(vtumor_patch_folder, '*.png'))
vtumor_patch = pd.Series(vtumor_patch_paths)
validation_patches_tumor['patch_path'] = vtumor_patch.values
validation_patches_tumor['is_tumor'] = 1

validation_patches_normal = pd.DataFrame(columns=['patch_path', 'is_tumor'])
vnormal_patch_folder_i = '/home/wli/Downloads/test/validation/augnormal'
vnormal_patch_paths_i = glob.glob(osp.join(vnormal_patch_folder_i, '*.png'))
vnormal_patch_folder_ii = '/home/li-computer1/augnormal_validation'
vnormal_patch_paths_ii = glob.glob(osp.join(vnormal_patch_folder_ii, '*.png'))
vnormal_patch_paths = vnormal_patch_paths_i + vnormal_patch_paths_ii
vnormal_patch = pd.Series(vnormal_patch_paths)
validation_patches_normal['patch_path'] = vnormal_patch.values
validation_patches_normal['is_tumor'] = 0
validation_patches = pd.concat([validation_patches_tumor, validation_patches_normal])

def gen_imgs(samples, batch_size, shuffle=True):

    num_samples = len(samples)
    while 1:
        if shuffle:
            samples = samples.sample(frac=1) # if frac = 1 will organized list randomly

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            images = []
            labels = []
            for _, batch_sample in batch_samples.iterrows():
                img = io.imread(batch_sample.patch_path)
                img = img[:,:,:3]
                label = batch_sample['is_tumor']

                images.append(np.array(img))
                labels.append(label)

            X_train = np.array(images)
            y_train = np.array(labels)
            y_train = to_categorical(y_train, num_classes=2)

            yield X_train, y_train

# -*- coding: utf-8 -*-
"""Inception V1 model for Keras.
Note that the input preprocessing function is different from the the VGG16 and ResNet models (same as Xception).
Also that (currently) the output predictions are for 1001 classes (with the 0 class being 'background'), 
so require a shift compared to the other models here.
# Reference
- [Going deeper with convolutions](http://arxiv.org/abs/1409.4842v1)
"""
import warnings
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing import image
from keras.regularizers import l2
WEIGHTS_PATH = ''
WEIGHTS_PATH_NO_TOP = ''
# conv2d_bn is similar to (but updated from) inception_v3 version
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer= False,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, (num_row, num_col),
            strides=strides, padding=padding,
            use_bias=False, name=conv_name, kernel_regularizer=l2(0.0005))(x)
    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x
    
# Convenience function for 'standard' Inception concatenated blocks
def concatenated_block(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    
    branch_0 = conv2d_bn(x, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")
    branch_1 = conv2d_bn(x, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")
    branch_2 = conv2d_bn(x, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")
    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3_a_max")(x)  
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")
    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x
def InceptionV1(include_top=True,
                weights= None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=2):
    """Instantiates the Inception v1 architecture.
    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/abs/1409.4842v1
    
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    if weights == 'imagenet' and include_top and classes != 1001:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1001')
    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        #default_size=299,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    # 'Sequential bit at start'
    x = img_input
    x = conv2d_bn(x,  64, 7, 7, strides=(2, 2), padding='same',  name='Conv2d_1a_7x7')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_2a_3x3')(x)  
    
    x = conv2d_bn(x,  64, 1, 1, strides=(1, 1), padding='same', name='Conv2d_2b_1x1')  
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='same', name='Conv2d_2c_3x3')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_3a_3x3')(x)  
    
    # Now the '3' level inception units
    x = concatenated_block(x, (( 64,), ( 96,128), (16, 32), ( 32,)), channel_axis, 'Mixed_3b')
    x = concatenated_block(x, ((128,), (128,192), (32, 96), ( 64,)), channel_axis, 'Mixed_3c')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_4a_3x3')(x)  
    # Now the '4' level inception units
    x = concatenated_block(x, ((192,), ( 96,208), (16, 48), ( 64,)), channel_axis, 'Mixed_4b')
    x = concatenated_block(x, ((160,), (112,224), (24, 64), ( 64,)), channel_axis, 'Mixed_4c')
    x = concatenated_block(x, ((128,), (128,256), (24, 64), ( 64,)), channel_axis, 'Mixed_4d')
    x = concatenated_block(x, ((112,), (144,288), (32, 64), ( 64,)), channel_axis, 'Mixed_4e')
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_4f')
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='MaxPool_5a_2x2')(x)  
    # Now the '5' level inception units
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_5b')
    x = concatenated_block(x, ((384,), (192,384), (48,128), (128,)), channel_axis, 'Mixed_5c')
    
    if include_top:
        # Classification block
        
        # 'AvgPool_0a_7x7'
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  
        
        # 'Dropout_0b'
        x = Dropout(0.5)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction
        
        #logits = conv2d_bn(x,  classes+1, 1, 1, strides=(1, 1), padding='valid', name='Logits',
        #                   normalizer=False, activation=None, )  
        
        # Write out the logits explictly, since it is pretty different
        x = Conv2D(classes, (1, 1), strides=(1,1), padding='valid', use_bias=True, name='Logits')(x)
        
        x = Flatten(name='Logits_flat')(x)
        #x = x[:, 1:]  # ??Shift up so that first class ('blank background') vanishes
        # Would be more efficient to strip off position[0] from the weights+bias terms directly in 'Logits'
        
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(    name='global_pooling')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Finally : Create model
    model = Model(inputs, x, name='inception_v1')
    
    # LOAD model weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v1_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='723bf2f662a5c07db50d28c8d35b626d')
        else:
            weights_path = get_file(
                'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='6fa8ecdc5f6c402a59909437f0f5c975')
        model.load_weights('~/Downloads/model0912googlenet.h5')
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)    
    
    return model




model =  InceptionV1()

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="\home\wli\Downloads\googlenet0917-{epoch:02d}-{val_acc:.2f}.hdf5"


model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1, save_best_only=True)

BATCH_SIZE = 32
N_EPOCHS = 5

from datetime import datetime

train_generator = gen_imgs(training_patches, BATCH_SIZE)
validation_generator = gen_imgs(validation_patches, BATCH_SIZE)

#train neural network

train_start_time = datetime.now()


def step_decay_schedule(initial_lr, decay_factor, step_size):   
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=0.01, decay_factor=0.5, step_size=1)


history=model.fit_generator(train_generator, np.ceil(len(training_patches) / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_patches) / BATCH_SIZE),
                            epochs=N_EPOCHS, callbacks=[model_checkpoint, lr_sched])

train_end_time = datetime.now()
print("Model training time: %.1f minutes" % ((train_end_time - train_start_time).seconds / 60,))

model.save('model0912googlenet.h5')

model_json = model.to_json()
with open("modelgooglenet0912.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("googlenetweight.h5")
print("Saving model to disk")
#  "Accuracy"
fig1 = plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig1.savefig('./Accuracy_plot_googlenet.png')
#plt.imsave('accuracy_plot_googlenet', accu)
plt.close(fig1)
# "Loss"
fig2 = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt2.savefig('./Loss_plot_googlenet.png')
#plt.imsave('Loss_plot_googlenet',loss)
#np.save('history_googlenet', history.history)
#import json
# Get the dictionary containing each metric and the loss for each epoch
#history_dict = history.history
# Save it under the form of a json file
#json.dump(history_dict, open('./', 'w'))
#print(history_dict['loss'][49])
