"""
This script implements the backbones used in the research including ResNet50, ResNet11, DenseNet, MSMLA50, 3D ResNet11, and GCN.

Author: ctrlovefly
Date: January 21, 2024

"""

from keras.applications import ResNet50
import tensorflow as tf
from keras.layers import *
from keras.models import Model
import numpy as np
from utils import se_convolutional_block, se_identity_block
from utils import cbam_block as cbam
from spektral.layers import GCSConv, GlobalAvgPool
# GCN
def gnn_Net(num_features=5):
    x_input = Input(shape=( num_features,))
    a_input = Input(shape=(None,), sparse=True)
    i_input = Input(shape=(),dtype=tf.int64)
    x=GCSConv(32, activation="relu")([x_input, a_input])
    x=GCSConv(32, activation="relu")([x, a_input])
    x=GCSConv(32, activation="relu")([x, a_input])
    output=GlobalAvgPool()([x, i_input])
    output=Dense(17, activation="softmax")(output)
    model = tf.keras.Model(inputs=(x_input,a_input,i_input), outputs=output)
    return model

# DenseNet
def dense_block(x, growth_rate, num_layers):
    for _ in range(num_layers):
        x1 = BatchNormalization()(x)
        x1 = ReLU()(x1)
        x1 = Conv2D(growth_rate * 4, kernel_size=1, padding='same')(x1)
        x1 = Conv2D(growth_rate, kernel_size=3, padding='same')(x1)
        x = Concatenate()([x, x1])

    return x

def transition_block(x):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, kernel_size=1, padding='same')(x)
    x = AveragePooling2D(pool_size=2, strides=2)(x)
    return x

def densenet(input_shape,growth_rate=12):
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # First dense block
    x = dense_block(x, growth_rate, num_layers=7)
    x = transition_block(x)

    # Second dense block
    x = dense_block(x, growth_rate, num_layers=7)
    x = transition_block(x)

    # Third dense block
    x = dense_block(x, growth_rate, num_layers=7)
    x = GlobalAveragePooling2D()(x)

    # Output layer with 17 neurons
    outputs = Dense(17, activation='softmax')(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# ResNet 11
def resnet_block(x, filters, kernel_size=1, stride=1):
    # Shortcut
    shortcut = x
    
    # First convolution
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second convolution
    x = Conv2D(filters, kernel_size*3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Third convolution
    x = Conv2D(filters*4, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    # print(shortcut.shape[-1])
    # print(filters*4)
    # Shortcut connection
    if stride > 1 :
        shortcut = Conv2D(filters*4, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut and residual
    print(x.shape)
    print(shortcut.shape)
    x = add([x, shortcut])
    x = ReLU()(x)
    
    return x

def resnet11(input_shape):
    x_input = Input(input_shape)
    
    x = Conv2D(kernel_size=3,
               strides=1,
               filters=64,
               padding="same")(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = resnet_block(x, filters=64, stride=2)
    x = resnet_block(x, filters=128, stride=2)
    x = resnet_block(x, filters=256, stride=2)
    x = GlobalAveragePooling2D()(x)
    # Fully connected layer
    outputs = Dense(17, activation='softmax')(x)  
    
    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    return model

# 3D Resnet11
def resnet_block_3D(x, filters, kernel_size=(1,1,1), stride=1):
    # Shortcut
    shortcut = x
    
    # First convolution
    x = Conv3D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second convolution
    x = Conv3D(filters, (3,3,3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Third convolution
    x = Conv3D(filters*4, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection
    if stride > 1 or shortcut.shape[-1] != filters*4:
        shortcut = Conv3D(filters*4, (1,1,1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut and residual
    print(x.shape)
    print(shortcut.shape)
    x = add([x, shortcut])
    x = ReLU()(x)
    
    return x

def resnet11_3D(input_shape):
    x_input = Input(input_shape)
    x = Conv3D(kernel_size=(3,3,3),
               strides=(1,1,1),
               filters=64,
               padding="same")(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
       
    x = resnet_block_3D(x, filters=64, stride=1)
    x = resnet_block_3D(x, filters=128, stride=2)
    x = resnet_block_3D(x, filters=256, stride=2)
    # x = resnet_block_3D(x, filters=512, stride=2)

    x = GlobalAveragePooling3D()(x)
    # Fully connected layer
    outputs = Dense(17, activation='softmax')(x)  
    
    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    return model

# ResNet50
def custom_resnet(input_shape, num_classes=17):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    new_input = tf.keras.layers.Input(shape=input_shape)
    x = base_model(new_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=new_input, outputs=x)
    
    return model

# MSMLA50

# According to: https://github.com/minhokim93/LCZ_MSMLA
def MSMLA50(input_shape, depth):
    # Input stage
    inputs = Input(input_shape)

    # Multi-scale layer
    x0 = Conv2D(16, (5, 5), padding='same', kernel_initializer='he_normal')(inputs)
    x1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x2 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # Fuse to multi-scale features (dim: 64)
    x3 = Concatenate(axis=-1)([x0, x1, x2]) 

    # Multi-Level Attention Layer (Branched Unit)
    in_cbam = cbam(x3, 'xCBAM')
    in_cbam = GlobalAveragePooling2D()(in_cbam)

    # SE-ResBlock 1 (16 filters) in main backbone & MLA (16 filters) in branch
    X = se_convolutional_block(x3, f=3, filters=[depth[0], depth[0], depth[0] * 4], stage=2, block='a', s=1)
    cbam1 = cbam(X, 'cbam1')
    cbam1 = GlobalAveragePooling2D()(cbam1)
    X = se_identity_block(X, 3, [depth[0], depth[0], depth[0] * 4], stage=2, block='b')
    X = se_identity_block(X, 3, [depth[0], depth[0], depth[0] * 4], stage=2, block='c')
  
    # SE-ResBlock 2 (32 filters) in main backbone & MLA (32 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[1], depth[1], depth[1] * 4], stage=3, block='a', s=2)
    cbam2 = cbam(X, 'cbam2')
    cbam2 = GlobalAveragePooling2D()(cbam2)
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='b')
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='c')
    X = se_identity_block(X, 3, [depth[1], depth[1], depth[1] * 4], stage=3, block='d')
   
    # SE-ResBlock 3 (64 filters) in main backbone & MLA (64 filters) in branch
    X = se_convolutional_block(X, f=3, filters=[depth[2], depth[2], depth[2] * 4], stage=4, block='a', s=2)
    cbam3 = cbam(X, 'cbam3')
    cbam3 = GlobalAveragePooling2D()(cbam3)
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='b')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='c')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='d')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='e')
    X = se_identity_block(X, 3, [depth[2], depth[2], depth[2] * 4], stage=4, block='f')

    # Context aggregation to create multi-level attention features (dim: 240)
    X = GlobalAveragePooling2D()(X)
    X = Concatenate(axis=-1)([X, in_cbam, cbam1, cbam2, cbam3])

    # FC layer for LCZ classification
    X = Dense(17, activation='softmax', name='fc' + str(8), kernel_initializer='he_normal')(X)

    print(X.shape)
    print(in_cbam.shape)
    print(cbam1.shape)
    print(cbam2.shape)
    print(cbam3.shape)
    # Create model
    model = Model(inputs=inputs, outputs=X, name='MSMLA-50')
    # ddd
    return model
