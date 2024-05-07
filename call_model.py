"""
This script implements a function to call the different models.

Author: ctrlovefly
Date: January 21, 2024

Based on: https://github.com/minhokim93/LCZ_MSMLA

"""
from keras.models import Model
from keras.layers import *

# Call models

def get_model(model, input_shape, d):
    if model == 'MSMLA50':
        from model import MSMLA50
        net = MSMLA50(input_shape, [d, 2*d, 3*d])
        net.build(input_shape) 
    elif model == 'resnet50':
        from model import custom_resnet
        net = custom_resnet(input_shape)
        net.build(input_shape)
    elif model == 'resnet11':
        from model import resnet11
        net = resnet11(input_shape)
        net.build(input_shape)           
    elif model == 'densenet':
        from model import densenet
        net = densenet(input_shape)
        net.build(input_shape)       
    elif model =='resnet11_3D':
        from model import resnet11_3D
        input_shape=(input_shape[0],input_shape[1],input_shape[2],1)
        net= resnet11_3D(input_shape)
        net.build(input_shape)   
    elif model == 'gnn':
        from model import gnn_Net
        net = gnn_Net()
        # net.build(input_shape)      
    return net