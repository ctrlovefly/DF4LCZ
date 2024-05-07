"""
This script implements the custom data generator classes designed for loading data in batches.

Author: ctrlovefly
Date: January 21, 2024

"""

from keras.utils.data_utils import Sequence
import numpy as np
import pickle
import os
from spektral.data import Dataset,Graph
 
def to_one_hot(label, num_classes):
    one_hot_label = np.zeros(num_classes)
    one_hot_label[label-1] = 1#The range of label is 1-17.
    return one_hot_label

class MyGenerator_fix_augment(Sequence):
    def __init__(self, dataset_name, batch_size=64, augmentations=None, shuffle=True):
        # Data source
        self.sen2_path='./sen2_img_patches'
        with open('./patches_split/partition_random.npz', 'rb') as f:
            loaded_indexes = pickle.load(f)
        self.indexes = loaded_indexes[dataset_name]       
        self.batch_size = batch_size
        self.augment = augmentations
        self.shuffle = shuffle
        self.aug_flag = dataset_name
        self.on_epoch_end()       
        self.n = 0
        self.max = self.__len__()
        
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def __getitem__(self, idx):
        inds = sorted(self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = []
        batch_y = []
        for item in inds:
            sen2_img = np.load(os.path.join(self.sen2_path,f'sen2_image_{item}.npz')) 
            batch_y.append(to_one_hot(sen2_img['label'],17))
            if self.aug_flag=='validation' or self.aug_flag=='test':
                batch_x.append(sen2_img['image'])
            else:
                batch_x.append(self.augment(image=sen2_img['image'])["image"])
                
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y) 

        return batch_x, batch_y
    
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result 

class MyDataset_simplified(Dataset):
    def __init__(self, **kwargs):
        self.path = './gg_nodes_refine' 
        super().__init__(**kwargs)      
    def download(self):
        pass
    def read(self):
        with open('./patches_split/partition_random.npz', 'rb') as f:
            loaded_indexes = pickle.load(f)
        output = []
        for i in loaded_indexes['train']:
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'))
            binary_adjacency = convert_to_binary_adjacency(data['a'], 0)
            x_mod=data['x']
            output.append(
                Graph(x=x_mod, a=binary_adjacency, y=data['y'])
            )
        return output 
    
class MyDataset_simplified_val(Dataset):
    def __init__(self, **kwargs):
        self.path = './gg_nodes_refine'                  
        super().__init__(**kwargs)      
    def download(self):
        pass
    def read(self):
        with open('./patches_split/partition_random.npz', 'rb') as f:
            loaded_indexes = pickle.load(f)
        output = []
        for i in loaded_indexes['validation']:
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'))
            binary_adjacency = convert_to_binary_adjacency(data['a'], 0)
            x_mod=data['x']
            output.append(
                Graph(x=x_mod, a=binary_adjacency, y=data['y'])
            )
        return output 
    
def convert_to_binary_adjacency(adjacency_matrix, threshold):
    binary_adjacency = np.where(adjacency_matrix > threshold, 1, 0)
    return binary_adjacency

class MyDataset_simplified_test(Dataset):
    def __init__(self, **kwargs):
        self.path = './gg_nodes_refine' 
        super().__init__(**kwargs)      
    def download(self):
        pass
    def read(self):
        with open('./patches_split/partition_random.npz', 'rb') as f:
            loaded_indexes = pickle.load(f)
        output = []
        for i in loaded_indexes['test']:
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'))
            binary_adjacency = convert_to_binary_adjacency(data['a'], 0)
            x_mod=data['x']
            output.append(
                Graph(x=x_mod, a=binary_adjacency, y=data['y'])
            )
        return output 