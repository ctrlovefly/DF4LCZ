####graph construction####
#
#auther: ctrlovefly
#github: https://github.com/ctrlovefly/DF4LCZ
#
#####################
import numpy as np
import time
import cv2
import pickle
from spektral.data import Dataset, DisjointLoader, Graph
import os
from scipy.spatial.distance import cdist
import scipy.sparse

def to_one_hot(label, num_classes):#
    one_hot_label = np.zeros(num_classes)
    one_hot_label[label-1] = 1#文件夹的名称是1-17这里索引需要-1
    return one_hot_label

def xywh_to_center(xywh):
    x, y, w, h = xywh
    center_x = (x + 0.5 * w)/320
    center_y = (y + 0.5 * h)/320#
    return center_x, center_y

def save_graph(n_samples):
    #获取google图片
    gg_path='./google_img_320'
    mask_path='./test_masks'# the outpath in ins_gen.py 
    output_path='./test_gg_output'
    new_size=(320,320)
    for i in range(n_samples):
        # if i >=2120: 
            gg_img = np.load(os.path.join(gg_path, f'gg_image_{i}.npz'))     
            label=to_one_hot(gg_img['label'],17)
            mask_data = np.load(os.path.join(mask_path, f'graph_{i}.npz'))
            # 获取gg中objects特征; Get the features of objects 
            x_g=[]
            coordinates=[]
            for idx,mask_g in enumerate(mask_data['masks']):#提取一个图中的每个mask; Extract each mask in a graph
                # 将像素值缩放到0到1之间
                scaled_g=cv2.resize(gg_img['image'], new_size, interpolation=cv2.INTER_LINEAR)#
                scaled_img = scaled_g/255 #除以/255
                scaled_img[~mask_g,:]=0#将goolge图片的非掩模区域都设为0，以计算淹没区域的平均RGB;Set all non-masked areas of the google image to 0 to calculate the average RGB of the flooded area
                avg_RGB_feature=np.mean(scaled_img[mask_g,:], axis=0)#
                cx,cy=xywh_to_center(mask_data['bbox'][idx])#
                coordinates.append([cx,cy])
                feature_p=np.append(avg_RGB_feature, [cx,cy])
                x_g.append(feature_p)
            coordinates=np.array(coordinates)
            distances = cdist(coordinates, coordinates)
            # 根据阈值生成邻接矩阵; Generate adjacency matrix based on threshold
            a_g = distances
            if a_g.shape==(1,1):# 防止有单个mask整张图的情况；Prevent a single mask from covering the entire image
                adj_sparse_coo = scipy.sparse.coo_matrix(a_g)
            else:
                # 获取非零元素的行、列索引和对应的值
                nonzero_rows, nonzero_cols = np.nonzero(a_g)
                nonzero_values = a_g[nonzero_rows, nonzero_cols]
                # 转化为 COO 稀疏矩阵
                adj_sparse_coo = scipy.sparse.coo_matrix((nonzero_values, (nonzero_rows, nonzero_cols)))
            filename = os.path.join(output_path, f'graph_{i}')        
            np.savez(filename, x=np.array(x_g), a=a_g, e=adj_sparse_coo, y=label)
            print(f'saved_{i}')
            
start_time = time.time() 
save_graph(1) # numbers of graphs 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码运行时间：{elapsed_time} 秒")#2528.8098332881927 s
