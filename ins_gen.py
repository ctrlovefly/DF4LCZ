###################
#
#segment anything
#生成masks 2024.6.7整理
#auther: ctrlovefly
#github: https://github.com/ctrlovefly/DF4LCZ
#
#####################

import os
HOME='.'
print("HOME:",HOME)
#########dataset#########
SAM_SAM_CHECKPOINT_PATH=f"{HOME}/weights/sam_vit_h_4b8939.pth"
import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

sam = sam_model_registry["vit_h"](checkpoint=SAM_SAM_CHECKPOINT_PATH)
device = "cuda"
sam.to(device=device)
       
def save_to_npz_one(masks,bbox,idx,output_file):
    filename = os.path.join(output_file, f'graph_{idx}')
    np.savez(filename, masks=masks, bbox=bbox)        
        
pred_iou_thresh=[0.88]
points_per_side=12
box_nms_thresh=0.4
crop_n_layers=1
crop_nms_thresh=0.4
output_file='./test_masks/'#输出mask位置
print(pred_iou_thresh)
for iou in pred_iou_thresh:
    mask_generator= SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=iou,
        stability_score_thresh=0.92,
        box_nms_thresh=box_nms_thresh,
        crop_nms_thresh=crop_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    mask_list=[]
    bbox_list=[]
    new_size = (320, 320)
      
    # 使用 os.listdir 获取目录下所有文件和文件夹的列表
    gg_image_patches_path='./google_img_320'
    gg_image_files = os.listdir(gg_image_patches_path)

    # 使用列表推导式过滤出所有的文件（不包括文件夹）
    files_only = [file for file in gg_image_files if os.path.isfile(os.path.join(gg_image_patches_path, file))]

    # 获取文件数量
    num_files = len(files_only)
    
    
    for i in range(num_files):        
        loaded_gg_image=np.load(os.path.join(gg_image_patches_path,f'gg_image_{i}.npz'))
        gg_img_array = loaded_gg_image['image']
        resized_img = gg_img_array
        resized_img = cv2.resize(gg_img_array, new_size, interpolation=cv2.INTER_LINEAR)#cv2.INTER_LINEAR 实际上就是双线性插值（Bilinear Interpolation）

        masks = mask_generator.generate(resized_img) # SAM 
     
        def save_anns(anns): # save masks
            if len(anns) == 0:
                return 
            one_image_mask_list=[]
            one_image_bbox_list=[]
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            index=0
            for ann in sorted_anns:
                if ann['area']<50:
                    continue
                if index>49:
                    break
                one_image_mask_list.append(ann['segmentation'])
                one_image_bbox_list.append(ann['bbox'])
                index=index+1
            print(np.array(one_image_mask_list).shape)
            print(np.array(one_image_bbox_list).shape)
            return np.array(one_image_mask_list),np.array(one_image_bbox_list)
        # print(masks.shape)
        def show_anns(anns): # show masks in .jpg
            print(anns)
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            idx=0
            for ann in sorted_anns:
                if ann['area']<50:
                    continue
                if idx==50:
                    break
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
                idx=idx+1
                
            ax.imshow(img)
            if not os.path.exists(f"{HOME}/ins_test/"):
                # 如果不存在，创建文件夹
                os.makedirs(f"{HOME}/ins_test/")
            plt.savefig(f"{HOME}/ins_test/output_{i}_320.jpg")
            

        plt.figure(figsize=(20,20))
        plt.imshow(resized_img)
        show_anns(masks)
        one_image_mask_list,one_image_bbox_list=save_anns(masks)
        save_to_npz_one(one_image_mask_list,one_image_bbox_list,i,output_file)
        plt.axis('off')
        plt.close('all')
