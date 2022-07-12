import os,json
import pandas as pd
from pycocotools.coco import COCO
import urllib.request
import skimage.io as io
import pathlib
import pylab
import cv2
from skimage.io import imread, imshow
import imutils
import numpy as np
import math
import copy
from PIL import Image
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline

def scaled_image (original_json, destination, new_width, new_height) :
    
    annFile= original_json
    coco=COCO(annFile)

    rotated_data_1 = {}

    images = []
    nonrotate_annos = []
    nonrotate_pts = []
    rotate_annos = []

    rotated_data_1['infos'] = coco.dataset['infos']

    imgIds = coco.getImgIds()

    destination_path_scaled_images = destination+'/scaled_images'
    destination_path_scaled_annotations = destination+'/scaled_annotations'
    destination_path_scaled_debug_images  = destination+'/display_both_images'
    
    pathlib.Path(destination_path_scaled_images).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(destination_path_scaled_annotations).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(destination_path_scaled_debug_images).mkdir(parents=True, exist_ok=True) 
    

    for i in range(len(imgIds)):

        img = coco.loadImgs(imgIds[i])[0]
        IMAGE = imread(img['file_name'])

        #print(img['file_name'])

        images.append(img)

        rotated_data_1['images'] = images

        catIds=[]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns_segment = coco.loadAnns(annIds)
        
        hgt = IMAGE.shape[0]
        wid = IMAGE.shape[1]
        
        x_scale = new_width / wid
        y_scale = new_height / hgt
    
        if (hgt*wid) > (x_scale*y_scale):
            flag = cv2.INTER_AREA
        else:
            flag = cv2.INTER_LANCZOS4

        dim = (new_width, new_height)
        
        scale_IMAGE = cv2.resize(IMAGE, dim, interpolation=flag)
        
        debug_image = scale_IMAGE.copy()
        
        for id, anno in enumerate(anns_segment):
            
            if anno['image_id']== imgIds[i]:            
            
                cat_id = []
                bbox = []

    #------------------- Create BBOX ---------------------------------------------------------

                cat_id = int(anno["category_id"])
                r_bbox = anno["bbox"] # [x,y,w,h] object positioning frame
            
                #print('Before: ', list(r_bbox))

                bbx_xmin = r_bbox[0]
                bbx_ymin = r_bbox[1]
                bbx_xmax = r_bbox[2]
                bbx_ymax = r_bbox[3]

                (origLeft, origTop, origRight, origBottom) = (bbx_xmin, bbx_ymin, bbx_xmax, bbx_ymax)

                rect_x = int(np.round(origLeft * x_scale))
                rect_y = int(np.round(origTop * y_scale))
                rect_xmax = int(np.round(origRight * x_scale))
                rect_ymax = int(np.round(origBottom * y_scale))

                xmin = rect_x
                ymin = rect_y
                xmax1 = xmin + rect_xmax
                ymax1 = ymin + rect_ymax

                start_point = (xmin, ymin) 
                end_point = (xmax1, ymax1) 
                color = (0, 255, 0) 
                thickness = 2

                scale_rect = cv2.rectangle(debug_image, start_point, end_point, color, thickness) 
                #imshow(scale_rect)

    #------------------------------ Create Polygon ----------------------------------------

                (h1, w1) = IMAGE.shape[:2]
                (h, w) = scale_IMAGE.shape[:2]

                (cX1, cY1) = (w1 // 2, h1 // 2)
                (cX, cY) = (w // 2, h // 2)

                image= []
                r_axis = []
                
                segmnt_val = anno["segmentation"]
                
                #print('First: ', segmnt_val)

                for val in segmnt_val:

                    r_axis = val   

                    for k in range(0, len(r_axis), 2):

                        temp_point = r_axis[k] - cX1, r_axis[k+1] - cY1
                        temp_point = (temp_point[0]*x_scale, temp_point[1]*y_scale)
                        temp_point = temp_point[0]+cX, temp_point[1]+cY

                        r_axis[k] =  int(temp_point[0])
                        r_axis[k+1] = int(temp_point[1])

                        pts = np.array(r_axis, np.int32)
                        pts = pts.reshape((-1, 1, 2))

                    isClosed = True
                    color = (255, 0, 0)
                    thickness = 3

                    scale_segment = cv2.polylines(scale_rect, pts, isClosed, color, thickness)  
                    
                    segments = []
                    segments.append(r_axis)
                    
                    scaled_box = (rect_x, rect_y, rect_xmax, rect_ymax)
                    #print('After: ', list(scaled_box))
                    
                    rotate_annos.append({"segmentation" : segments,
                            "area" : rect_xmax*rect_ymax,
                            "bbox" : list(scaled_box),
                            "iscrowd" : anno['iscrowd'],
                            "id": anno['id'],
                            "image_id" : anno['image_id'],
                            "category_id" : anno['category_id']})
        
        head, tail = os.path.split(img['file_name'])
        image_name = tail 
        
        file_name = destination_path_scaled_images+'/'+image_name
        plt.imsave(file_name, scale_IMAGE)
        

        ori_pil =Image.open(img['file_name']) # original pic
        scale_pil =Image.open(file_name)  # scaled pic
        mask_pil = Image.fromarray(scale_segment) # masked pic
        
        new_image = Image.new('RGB',(3*max(ori_pil.size[0], scale_pil.size[0], mask_pil.size[0]),max(ori_pil.size[1],scale_pil.size[0], mask_pil.size[1])), (250,250,250))
        new_image.paste(ori_pil,(0,0))
        new_image.paste(scale_pil,(ori_pil.size[0],0))
        new_image.paste(mask_pil,(scale_pil.size[0]+ori_pil.size[0],0))

        new_image.save(destination_path_scaled_debug_images+'/'+image_name,"JPEG")

        rotated_data_1['images'][i]['file_name'] = file_name
        rotated_data_1['images'][i]['width'] = new_width
        rotated_data_1['images'][i]['height'] = new_height
    
    rotated_data_1['annotations'] = rotate_annos
    rotated_data_1['licenses'] = coco.dataset['licenses']
    cats = coco.loadCats(coco.getCatIds())       
    rotated_data_1['categories'] = cats

    json.dump(rotated_data_1,open(destination_path_scaled_annotations+'/scaled_'+str(new_width)+'_'+str(new_height)+'.json','w'))  
    
    print('------------------------------------------------------------')

    print('CREATED')
