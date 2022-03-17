import os, glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import ceil



def data_check(image_paths, mask_paths,landmark_num, image_size):
    clean_images = []
    clean_points = []
    W = image_size
    H = image_size
    
    for i in range(len(image_paths)):
        temp_image = cv2.imread(image_paths[i],0)
        point = np.load(mask_paths[i])
        
        check_x = point[landmark_num][0]
        check_y = point[landmark_num][1]
        
        if(1 == np.isnan(check_x) or 1 == np.isnan(check_y)):
            print("np.nan!")
            print(image_paths[i].split("/")[-4:])
            continue
        
        origin_lu_x = (point[landmark_num][0]) - int(W/2)
        origin_lu_y = (point[landmark_num][1]) - int(H/2)
        origin_rd_x = (point[landmark_num][0]) + int(W/2)
        origin_rd_y = (point[landmark_num][1]) + int(H/2)

        agu_lu_x = origin_lu_x - 50
        agu_lu_y = origin_lu_y - 50
        agu_rd_x = origin_rd_x + 50
        agu_rd_y = origin_rd_y + 50
        
        max_x = temp_image.shape[1] + 1024
        max_y = temp_image.shape[0] + 1024
        
        if(agu_lu_x < 0 or agu_lu_y < 0 or agu_rd_x < 0 or agu_rd_y < 0 or agu_lu_x > max_x or agu_lu_y > max_y or agu_rd_x > max_x or agu_rd_y > max_y):
            print("range!")
            print(image_paths[i].split("/")[-4:])
            print(check_x,check_y)
            continue
        
        clean_images.append(image_paths[i])
        clean_points.append(mask_paths[i])
        
    return clean_images,clean_points