import random
from torch.utils.data import Dataset
import numpy as np
import cv2


class Dental_Single_Data_Generator(Dataset):

    def __init__(self, img_size, input_img_paths, target_img_paths, landmark_num = 0, mode = "train", transform=None, loss = "dice"):
        self.img_size = img_size
        self.input_img_paths = input_img_paths    
        self.target_img_paths = target_img_paths 
        self.LN = landmark_num
        self.max_size = 6000
        self.transform = transform
        self.mode = mode
        self.shift = 500
        self.random_shift = 100
        self.loss = loss   

    def __len__(self):
        return len(self.target_img_paths)
    

    def __getitem__(self, idx):
        input_img_paths = self.input_img_paths[idx]
        target_img_paths = self.target_img_paths[idx]
        fname = self.input_img_paths[idx]
        

        temp_image = cv2.imread(input_img_paths,0)
        point = np.load(target_img_paths)
        point[self.LN][0] += self.shift
        point[self.LN][1] += self.shift
        
        if(point[self.LN][0] > self.max_size or point[self.LN][1] > self.max_size or point[self.LN][0] < 0 or point[self.LN][1] < 0):
            
            point[self.LN][0] = int(self.max_size / 2)
            point[self.LN][1] = int(self.max_size / 2)
        
        pading_image = np.zeros((self.max_size,self.max_size),dtype=np.uint8) 
        pading_image[self.shift:temp_image.shape[0]+self.shift,self.shift:temp_image.shape[1]+self.shift] = temp_image   
#         print('pad',pading_image.shape)
        
        if(self.loss == "dice"):
            pading_mask = np.zeros((self.max_size,self.max_size),dtype=np.uint8)
            cv2.circle(pading_mask,(int(point[self.LN][0]),int(point[self.LN][1])),40,(255),-1 )
        
        elif(self.loss == "mse"):
            pading_mask = gaussian_kernel(point[self.LN],800)    

        
        SHIFT_X = random.randint(-self.random_shift, self.random_shift)
        SHIFT_Y = random.randint(-self.random_shift, self.random_shift)
        SHIFT_X = 0
        SHIFT_Y = 0
    
        lu_x = int(point[self.LN][0]) - int(self.img_size[1]/2) + SHIFT_X 
        lu_y = int(point[self.LN][1]) - int(self.img_size[0]/2) + SHIFT_Y 
        rd_x = int(point[self.LN][0]) + int(self.img_size[1]/2) + SHIFT_X 
        rd_y = int(point[self.LN][1]) + int(self.img_size[0]/2) + SHIFT_Y 
#         print(lu_y,rd_y,lu_x,rd_x)
        delta_y = lu_y 
        delta_x = lu_x 
        if delta_y < 0:
            lu_y -= delta_y
            rd_y -= delta_y
        if delta_x < 0:
            lu_x -= delta_x
            rd_x-= delta_x

        image = pading_image[lu_y:rd_y,lu_x:rd_x]
        mask = pading_mask[lu_y:rd_y,lu_x:rd_x]
        
        image = np.expand_dims(image, -1)
        mask = np.expand_dims(mask, -1)
#         print('before_transform',image.shape,mask.shape)
        sample = {'image': image, 'landmarks': mask}   
        
        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['landmarks'] /= 255.
#             print('after_transform',sample['image'].shape,sample['landmarks'].shape)
#             sample['image'].unsqueeze(1)
#             sample['landmarks'].unsqueeze(1)
        
        sample['fname'] = fname

        return sample
            

def gaussian_kernel(point, kernel_size = 100):
    
    center = kernel_size
    output_shape = (4000,4000)

    image = np.zeros(output_shape, dtype=np.float32)
    x, y = np.mgrid[-center:center+1, -center:center+1]
    g = np.exp(-(x**2/float(center)+y**2/float(center)))
    g = g * 255
    g = g.astype(np.float32)

    pos_x = int(point[0])
    pos_y = int(point[1])

    image_x1 = pos_x-center
    image_x2 = pos_x+center+1
    image_y1 = pos_y-center
    image_y2 = pos_y+center+1

    D1 = D2 = D3 = D4 = 0
    if image_x1 < 0:
        D1 = abs(image_x1)
        image_x1 = 0
    if image_x2 > output_shape[1]:
        D2 = image_x2 - output_shape[1]
        image_x2 = output_shape[1]
    if image_y1 < 0:
        D3 = abs(image_y1)
        image_y1 = 0
    if image_y2 > output_shape[0]:
        D4 = image_y2 - output_shape[0]
        image_y2 = output_shape[0]

    image[image_y1:image_y2, image_x1:image_x2] += g[0+D3:g.shape[0]-D4, 0+D1:g.shape[1]-D2]

    image[np.where(image>255)] = 255
        
    return image

            