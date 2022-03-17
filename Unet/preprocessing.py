import torch
import numpy as np
import random
import cv2
import pylab as plt

#DICE
class ToTensor(object):
    def __call__(self, data):
        input, label = data['image'], data['landmarks']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        data = {'image': torch.from_numpy(input), 'landmarks': torch.from_numpy(label)}

        return data

class RandomFlip(object):
    def __call__(self, sample):
        p = 0.5
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random() < p:
            image = cv2.flip(image, 0)
            image = np.array(image)
            image = np.expand_dims(image, axis=2)
            
            landmarks = cv2.flip(landmarks, 0)
            landmarks = np.expand_dims(landmarks, axis=2)
#           landmarks[:,0] = image.shape[1]-landmarks[:,0]
        return {'image': image, 'landmarks': landmarks}

class Invert(object):
    def __call__(self, sample):
        p = 0.5
        image = sample['image']
        if random.random() < p:
            image = 255-image
            sample['image'] = image
#         plt.imshow(image,cmap='gray')
#         plt.show()
        return sample    
    
class Gamma_2D(object):
    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        image = image.astype("uint8")
        
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def __call__(self, sample):
        p = 0.5
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random() < p:
            numlist = [0.5,0.8,1.1,1.5,1.8,2.0,2.3,2.6,2.9,3.2,3.5]
            var_gamma = random.sample(numlist, 1)
            var_gamma = var_gamma[0]
            image = self.adjust_gamma(image, gamma=var_gamma)
            image = image.astype("uint8")
            image = np.expand_dims(image, axis=-1)
        return {'image': image, 'landmarks': landmarks}
    
    
class RandomBrightness(object):
#     def __init__(self, delta=32):
    def __init__(self, delta=64):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        image = sample['image']
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            np.add(image, delta, out=image, casting="unsafe")
            #image += delta
            sample['image'] = image
        return sample
    
    
class Rotation_2D(object):
    def __call__(self, sample, degree = 10):
        p = 0.5
        image = sample['image']
        landmarks = sample['landmarks']
        R_move = random.randint(-degree,degree)
        if random.random() < p:
            #print("_rotation_2D")
            M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
            image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
            image = np.expand_dims(image, axis=-1)
            landmarks = cv2.warpAffine(landmarks,M,(image.shape[1],image.shape[0]))
            landmarks = np.expand_dims(landmarks, axis=-1)
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
        return {'image': image, 'landmarks': landmarks}
    
    
class Shift_2D(object):
    def __call__(self, sample, shift = 10):
        p = 0.5
       
        image = sample['image']
        landmarks = sample['landmarks']
        
        x_move = random.randint(-shift,shift)
        y_move = random.randint(-shift,shift)
        if random.random() < p:
            shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
            image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
            landmarks = cv2.warpAffine(landmarks, shift_M, (landmarks.shape[1], landmarks.shape[0]))
            image = np.expand_dims(image, axis=-1)
            landmarks = np.expand_dims(landmarks, axis=-1)
            
        return {'image': image, 'landmarks': landmarks}
    
        
class RandomSharp(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.5
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        if random.random() < p:
            
            image = cv2.filter2D(image, -1, kernel)
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image
                   
        return sample
    
       
class RandomBlur(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.5
        if random.random() < p:
            image = sample['image']
            image = cv2.blur(image,(3,3))
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image

        return sample
    
    
class RandomNoise(object):
    def __call__(self, sample):
        image = sample['image']
        
        p = 0.5
        if random.random() < p:
            image = image/255.0
            noise =  np.random.normal(loc=0, scale=1, size=image.shape)
            img2 = image*2
            n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.05)), (1-img2+1)*(1 + noise*0.05)*-1 + 2)/2, 0,1)
            n2 = n2 * 255
            n2 = n2.astype("uint8")
            #n2 = np.expand_dims(n2, axis=-1)
            sample['image'] = n2         
        
        return sample
       
class RandomClahe(object):
    def __call__(self, sample):
        image = sample['image']
        
        p = 0.5
        if random.random() < p:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image
        
        return sample
