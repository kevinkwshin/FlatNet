import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import matplotlib.pyplot as plt


def euclidean_distance(pt1, pt2):
    distance = 0
    for i in range(len(pt1)):
        distance += (pt1[i] - pt2[i]) ** 2
    return distance ** 0.5


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image_origin(image_path, model_path, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.1)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow('detections', image_orig)
            cv2.waitKey(0)


            
def load_model(model_path):
    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    
    return model
            
            
                 
def detect_image(model,image_path):
    
    image = cv2.imread(image_path)

    if image is None:
        print("Error!")
        
    image_orig = image.copy()
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)

    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
#     print('resized image',image.shape)
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():
        
        Predict_Point = np.zeros((45,6),dtype=np.uint32)
        best_scores = [0.0]*45

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        #print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.cuda().float())

        idxs = np.where(scores.cpu() > 0.0)
        
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            current_class = int(classification[j])
                      
            if(scores[j] > best_scores[current_class]):
                best_scores[current_class] = scores[j] 
                Predict_Point[current_class][0] = (int(scores[j]*100))
                Predict_Point[current_class][1] = (int(bbox[0] / scale))
                Predict_Point[current_class][2] = (int(bbox[1] / scale))
                Predict_Point[current_class][3] = (int(bbox[2] / scale))
                Predict_Point[current_class][4] = (int(bbox[3] / scale))                   
        
    return Predict_Point



def Get_Center_Points(y_temp):
    xy_args = np.argwhere(y_temp > 0.7)
    if xy_args.size != 0:
        transposed_xy_args = xy_args.transpose()
        y_check = transposed_xy_args[0]
        x_check = transposed_xy_args[1]
        pr_y_mean = int(y_check.mean())
        pr_x_mean = int(x_check.mean())

    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        pixel = []
        max_value = np.max(y_temp)
        xy_args = np.argwhere(y_temp == max_value)
        transposed_xy_args = xy_args.transpose()
        y_check = transposed_xy_args[0]
        x_check = transposed_xy_args[1]
        pr_y_mean = y_check.mean()
        pr_x_mean = x_check.mean()
        
    return pr_x_mean,pr_y_mean


def Draw_OOD_List(image_paths,gt_points,ppoints,distances,Landmark_num,th_dis):
    ood_list = []
    for i in range(len(distances)):
        if(distances[i][Landmark_num] > th_dis):
            ood_list.append(i)


    for i in range(len(ood_list)):
        print(ood_list[i],distances[ood_list[i]][Landmark_num])
        image = cv2.imread(image_paths[ood_list[i]])

        cv2.circle(image,(int(gt_points[i][Landmark_num][0]),int(gt_points[i][Landmark_num][1])),10,(0,255,0),-1 )
        cv2.circle(image,(int(ppoints[ood_list[i]][Landmark_num][0]),int(ppoints[ood_list[i]][Landmark_num][1])),10,(255,0,0),-1 )

        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.show()
        
    return ood_list


def Get_Distance(image_paths,gt_points,pr_points):
    del_list = [3,4,13,14,16,17,30,38,39]
    distances = []
    for i in range(len(image_paths)):
        distance = []
        for j in range(len(pr_points[0])):
            if(j in del_list):
                dis = 0
                distance.append(dis)
            else:
                dis = (euclidean_distance(gt_points[i][j], pr_points[i][j]) * 0.12)
                distance.append(dis)

        distances.append(distance)
    distances = np.array(distances)
    
    return distances

def Pre_Processing(image):
    image = image / 255.
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.cuda()
    return image

def Get_Distance2(image_paths,gt_points,pr_points):
    distances = []
    for i in range(len(image_paths)):
        distance = []
        for j in range(len(pr_points[0])):
            dis = (euclidean_distance(gt_points[i][j], pr_points[i][j]) * 0.12)
            distance.append(dis)

        distances.append(distance)
    distances = np.array(distances)
    
    return distances




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
