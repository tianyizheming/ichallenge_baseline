import sys 
import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances 
import matplotlib.pylab as plt

import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50
from paddle.io import Dataset


test_file = '.../data/testing/Images/'  # the path to the testing data

image_size = 256 # the image size to the network (image_size, image_size, 3)

class FundusDataset(Dataset):
    def __init__(self, image_file, gt_file=None, filelists=None,  mode='train'):
        super(FundusDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(self.image_path)
        self.gt_file = gt_file

        if self.mode == 'train':
            label = {row['imgName']: row[1:].values 
                        for _, row in pd.read_excel(gt_file).iterrows()}
            self.file_list = [[image_idxs[i], label[image_idxs[i]]] for i in range(len(image_idxs))]
        
        elif self.mode == 'test':
            self.file_list = [[image_idxs[i], None] for i in range(len(image_idxs))]
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists] 
   
    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]

        fundus_img_path = os.path.join(self.image_path, real_index)

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1] # BGR -> RGB        
        h,w,c = fundus_img.shape
        if self.mode == 'train':
            label_nor = (float(label[0])/w, float(label[1])/h)
            label_nor = np.array(label_nor).astype('float32').reshape(2)
        fundus_re = cv2.resize(fundus_img,(image_size, image_size))
        img = fundus_re.transpose(2, 0, 1) # H, W, C -> C, H, W
        # print(img.shape)
        # img = fundus_re.astype(np.float32)
        
        if self.mode == 'test':
            return img, real_index, h, w
        if self.mode == 'train':
            return img, label_nor

    def __len__(self):
        return len(self.file_list)


class Network(paddle.nn.Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.resnet = resnet50(pretrained=True, num_classes=2) # remove final fc 输出为[?, 2048, 1, 1]
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(2048, 512)
        self.linear_2 = paddle.nn.Linear(512, 256)
        self.linear_3 = paddle.nn.Linear(256, 2)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)
    
    def forward(self, inputs):
        # print('input', inputs)
        y = self.resnet(inputs)
        y = self.flatten(y)
        y = self.linear_1(y)
        y = self.linear_2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_3(y)
        y = paddle.nn.functional.sigmoid(y)

        return y


best_model_path = ".../Localization/best_model_0.0374/model.pdparams"
model = Network()
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

test_dataset = FundusDataset(image_file = test_file, 
                       mode='test')

cache = []
for fundus_img, idx, h, w in test_dataset:
    fundus_img = fundus_img[np.newaxis, ...]    
    fundus_img = paddle.to_tensor((fundus_img / 255.).astype("float32"))    
    logits = model(fundus_img)
    pred_coor = logits.numpy()
    # print(pred_coor)
    x = pred_coor[0][0] * w
    y = pred_coor[0][1] * h
    cache.append([idx, x, y])

submission_result = pd.DataFrame(cache, columns=['imgName', 'Fovea_X', 'Fovea_Y'])
submission_result[['imgName', 'Fovea_X', 'Fovea_Y']].to_csv(".../results/Localization_Results-test.csv", index=False)

