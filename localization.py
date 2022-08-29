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

path = '.../data/training/'  # the path to the training data
images_file = path + 'Images/'
gt_file = path + 'Fovea Localization.xlsx'
test_file = '/.../data/validation/Images/'  # the path to the testing data
image_size = 256 # the image size to the network (image_size, image_size, 3)
val_ratio = 0.2 # the ratio of train/validation splitition
BATCH_SIZE = 32  # batch size
iters = 500 # training iteration
optimizer_type = 'adam' # the optimizer, can be set as SGD, RMSprop,...
num_workers = 4 # Number of workers used to load data
init_lr = 1e-4 # the initial learning rate

filelists = os.listdir(images_file)
train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio,random_state = 42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

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


def cal_ed(logit, label):
    ed_loss = []
    for i in range(logit.shape[0]):
        logit_tmp = logit[i,:].numpy()
        label_tmp = label[i,:].numpy()
        # print('cal_coordinate_loss_ed', logit_tmp, label_tmp)        
        ed_tmp = euclidean_distances([logit_tmp], [label_tmp])
        # print('ed_tmp:', ed_tmp[0][0])
        ed_loss.append(ed_tmp)
    
    ed_l = sum(ed_loss)/len(ed_loss)
    return ed_l


def cal_ed_val(logit, label):
    ed_loss = []
    for i in range(logit.shape[0]):
        logit_tmp = logit[i,:]
        label_tmp = label[i,:]
        ed_tmp = euclidean_distances([logit_tmp], [label_tmp])
        ed_loss.append(ed_tmp)
    
    ed_l = sum(ed_loss)/len(ed_loss)
    
    return ed_l


# loss
def cal_coordinate_Loss(logit, label, alpha = 0.5):
    """
    logit: shape [batch, ndim]
    label: shape [batch, ndim]
    ndim = 2 represents coordinate_x and coordinaate_y
    alpha: weight for MSELoss and 1-alpha for ED loss
    return: combine MSELoss and ED Loss for x and y, shape [batch, 1]
    """
    alpha = alpha
    mse_loss = nn.MSELoss(reduction='mean')

    mse_x = mse_loss(logit[:,0],label[:,0])
    mse_y = mse_loss(logit[:,1],label[:,1])
    mse_l = 0.5*(mse_x + mse_y)
    # print('mse_l', mse_l)

    ed_loss = []
    # print(logit.shape[0])
    for i in range(logit.shape[0]):
        logit_tmp = logit[i,:].numpy()
        label_tmp = label[i,:].numpy()
        # print('cal_coordinate_loss_ed', logit_tmp, label_tmp)        
        ed_tmp = euclidean_distances([logit_tmp], [label_tmp])
        # print('ed_tmp:', ed_tmp[0][0])
        ed_loss.append(ed_tmp)
    
    ed_l = sum(ed_loss)/len(ed_loss)
    # print('ed_l', ed_l)
    # print('alpha', alpha)
    loss = alpha * mse_l + (1-alpha) * ed_l
    # print('loss in function', loss)
    return loss


### Training function

def train(model, iters, train_dataloader, val_dataloader, optimizer, log_interval, evl_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_ED_list = []
    best_ED = sys.float_info.max
    while iter < iters:
        for img, lab in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_imgs = (img / 255.).astype('float32')
            label = lab.astype("float32")

            logits = model(fundus_imgs)
            loss = cal_coordinate_Loss(logits, label)
            # print('loss in train',loss)

            for p,l in zip(logits.numpy(), label.numpy()):
                avg_ED_list.append([p,l])
            
            # print('avg_ED_list', avg_ED_list)
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            avg_loss_list.append(loss.numpy()[0])
            
            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                # print(avg_loss)
                avg_ED_list = np.array(avg_ED_list)
                avg_ED = cal_ed_val(avg_ED_list[:, 0], avg_ED_list[:, 1]) # cal_ED
                # print('ed in training', avg_ED)
                avg_loss_list = []
                avg_ED_list = []
                
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_ED={:.4f}".format(iter, iters, avg_loss, avg_ED[0][0]))

            if iter % evl_interval == 0:
                avg_loss, avg_ED = val(model, val_dataloader)
                print("[EVAL] iter={}/{} avg_loss={:.4f} ED={:.4f}".format(iter, iters, avg_loss, avg_ED[0][0]))
                if avg_ED <= best_ED:
                    best_ED = avg_ED[0][0]
                    paddle.save(model.state_dict(),
                            os.path.join("best_model_{:.4f}".format(best_ED), 'model.pdparams'))
                model.train()


### validation function

def val(model, val_dataloader):
    model.eval()
    avg_loss_list = []
    cache = []
    with paddle.no_grad():
        for data in val_dataloader:
            fundus_imgs = (data[0] / 255.).astype("float32")
            labels = data[1].astype('float32')
            
            logits = model(fundus_imgs)
            for p, l in zip(logits.numpy(), labels.numpy()):
                cache.append([p, l])

            loss = cal_coordinate_Loss(logits, labels)
            avg_loss_list.append(loss.numpy()[0])

    cache = np.array(cache)
    ED = cal_ed_val(cache[:, 0], cache[:, 1])
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, ED

### generate training Dataset and validation Dataset 

train_dataset = FundusDataset(image_file = images_file, 
                       gt_file=gt_file,
                       filelists=train_filelists)

val_dataset = FundusDataset(image_file = images_file, 
                       gt_file=gt_file,
                       filelists=val_filelists)

train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False),
    # num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False),
    # num_workers=num_workers,
    return_list=True,
    use_shared_memory=False
)

model = Network()

if optimizer_type == "adam":
    optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())


train(model, iters, train_loader, val_loader, optimizer, log_interval=10, evl_interval=100)
