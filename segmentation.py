### 导入必要的包

import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset


### 设置参数

images_file = '.../data/training/Images/'  # 训练图像路径
gt_file = '.../data/training/Lesion Masks/Detachment'
image_size = 256 # 输入图像统一尺寸 (image_size, image_size, 3)
val_ratio = 0.2  # 训练/验证图像划分比例
BATCH_SIZE = 8 # 批大小
iters = 3000 # 训练迭代次数
optimizer_type = 'adam' # 优化器, 可自行使用其他优化器，如SGD, RMSprop,...
num_workers = 4 # 数据加载处理器个数
init_lr = 1e-3 # 初始学习率


filelists = os.listdir(images_file)
train_filelists, val_filelists = train_test_split(filelists, test_size = val_ratio,random_state = 42)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))


### 从数据文件夹中加载眼底图像，提取相应的金标准，生成训练样本

class OCTDataset(Dataset):
    def __init__(self, image_file, gt_path=None, filelists=None,  mode='train'):
        super(OCTDataset, self).__init__()
        self.mode = mode
        self.image_path = image_file
        image_idxs = os.listdir(self.image_path) # 0001.png,
        self.gt_path = gt_path

        self.file_list = [image_idxs[i] for i in range(len(image_idxs))]        
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item in filelists] 
   
    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        # print(real_index)
        img_path = os.path.join(self.image_path, real_index)
        img = cv2.imread(img_path) 
        h,w,c = img.shape       

        if self.mode == 'train':
            img_index = real_index.split('.')[0] + '.bmp'            
            gt_tmp_path = os.path.join(self.gt_path, img_index)
            if os.path.isfile(gt_tmp_path):
                gt_img = cv2.imread(gt_tmp_path)
            else:
                # print(img_index)
                gt_img = np.ones((h, w, c)) * 255

            ### 像素值为0的是RNFL(类别 0)，像素值为80的是GCIPL(类别 1)，像素值为160的是脉络膜(类别 2)，像素值为255的是其他（类别3）。
            
            gt_img[gt_img == 255] = 1
            gt_img = cv2.resize(gt_img,(image_size, image_size))
            gt_img = gt_img[:,:,1]
            # print('gt shape', gt_img.shape)           

        img_re = cv2.resize(img,(image_size, image_size))
        img = img_re.transpose(2, 0, 1) # H, W, C -> C, H, W
        # print(img.shape)
        # img = img_re.astype(np.float32)
        
        if self.mode == 'test':
            ### 在测试过程中，加载数据返回眼底图像，数据名称，原始图像的高度和宽度
            return img, real_index, h, w
        
        if self.mode == 'train':
            ###在训练过程中，加载数据返回眼底图像及其相应的金标准           
            return img, gt_img

    def __len__(self):
        return len(self.file_list)


class SeparableConv2D(nn.Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=None, 
                 weight_attr=None, 
                 bias_attr=None, 
                 data_format="NCHW"):
        super(SeparableConv2D, self).__init__()

        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._in_channels = in_channels
        self._data_format = data_format

        # 第一次卷积参数，没有偏置参数
        filter_shape = [in_channels, 1] + self.convert_to_list(kernel_size, 2, 'kernel_size')
        self.weight_conv = self.create_parameter(shape=filter_shape, attr=weight_attr)

        # 第二次卷积参数
        filter_shape = [out_channels, in_channels] + self.convert_to_list(1, 2, 'kernel_size')
        self.weight_pointwise = self.create_parameter(shape=filter_shape, attr=weight_attr)
        self.bias_pointwise = self.create_parameter(shape=[out_channels], 
                                                    attr=bias_attr, 
                                                    is_bias=True)
    
    def convert_to_list(self, value, n, name, dtype=np.int):
        if isinstance(value, dtype):
            return [value, ] * n
        else:
            try:
                value_list = list(value)
            except TypeError:
                raise ValueError("The " + name +
                                "'s type must be list or tuple. Received: " + str(
                                    value))
            if len(value_list) != n:
                raise ValueError("The " + name + "'s length must be " + str(n) +
                                ". Received: " + str(value))
            for single_value in value_list:
                try:
                    dtype(single_value)
                except (ValueError, TypeError):
                    raise ValueError(
                        "The " + name + "'s type must be a list or tuple of " + str(
                            n) + " " + str(dtype) + " . Received: " + str(
                                value) + " "
                        "including element " + str(single_value) + " of type" + " "
                        + str(type(single_value)))
            return value_list
    
    def forward(self, inputs):
        conv_out = F.conv2d(inputs, 
                            self.weight_conv, 
                            padding=self._padding,
                            stride=self._stride,
                            dilation=self._dilation,
                            groups=self._in_channels,
                            data_format=self._data_format)
        
        out = F.conv2d(conv_out,
                       self.weight_pointwise,
                       bias=self.bias_pointwise,
                       padding=0,
                       stride=1,
                       dilation=1,
                       groups=1,
                       data_format=self._data_format)

        return out


class Encoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        
        self.relus = nn.LayerList(
            [nn.ReLU() for i in range(2)])
        self.separable_conv_01 = SeparableConv2D(in_channels, 
                                                 out_channels, 
                                                 kernel_size=3, 
                                                 padding='same')
        self.bns = nn.LayerList(
            [nn.BatchNorm2D(out_channels) for i in range(2)])
        
        self.separable_conv_02 = SeparableConv2D(out_channels, 
                                                 out_channels, 
                                                 kernel_size=3, 
                                                 padding='same')
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.residual_conv = nn.Conv2D(in_channels, 
                                        out_channels, 
                                        kernel_size=1, 
                                        stride=2, 
                                        padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs
        
        y = self.relus[0](inputs)
        y = self.separable_conv_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.separable_conv_02(y)
        y = self.bns[1](y)
        y = self.pool(y)
        
        residual = self.residual_conv(previous_block_activation)
        y = paddle.add(y, residual)

        return y

class Decoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.relus = nn.LayerList(
            [nn.ReLU() for i in range(2)])
        self.conv_transpose_01 = nn.Conv2DTranspose(in_channels, 
                                                           out_channels, 
                                                           kernel_size=3, 
                                                           padding=1)
        self.conv_transpose_02 = nn.Conv2DTranspose(out_channels, 
                                                           out_channels, 
                                                           kernel_size=3, 
                                                           padding=1)
        self.bns = nn.LayerList(
            [nn.BatchNorm2D(out_channels) for i in range(2)]
        )
        self.upsamples = nn.LayerList(
            [nn.Upsample(scale_factor=2.0) for i in range(2)]
        )
        self.residual_conv = nn.Conv2D(in_channels, 
                                        out_channels, 
                                        kernel_size=1, 
                                        padding='same')

    def forward(self, inputs):
        previous_block_activation = inputs

        y = self.relus[0](inputs)
        y = self.conv_transpose_01(y)
        y = self.bns[0](y)
        y = self.relus[1](y)
        y = self.conv_transpose_02(y)
        y = self.bns[1](y)
        y = self.upsamples[0](y)
        
        residual = self.upsamples[1](previous_block_activation)
        residual = self.residual_conv(residual)
        
        y = paddle.add(y, residual)
        
        return y

class OCT_Layer_UNet(nn.Layer):
    def __init__(self, num_classes):
        super(OCT_Layer_UNet, self).__init__()

        self.conv_1 = nn.Conv2D(3, 32, 
                                kernel_size=3,
                                stride=2,
                                padding='same')
        self.bn = nn.BatchNorm2D(32)
        self.relu = nn.ReLU()

        in_channels = 32
        self.encoders = []
        self.encoder_list = [64, 128, 256]
        self.decoder_list = [256, 128, 64, 32]

        # 根据下采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.encoder_list:
            block = self.add_sublayer('encoder_{}'.format(out_channels),
                                      Encoder(in_channels, out_channels))
            self.encoders.append(block)
            in_channels = out_channels

        self.decoders = []

        # 根据上采样个数和配置循环定义子Layer，避免重复写一样的程序
        for out_channels in self.decoder_list:
            block = self.add_sublayer('decoder_{}'.format(out_channels), 
                                      Decoder(in_channels, out_channels))
            self.decoders.append(block)
            in_channels = out_channels

        self.output_conv = nn.Conv2D(in_channels, 
                                            num_classes, 
                                            kernel_size=3, 
                                            padding='same')
    
    def forward(self, inputs):
        y = self.conv_1(inputs)
        y = self.bn(y)
        y = self.relu(y)
        
        for encoder in self.encoders:
            y = encoder(y)

        for decoder in self.decoders:
            y = decoder(y)
        
        y = self.output_conv(y)
        return y

class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.
    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=2):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = paddle.unsqueeze(labels, 1)
        num_classes = logits.shape[1]
        mask = (labels != self.ignore_index)
        logits = logits * mask
        labels = paddle.cast(labels, dtype='int32')
        single_label_lists = []
        for c in range(num_classes):
            single_label = paddle.cast((labels == c), dtype='int32')
            single_label = paddle.squeeze(single_label, axis=1)
            single_label_lists.append(single_label)
        labels_one_hot = paddle.stack(tuple(single_label_lists), axis=1)
        logits = F.softmax(logits, axis=1)
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')
        dims = (0,) + tuple(range(2, labels.ndimension()))
        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return dice_loss

### 训练函数

def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, metric, log_interval, evl_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_dice_list = []
    best_dice = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            img = (data[0]/255.).astype("float32")
            gt_label = (data[1]).astype("int64")
            # print('label shape: ', gt_label.shape)
            logits = model(img)
            # print('logits shape: ', logits.shape)
            loss = criterion(logits, gt_label)
            # print('loss: ',loss)
            dice = metric(logits, gt_label) 
            # print('dice: ', dice)

            loss.backward()
            optimizer.step()

            model.clear_gradients()
            avg_loss_list.append(loss.numpy()[0])
            avg_dice_list.append(dice.numpy()[0]) 

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_dice = np.array(avg_dice_list).mean()
                avg_loss_list = []
                avg_dice_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_dice={:.4f}".format(iter, iters, avg_loss, avg_dice))

            if iter % evl_interval == 0:
                avg_loss, avg_dice = val(model, val_dataloader)
                print("[EVAL] iter={}/{} avg_loss={:.4f} dice={:.4f}".format(iter, iters, avg_loss, avg_dice))
                if avg_dice >= best_dice:
                    best_dice = avg_dice
                    paddle.save(model.state_dict(),
                                os.path.join(".../segmentation-detachment/best_model_{:.4f}".format(best_dice), 'model.pdparams'))
                model.train()

### 验证函数

def val(model, val_dataloader):
    model.eval()
    avg_loss_list = []
    avg_dice_list = []
    with paddle.no_grad():
        for data in val_dataloader:
            img = (data[0] / 255.).astype("float32")
            gt_label = (data[1]).astype("int64")

            pred = model(img)
            loss = criterion(pred, gt_label)
            dice = metric (pred, gt_label)  

            avg_loss_list.append(loss.numpy()[0])
            avg_dice_list.append(dice.numpy()[0])

    avg_loss = np.array(avg_loss_list).mean()
    avg_dice = np.array(avg_dice_list).mean()

    return avg_loss, avg_dice


# # 训练阶段
# ### 生成训练集和验证集
# train_dataset = OCTDataset(image_file = images_file, 
#                         gt_path = gt_file,
#                         filelists=train_filelists)

# val_dataset = OCTDataset(image_file = images_file, 
#                         gt_path = gt_file,
#                         filelists=val_filelists)

# ### 加载数据
# train_loader = paddle.io.DataLoader(
#     train_dataset,
#     batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
#     num_workers=num_workers,
#     return_list=True,
#     use_shared_memory=False
# )

# val_loader = paddle.io.DataLoader(
#     val_dataset,
#     batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
#     num_workers=num_workers,
#     return_list=True,
#     use_shared_memory=False
# )

# model = OCT_Layer_UNet(num_classes=2)

# if optimizer_type == "adam":
#     optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())

# criterion = nn.CrossEntropyLoss(axis=1)
# metric = DiceLoss()

# ### 开始训练
# train(model, iters, train_loader, val_loader, optimizer, criterion, metric, log_interval=10, evl_interval=50)


#预测阶段
### 加载模型参数
test_file = '.../data/validation/Images'  # 测试图像路径
best_model_path = ".../segmentation-detachment/best_model_0.5282/model.pdparams"
model = OCT_Layer_UNet(num_classes = 2)
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

### 生成测试集

test_dataset = OCTDataset(image_file = test_file, 
                            mode='test')
### 一张一张分割测试集中的图像
### 分割结果存储格式为bmp

for img, idx, h, w in test_dataset:
    # print(idx)
    img = img[np.newaxis, ...]
    img = paddle.to_tensor((img / 255.).astype("float32"))
    logits = model(img)
    pred_img = logits.numpy().argmax(1)
    pred_gray = np.squeeze(pred_img, axis=0)
    pred_gray = pred_gray.astype('float32')
    # print(pred_gray.shape)
    pred_gray[pred_gray == 1] = 255
    # print(pred_gray)
    pred_ = cv2.resize(pred_gray, (w, h))
    # print(pred_.shape)
    cv2.imwrite('.../results/Detachment_Segmentations-val/'+idx, pred_)
