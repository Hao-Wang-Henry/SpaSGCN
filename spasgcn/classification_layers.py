import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

import numpy as np

import sys, os, pdb, pickle
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(file_path)[0])
from spasgcn.utils import gen_indexes, v_num_dict


def Model_select(model='resnet_6np2', *args, **kwargs):
    return ResNet_6np2(*args, **kwargs)

def Data_select(dataset='cifar10', *args, **kwargs):
    return Cifar10_Data(*args, **kwargs)


class SelectTrainSphereConv(nn.Module):
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=7,
                 stride=1,
                 select_index=None,
                 select_agl=None,
                 pre_layers=None,
                 post_layers=None,
                 *agrs, **kwargs):
        super(SelectTrainSphereConv, self).__init__()
        self.index, itp_mat = gen_indexes(graph_level, 'conv', kernel_size)
        self.register_buffer("itp_mat", itp_mat)
        self.angle_index = np.array(np.arange(v_num_dict[graph_level]))
        self.conv = nn.Conv2d(input_channel, output_channel, (1, kernel_size))
        self.graph_level = graph_level
        self.kernel_size = kernel_size
        self.stride = stride
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        # select_index1: input
        if select_index is None:
            select_index1 = np.arange(v_num_dict[graph_level])
        else:
            select_index1 = np.array(select_index).squeeze()
            assert select_index1.ndim == 1
        # select_index2: stride
        select_index2 = np.arange(v_num_dict[graph_level-stride+1])
        # select_index3: = self.angle_index
        self.test_select_index = np.intersect1d(select_index1, select_index2)
        self.train_select_index = np.intersect1d(self.test_select_index, self.angle_index)

        assert len(self.train_select_index) > 0
        if kernel_size > 1:
            test_itp_mat = self.itp_mat[self.test_select_index]
            train_itp_mat = self.itp_mat[self.train_select_index]
        else:
            test_itp_mat = torch.tensor(1)
            train_itp_mat = torch.tensor(1)
        self.register_buffer("test_itp_mat", test_itp_mat)  # v_num_test * 7 * kernel_size
        self.register_buffer("train_itp_mat", train_itp_mat)  # v_num_train * 7 * kernel_size


    def interpolate(self, tensor):
        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num * 7 * bs * inc
        if self.training:
            tensor = tensor[self.train_select_index]  # v_num2 * 7 * bs * inc
        else:
            tensor = tensor[self.test_select_index]  # v_num2 * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num2 * 7

        tensor = tensor.unsqueeze(-2)  # bs * inc * v_num2 * 1 * 7
        if self.kernel_size == 1:
            tensor = tensor[:, :, :, :, -1]  # bs * inc * v_num2 * 1
            return tensor

        # self.itp_mat  # v_num2 * 7 * ks
        out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks
        out_tensor = out_tensor.squeeze(-2)  # bs * inc * v_num2 * ks
        return out_tensor

    def forward(self, tensor):
        if self.pre_layers:
            tensor = self.pre_layers(tensor)
        assert tensor.dim() == 3  # batch_size * input_channel * v_num

        tensor = self.interpolate(tensor)  # batch_size * input_channel * v_num2 * kernel_size
        tensor = self.conv(tensor)  # batch_size * outc * v_num2 * 1
        tensor = tensor.squeeze(-1)  # batch_size * outc * v_num2

        if self.post_layers:
            tensor = self.post_layers(tensor)
        return tensor

class SpherePool(nn.Module):
    def __init__(self, graph_level,
                 method='sample',
                 *agrs, **kwargs):
        super(SpherePool, self).__init__()
        self.index = gen_indexes(graph_level, 'pool')
        self.method = method

    def forward(self, tensor):
        assert tensor.dim() == 3  # batch_size * input_channel * v_num

        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num' * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num' * 7

        if self.method == 'sample':
            tensor = tensor[:, :, :, -1]
        elif self.method == 'max':
            tensor = tensor.max(dim=-1).values
        elif self.method == 'mean':
            tensor = tensor.mean(dim=-1).values
        return tensor  # bs * inc * v_num'


class ResidualBlockSelTrain(nn.Module):
    def __init__(self, graph_level, in_channel, out_channel,
                 kernel_size=7, stride=1, select_agl=None, *args, **kwargs):
        super(ResidualBlockSelTrain, self).__init__()

        self.conv1 = SelectTrainSphereConv(graph_level, in_channel, out_channel,
                                      kernel_size, stride=stride, select_agl=select_agl,
                     post_layers=nn.Sequential(nn.BatchNorm1d(out_channel),
                                                nn.ReLU(inplace=True)))
        self.conv2 = SelectTrainSphereConv(graph_level-stride+1, out_channel, out_channel,
                                      kernel_size, stride=1, select_agl=select_agl,
                     post_layers=nn.BatchNorm1d(out_channel))
        if in_channel != out_channel or stride != 1:
            self.shortcut = SelectTrainSphereConv(graph_level, in_channel, out_channel,
                                      kernel_size=1, stride=stride, select_agl=select_agl,
                            post_layers=nn.BatchNorm1d(out_channel))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_6np2(nn.Module):
    def __init__(self, graph_level=5, kernel_size=7, pooling='max',
                 num_classes=10, Nstacks=1, select_agl=None, *args, **kwargs):
        super(ResNet_6np2, self).__init__()
        channel = 16
        gl, ks, self.graph_level = graph_level, kernel_size, graph_level
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.num_classes = num_classes
        self.Nstacks = Nstacks
        self.conv1 = SelectTrainSphereConv(gl, 3, channel, ks, stride=1, select_agl=select_agl,
                     post_layers=nn.Sequential(nn.BatchNorm1d(channel),
                                                nn.ReLU(inplace=True)))
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []


        for stack in range(Nstacks):
            block = ResidualBlockSelTrain(gl, channel, channel, ks, stride=1, select_agl=select_agl)
            setattr(self, 'block1%d' % (stack+1), block)
            self.blocks1.append(block)
        self.pool1 = SpherePool(gl, 'max')

        for stack in range(Nstacks):
            if stack == 0:
                block = ResidualBlockSelTrain(gl-1, channel, channel*2, ks, stride=1, select_agl=select_agl)
            else:
                block = ResidualBlockSelTrain(gl-1, channel*2, channel*2, ks, stride=1, select_agl=select_agl)
            setattr(self, 'block2%d' % (stack+1), block)
            self.blocks2.append(block)
        self.pool2 = SpherePool(gl-1, 'max')

        for stack in range(Nstacks):
            if stack == 0:
                block = ResidualBlockSelTrain(gl-2, channel*2, channel*4, ks, stride=1, select_agl=select_agl)
            else:
                block = ResidualBlockSelTrain(gl-2, channel*4, channel*4, ks, stride=1, select_agl=select_agl)
            setattr(self, 'block3%d' % (stack+1), block)
            self.blocks3.append(block)
        self.linear = nn.Linear(channel*4, num_classes)


    def forward(self, x):
        assert x.dim() == 3  # batch * 3 * v_5(10242)
        assert x.shape[2] == 10242
        assert x.shape[1] == 3

        x = self.conv1(x)  # batch * 16 * v_5(10242)
        for stack in range(self.Nstacks):
            x = self.blocks1[stack](x)  # batch * 16 * v_5(10242)
        x = self.pool1(x)  # batch * 16 * 2562
        for stack in range(self.Nstacks):
            x = self.blocks2[stack](x)  # batch * 32 * 2562
        x = self.pool2(x)  # batch * 32 * 642
        for stack in range(self.Nstacks):
            x = self.blocks3[stack](x)  # batch * 64 * 642
        x = x.max(dim=2).values  # batch * 64
        x = self.linear(x)  # batch * 10
        result = x.argmax(dim=1)  # batch
        return x, result


class Cifar10_Data(Dataset):
    def __init__(self, root='data', act='test', graph_level=5,
                       lookup_angle=0.2, rotate='fr'):
        super().__init__()
        self.imgs, self.labels = [], []
        imgs, labels = self.data_prepare(root, act, graph_level, lookup_angle, rotate)
        # img_num * v_num * inc, img_num
        with torch.no_grad():
            for i in range(len(labels)):
                self.imgs.append(torch.tensor(imgs[i].T).float())  # inc * v_num
                self.labels.append(labels[i])

    def data_prepare(self, root, act, graph_level, lookup_angle, rotate):
        try:
            data = np.load('data/cifar10test.npy', allow_pickle=True).item()
            imgs = data['img']  # img_num * v_num * inc
            labels = data['label']  # img_num
            print(f"load data from data/cifar10test.npy, find {len(imgs)} {act} imgs")
            return imgs, labels
        except (FileNotFoundError, AssertionError):
            print("will release soon")
            sys.exit()

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    sys.exit()
