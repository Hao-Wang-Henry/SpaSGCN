import spasgcn as spag
from spasgcn.utils import v_num_dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

import argparse
import sys, pdb, os, time, datetime
file_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(file_path, 'data')
# os.system(f'pip install -r requirements.txt')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# setting parameters
parser = argparse.ArgumentParser()
# data
parser.add_argument('-ds', '--data_set', default='cifar10', choices=['cifar10'])
parser.add_argument('-gl', '--graph_level', default=5, type=int)
# model
parser.add_argument('-ms', '--model', default='resnet_6np2', choices=['resnet_6np2'])
parser.add_argument('-bs', '--batch_size', default=10, type=int)
# train
parser.add_argument('-gpu', '--gpu_num', type=str)
# actions
parser.add_argument('-act', '--action', default='test', choices=['test'])
args = parser.parse_args()
print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------')

# load model
net = spag.Model_select()
dict = torch.load('data/cifar10.pth', map_location='cpu')
net.load_state_dict(dict)
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()
print(net)

# test
def test(test_rate=1, dataloader=None):
    loss_ave, accuracy, img_num = 0, 0, 0
    for i_batch, (input, label) in enumerate(dataloader):
        if use_gpu:
            input = input.cuda()  # bs * inc * v_num
            label = label.cuda()
        assert input.shape[2] == v_num_dict[args.graph_level]
        net.eval()
        output, result = net(input)
        loss = nn.CrossEntropyLoss(reduction='none')(output, label)
        accuracy += (result == label).sum()
        loss_ave += torch.sum(loss).detach()
        img_num = img_num + len(loss)

        if test_rate == 2 and i_batch < 10:
            print(f"label:   {label}")
            print(f"predict: {result}")

    loss_ave = float(loss_ave / img_num)
    accuracy = float(accuracy.float() / img_num * 100)

    if test_rate == 2:
        print(f"tested {img_num} imgs, accuracy = {accuracy:.2f}%")
        return

if __name__ == '__main__':
    if args.action == 'test':
        # load data
        data_test = spag.Data_select()
        data_loader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
        print(f"found {len(data_test)} test images")
        with torch.no_grad():
            test(test_rate=2, dataloader=data_loader_test)
        # pdb.set_trace()
        sys.exit()
    else:
        print("will release soon")
