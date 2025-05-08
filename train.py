import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import Trainer
from torchvision.models import vgg16
from backbone import VGG16

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/STL100 Training')
    parser.add_argument('--dataset', default='stl10', type=str)
    parser.add_argument('--data_dir', default='datasets/datasets', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='vgg11')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    # Define your model here (replace with your custom model if needed)
    model = VGG16(attn_type='cbam', num_heads=8, weights=None, num_classes=10)

    # Run trainer
    trainer = Trainer(args, model)
    trainer.run()
