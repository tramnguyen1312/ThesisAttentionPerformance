import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import backbone.ResNET as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from PIL import Image
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from attention.CBAM import CBAMBlock
from attention.BAM import BAMBlock
from attention.scSE import ChannelSpatialSELayer

def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]       # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def main(args):
    if not os.path.exists("/kaggle/working/logs"):
        os.makedirs("/kaggle/working/logs")
    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('/kaggle/working/logs/'+args.model+'.log', True)

    logger.append(vars(args))


    writer = SummaryWriter()


    gpus = args.gpu.split(',')
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_datasets = datasets.ImageFolder(os.path.join(args.data_root, 'train'), data_transforms['train'])
    val_datasets   = datasets.ImageFolder(os.path.join(args.data_root, 'test'), data_transforms['val'])
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size*len(gpus), shuffle=True, num_workers=8)
    val_dataloaders   = torch.utils.data.DataLoader(val_datasets, batch_size=1024, shuffle=False, num_workers=8)

    if args.debug:
        x, y =next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    attention_block = None
    if args.attention == 'CBAM':
        attention_block = CBAMBlock
    elif args.attention == 'BAM':
        attention_block = BAMBlock
    elif args.attention == 'SCSE':
        attention_block = ChannelSpatialSELayer

    if  'resnet50' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50(pretrained=True, attention_block=attention_block)
    elif 'resnet101' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet101(pretrained=False, attention_block=attention_block)
    else:
        raise ModuleNotFoundError

    #my_model.apply(fc_init)
    if is_use_cuda and 1 == len(gpus):
        my_model = my_model.cuda()
    elif is_use_cuda and 1 < len(gpus):
        my_model = nn.DataParallel(my_model.cuda())

    print(my_model)
    loss_fn = [nn.CrossEntropyLoss()]
    optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)           #

    metric = [ClassErrorMeter([1,5], True)]
    start_epoch = 0
    num_epochs  = args.epochs

    trainer = Trainer(
        model=my_model,
        optimizer=optimizer,
        scheduler=lr_schedule,
        train_dataloader=train_dataloaders,
        val_dataloader=val_dataloaders,
        criterion=nn.CrossEntropyLoss(),
        max_epochs=100,
        max_plateau_count=10,
        min_lr=1e-5,
        checkpoint_path="best_model.pth",
        output_csv_path="training_log.csv",
    )

    trainer.fit()
    logger.append('Optimize Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')
    parser.add_argument('-d', '--data_root', default='./datasets',
                         type=str, help='data root')
    parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
                         type=str, help='train file')
    parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
                         type=str, help='validation file')
    parser.add_argument('-m', '--model', default='resnet50',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=12,
                         type=int, help='model train batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Use TensorboardX to Display')
    parser.add_argument('--attention', type=str, default="CBAM")
    args = parser.parse_args()

    main(args)