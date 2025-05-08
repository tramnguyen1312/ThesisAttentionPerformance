import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import STL10Dataset
from torchvision.models import vgg16
from transforms import *

# from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_args():
    parser = ArgumentParser(description='train VGG19')
    parser.add_argument('--root', '-r', type=str, default='data', help='root directory of dataset')
    parser.add_argument('--epochs', '-e', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='number of workers')
    parser.add_argument('--logging', '-l', type=str, default='logging', help='logging directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    root = args.root
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    logging = args.logging

    train_dataset = STL10Dataset(root=root, train=True, transform=train_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    test_dataset = STL10Dataset(root=root, train=False, transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 drop_last=False)

    # writer = SummaryWriter(logging)

    model = vgg16().to(device)
    if os.path.exists('model/last.pt'):
        model.load_state_dict(torch.load('model/last.pt', map_location=torch.device('cpu')))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    best_acc = 0
    best_model = vgg16().to(device)
    if os.path.exists('model/best.pt'):
        best_model.load_state_dict(torch.load('model/best.pt', map_location=torch.device('cpu')))
        best_model.eval()
        all_predictions_best = []
        all_labels_best = []
        for iter, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = best_model(images)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs.cpu(), dim=1)
                all_predictions_best.extend(predictions)
                all_labels_best.extend(labels.cpu())
        all_labels_best = [label.item() for label in all_labels_best]
        all_predictions_best = [prediction.item() for prediction in all_predictions_best]
        best_acc = accuracy_score(all_labels_best, all_predictions_best)

    print('fdghsfhgsdjkfhsdhfj')

    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            # writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader)+iter)
            progress_bar.set_description(
                'Epoch: {}/{} Iter: {} Loss: {:.4f}'.format(epoch + 1, epochs, iter + 1, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs.cpu(), dim=1)
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu())
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        acc = accuracy_score(all_labels, all_predictions)
        print('Epoch: {}/{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch + 1, epochs, loss.item(), acc))
        torch.save(model.state_dict(), 'model/last.pt')
        if acc > best_acc:
            torch.save(model.state_dict(), 'model/best.pt')
            best_acc = acc
        # writer.add_scalars('Val/Accuracy', acc, epoch)


