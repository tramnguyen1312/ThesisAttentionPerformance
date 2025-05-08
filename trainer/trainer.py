import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def train(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss, correct, total = 0, 0, 0
    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 100 - 100. * correct / total

def test(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / total, 100 - 100. * correct / total

def name_save_folder(args):
    folder = f"{args.dataset}_{args.optimizer}_lr={args.lr}_bs={args.batch_size}_wd={args.weight_decay}"
    return folder

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.net = model
        self.use_cuda = torch.cuda.is_available()
        self.lr = args.lr
        self.start_epoch = 1
        self.best_acc = 0.0
        self.save_folder = name_save_folder(args)
        os.makedirs(os.path.join(args.save, self.save_folder), exist_ok=True)
        self.log_file = open(os.path.join(args.save, self.save_folder, 'log.out'), 'a')

    def run(self):
        from dataloader import get_data_loaders  # Import here to avoid circular import
        trainloader, testloader = get_data_loaders(self.args)
        init_params(self.net)

        if self.args.ngpu > 1:
            self.net = nn.DataParallel(self.net)
        if self.use_cuda:
            self.net.cuda()

        criterion = nn.CrossEntropyLoss() if self.args.loss_name == 'crossentropy' else nn.MSELoss()
        if self.use_cuda:
            criterion.cuda()

        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay, nesterov=True) \
            if self.args.optimizer == 'sgd' else optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)

        # Initial evaluation
        train_loss, train_err = test(trainloader, self.net, criterion, self.use_cuda)
        test_loss, test_err = test(testloader, self.net, criterion, self.use_cuda)
        self._log(0, train_loss, train_err, test_err, test_loss)

        # Main loop
        for epoch in range(1, self.args.epochs + 1):
            loss, train_err = train(trainloader, self.net, criterion, optimizer, self.use_cuda)
            test_loss, test_err = test(testloader, self.net, criterion, self.use_cuda)
            self._log(epoch, loss, train_err, test_err, test_loss)

            acc = 100 - test_err
            if epoch % self.args.save_epoch == 0 or acc > self.best_acc:
                self.best_acc = max(acc, self.best_acc)
                self._save_checkpoint(optimizer, epoch, acc)

            if epoch in [150, 225, 275]:
                self.lr *= self.args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

        self.log_file.close()

    def _log(self, epoch, train_loss, train_err, test_err, test_loss):
        status = f'e: {epoch} loss: {train_loss:.5f} train_err: {train_err:.3f} test_top1: {test_err:.3f} test_loss {test_loss:.5f}\n'
        print(status)
        self.log_file.write(status)

    def _save_checkpoint(self, optimizer, epoch, acc):
        state = {'acc': acc, 'epoch': epoch, 'state_dict': self.net.module.state_dict() if self.args.ngpu > 1 else self.net.state_dict()}
        opt_state = {'optimizer': optimizer.state_dict()}
        path
