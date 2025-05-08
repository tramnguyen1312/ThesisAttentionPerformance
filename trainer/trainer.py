import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    LambdaLR
)
import math
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tqdm

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs]

class EarlyStopping:
    def __init__(self, patience=7, delta=0, monitor='val_acc'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class DatasetTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        configs: dict
    ):
        self.configs = configs
        self.device = torch.device(configs.get('device', 'cuda'))
        if configs.get('multi_gpu', False) and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if configs.get('freeze_backbone', False):
            for name, p in model.named_parameters():
                if 'stem' in name or 'features1' in name:
                    p.requires_grad = False
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss(label_smoothing=configs.get('label_smoothing', 0.0)).to(self.device)
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith('.bias') or 'norm' in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        self.optimizer = optim.AdamW([
            {'params': decay, 'weight_decay': configs.get('weight_decay', 1e-4)},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=configs.get('lr', 1e-3), betas=(0.9,0.999), eps=1e-8)

        sched_choice = configs.get('lr_scheduler', None)
        if sched_choice == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5,
                                              factor=0.5, min_lr=configs.get('min_lr',1e-6), verbose=True)
        elif sched_choice == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=configs.get('step_size',30),
                                    gamma=configs.get('step_gamma',0.1))
        elif sched_choice == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=configs.get('t_max',50),
                                               eta_min=configs.get('min_lr',1e-6), verbose=True)
        elif sched_choice == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=configs.get('t_0',10), T_mult=configs.get('t_mult',2),
                eta_min=configs.get('min_lr',1e-6), verbose=True)
        elif sched_choice == 'OneCycleLR':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=configs.get('lr',1e-3),
                steps_per_epoch=len(self.train_loader),
                epochs=configs.get('max_epochs',100),
                pct_start=configs.get('pct_start',0.3),
                anneal_strategy='cos',
                div_factor=configs.get('div_factor',25),
                final_div_factor=configs.get('final_div_factor',1e4)
            )
        elif sched_choice == 'CosineWarmup':
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=configs.get('warmup_epochs',5),
                max_epochs=configs.get('max_epochs',100),
                min_lr=configs.get('min_lr',1e-6)
            )
        elif sched_choice == 'LambdaLR':
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda e: 1.0 if e>=5 else 0.1+0.9*e/5)
        else:
            self.scheduler = None

        self.use_amp = configs.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.early_stopper = EarlyStopping(patience=configs.get('early_stopping_patience',10),
                                           monitor='val_acc')
        self.checkpoint_path = configs.get('checkpoint_path','checkpoint.pth')
        self.start_epoch = 1
        self._maybe_resume()

    def _maybe_resume(self):
        if self.configs.get('resume_checkpoint', False) and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.scheduler and ckpt.get('scheduler'):
                self.scheduler.load_state_dict(ckpt['scheduler'])
            self.start_epoch = ckpt.get('epoch',1) + 1

    def train(self):
        best_acc = 0.0
        total_epochs = self.configs.get('max_epochs', 100)
        for epoch in range(self.start_epoch, total_epochs + 1):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            self.model.train()
            loop = tqdm.tqdm(self.train_loader, desc=f"Epoch [{epoch}/{total_epochs}] Training", leave=False, ncols=100)
            for imgs, labels in loop:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                loop.set_postfix(loss=train_loss / ((loop.n + 1)), acc=100 * train_correct / train_total)
            train_loss_epoch = train_loss / len(self.train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss, val_acc = self._validate_with_progress(epoch, total_epochs)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            print(f"Epoch {epoch}: Train accuracy={train_acc:.2f}%, Train loss={train_loss_epoch}, Val accuracy={val_acc:.2f}%, Val loss={val_loss}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'epoch': epoch,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict() if self.scheduler else None},
                           self.checkpoint_path)
                print(f"Saved best model at epoch {epoch}")
            self.early_stopper(val_acc)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break
        print(f"Best ValAcc={best_acc:.2f}%")

    def _validate_with_progress(self, epoch, total_epochs):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm.tqdm(self.val_loader, desc=f"Epoch [{epoch}/{total_epochs}] Validation", leave=False, ncols=100)
        with torch.no_grad():
            for imgs, labels in loop:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix(loss=val_loss / ((loop.n + 1)), acc=100 * correct / total)
        return val_loss / len(self.val_loader), 100 * correct / total

    def test(self):
        """Evaluate and save confusion matrix CSV."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        print(f"Test Accuracy: {acc:.2f}%")
        cm = confusion_matrix(all_labels, all_preds)
        csv_path = self.configs.get('confusion_csv_path', 'confusion_matrix.csv')
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + [str(i) for i in range(cm.shape[1])])
            for i, row in enumerate(cm):
                writer.writerow([str(i)] + row.tolist())
        print(f"Confusion matrix saved to {csv_path}")
        return acc

    # separate plotting function
    def plot_confusion_from_csv(csv_path, class_names=None):
        """Load confusion matrix from CSV and plot."""
        cm = np.loadtxt(csv_path, delimiter=',', dtype=int, skiprows=1, usecols=range(1, None))
        labels = class_names if class_names else [str(i) for i in range(cm.shape[0])]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
