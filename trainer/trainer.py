import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
import tqdm
from torch.utils.data import DataLoader
import wandb  # Import wandb for logging
import csv
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(epoch):
    if epoch < 5:  # Warmup
        return 0.1 + 0.9 * epoch / 5
    return 1.0
import math

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
class Trainer:
    """Base trainer class."""

    def __init__(self):
        pass


class DatasetTrainer(Trainer):
    """General Dataset Trainer with WandB integration."""

    def __init__(self, model, train_loader, val_loader, test_loader, configs, wb=False):
        """
        Initialize the trainer.
        :param model: PyTorch model.
        :param train_loader: DataLoader for training set.
        :param val_loader: DataLoader for validation set.
        :param test_loader: DataLoader for testing set.
        :param configs: Configuration dictionary with hyperparameters.
        :param wb: Boolean to enable WandB logging (default False).
        """
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.configs = configs
        self.wb = wb

        # Device setup
        self.device = torch.device(configs["device"])
        self.model = model.to(self.device)

        # Hyperparameters
        self.batch_size = configs["batch_size"]
        self.learning_rate = configs["lr"]
        self.min_lr = configs["min_lr"]
        self.weight_decay = configs["weight_decay"]
        self.optimizer_choice = configs["optimizer"]
        self.scheduler_choice = configs["lr_scheduler"]
        self.max_epochs = configs["max_epoch_num"]
        self.best_val_acc = 0.0

        # Optimizer, Scheduler, and Loss
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(self.device)

        # Checkpoint and Logging
        self.start_time = datetime.datetime.now()
        self.checkpoint_path = configs.get("checkpoint_path", "best_model.pth")
        self.early_stopping_patience = configs.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        self.csv_log_path = configs.get("csv_log_path", "training_log.csv")
        self._initialize_csv_log()

        # WandB setup
        if self.wb:
            self._initialize_wandb()

    def _initialize_csv_log(self):
        # Open CSV log file and write header
        with open(self.csv_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate"
            ])

    def _initialize_optimizer(self):
        """Set up the optimizer."""
        if self.optimizer_choice == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_choice == "RAdam":
            return torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_choice == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_choice}.")

    def _initialize_scheduler(self):
        """Set up the learning rate scheduler."""
        if self.scheduler_choice == "ReduceLROnPlateau":
            return ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, min_lr=self.min_lr, verbose=True)
        elif self.scheduler_choice == "StepLR":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_choice == "CosineAnnealingLR":
            return CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.min_lr, verbose=True)
        elif self.scheduler_choice == "CosineAnnealingWarmRestarts":
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=self.min_lr, verbose=True)
        elif self.scheduler_choice == "LambdaLR":
            return LambdaLR(self.optimizer, lr_lambda)
        elif self.scheduler_choice == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=len(self.train_loader),
                epochs=self.max_epochs,
                pct_start=0.3
            )
        elif self.scheduler_choice == "CosineWarmup":
            warmup_epochs = self.configs.get("warmup_epochs", 5)
            return CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=self.max_epochs,
                min_lr=self.min_lr
            )
        else:
            print("No learning rate scheduler selected.")
            return None

    def _initialize_wandb(self):
        """Initialize WandB."""
        wandb.login(key=self.configs["wandb_api_key"])
        wandb.init(
            project=self.configs["project_name"],
            name=self.configs.get("run_name", f"run_{datetime.datetime.now()}"),
            config=self.configs,
        )
        wandb.watch(self.model, log="all", log_freq=10)
        print("WandB initialized.")

    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}/{self.max_epochs} [Training]",
                leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        # Log to WandB
        if self.wb:
            wandb.log({"Train Loss": avg_loss, "Train Accuracy": accuracy})

        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def validate_one_epoch(self, epoch):
        """Validate the model for one epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    desc=f"Epoch {epoch}/{self.max_epochs} [Validation]",
                    leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Log to WandB
        if self.wb:
            wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})

        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def test_model(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm.tqdm(
                    self.test_loader, total=len(self.test_loader), desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total

        # Log to WandB
        if self.wb:
            wandb.log({"Test Accuracy": accuracy})

        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def train(self):
        train_hist = {"loss": [], "accuracy": []}
        val_hist = {"loss": [], "accuracy": []}
        """Main training loop."""
        print(self.configs)

        early_stopper = EarlyStopping(patience=self.early_stopping_patience)

        for epoch in range(1, self.max_epochs + 1):
            print(f"\nEpoch {epoch}/{self.max_epochs}")
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate_one_epoch(epoch)
            # Append training history
            train_hist["loss"].append(train_loss)
            train_hist["accuracy"].append(train_acc)
            val_hist["loss"].append(val_loss)
            val_hist["accuracy"].append(val_acc)

            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            #Write log
            current_lr = self.optimizer.param_groups[0]['lr']
            with open(self.csv_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    current_lr
                ])


            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with accuracy: {val_acc:.2f}%")
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered!")
                break

        # Training Summary
        print("\nTraining Summary:")
        print(f"Start Time: {self.start_time}")
        print(f"End Time: {datetime.datetime.now()}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(
            f"Final Training Loss: {train_hist['loss'][-1]:.4f}, Final Training Accuracy: {train_hist['accuracy'][-1]:.2f}%")
        print(
            f"Final Validation Loss: {val_hist['loss'][-1]:.4f}, Final Validation Accuracy: {val_hist['accuracy'][-1]:.2f}%")

        # # Log summary to WandB
        # if self.wb:
        #     wandb.log({
        #         "Best Validation Accuracy": self.best_val_acc,
        #         "Final Training Loss": train_hist["loss"][-1],
        #         "Final Training Accuracy": train_hist["accuracy"][-1],
        #         "Final Validation Loss": val_hist["loss"][-1],
        #         "Final Validation Accuracy": val_hist["accuracy"][-1],
        #     })

        print(f"Training complete. Best validation accuracy: {self.best_val_acc:.2f}%")
