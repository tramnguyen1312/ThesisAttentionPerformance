import torch
import os
import csv

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dataloader, val_dataloader,
                 criterion, max_epochs, max_plateau_count, min_lr, checkpoint_path,
                 output_csv_path, debug=False):
        # Model, Optimizer, Scheduler, Loss Function
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        # Dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Training settings
        self.max_epochs = max_epochs
        self.max_plateau_count = max_plateau_count
        self.min_lr = min_lr
        self.debug = debug

        # Paths
        self.checkpoint_path = checkpoint_path
        self.output_csv_path = output_csv_path

        # Training state tracking
        self.current_epoch_num = 0
        self.plateau_count = 0
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.best_train_acc = 0
        self.best_train_loss = float('inf')
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.test_acc = 0
        self.test_acc_ttau = 0

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # Zero optimizer gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            self.optimizer.step()

            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Logging for training epoch
        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        epoch_acc = correct / total
        self.train_loss_list.append(epoch_loss)
        self.train_acc_list.append(epoch_acc)
        print(f"Train Epoch {self.current_epoch_num}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

    def validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Update metrics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Logging for validation epoch
        val_loss = running_loss / len(self.val_dataloader.dataset)
        val_acc = correct / total
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        print(f"Validation Epoch {self.current_epoch_num}: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")
        return val_loss, val_acc

    def save_weights(self):
        state_dict = self.model.state_dict()

        state = {
            "net": state_dict,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_train_loss": self.best_train_loss,
            "best_train_acc": self.best_train_acc,
            "train_loss_list": self.train_loss_list,
            "val_loss_list": self.val_loss_list,
            "train_acc_list": self.train_acc_list,
            "val_acc_list": self.val_acc_list,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, self.checkpoint_path)
        print(f"Model weights saved to {self.checkpoint_path}")

    def update_output_csv(self, epoch, lr, accuracy, loss, val_accuracy, val_loss):
        file_exists = os.path.isfile(self.output_csv_path)
        with open(self.output_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                # Write header once
                writer.writerow(['epoch', 'lr', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])
            writer.writerow([epoch, lr, accuracy, loss, val_accuracy, val_loss])

    def update_state_training(self):
        # Check for accuracy improvement
        if self.val_acc_list[-1] > self.best_val_acc:
            self.save_weights()
            self.plateau_count = 0
            self.best_val_acc = self.val_acc_list[-1]
            self.best_val_loss = self.val_loss_list[-1]
            self.best_train_acc = self.train_acc_list[-1]
            self.best_train_loss = self.train_loss_list[-1]
            print(f"Weight updated due to improved validation accuracy: {self.best_val_acc:.4f}")
        else:
            self.plateau_count += 1

        # Log CSV
        self.update_output_csv(
            epoch=self.current_epoch_num,
            lr=self.optimizer.param_groups[0]['lr'],
            accuracy=self.train_acc_list[-1],
            loss=self.train_loss_list[-1],
            val_accuracy=self.val_acc_list[-1],
            val_loss=self.val_loss_list[-1]
        )

        # Update learning rate scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self.val_acc_list[-1])
        else:
            self.scheduler.step()

        # Ensure LR doesn't fall below min_lr
        if self.optimizer.param_groups[0]['lr'] < self.min_lr:
            self.optimizer.param_groups[0]['lr'] = self.min_lr

    def update_epoch_num(self):
        self.current_epoch_num += 1

    def stop_train(self):
        return (
            self.plateau_count > self.max_plateau_count or
            self.current_epoch_num > self.max_epochs
        )

    def fit(self):
        print("Training started...\n")
        for epoch in range(self.max_epochs):
            self.update_epoch_num()
            print(f"Epoch {self.current_epoch_num}/{self.max_epochs}:")

            self.train_one_epoch()
            val_loss, val_acc = self.validate_one_epoch()
            self.update_state_training()

            # Check stopping condition
            if self.stop_train():
                print("Early stopping triggered.")
                break

        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")