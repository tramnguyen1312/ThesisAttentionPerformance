import argparse
import torch
from trainer import DatasetTrainer
from backbone import ResNet18, VGG16
from attention import CBAMBlock, BAMBlock, scSEBlock, HMHA_CBAM
from datasets import GeneralDataset
from torch.utils.data import DataLoader



def parse_arguments():
    """  
    Parse arguments for training configuration.  
    """
    parser = argparse.ArgumentParser(
        description="Train model with various backbones, attention mechanisms, and configurations.")

    # Dataset and model arguments  
    parser.add_argument("--dataset", type=str, default="STL10",
                        choices=["STL10", "Caltech101", "Caltech256", "Oxford-IIIT Pets"],
                        help="Choose dataset to train on (default: STL10)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for resizing (default: 224)")
    parser.add_argument("--backbone", type=str, default="VGG16",
                        choices=["VGG16", "ResNet18"],
                        help="Choose the backbone model (default: VGG16)")
    parser.add_argument("--attention", type=str, default="MHA_CBAM",
                        choices=["CBAM", "BAM", "scSE", "none", "MHA_CBAM"],
                        help="Choose an attention mechanism or none (default: CBAM)")
    parser.add_argument("--num_workers", type=int, default=0,  
                        help="Number of workers for DataLoader (default: 0)")

    # Training hyperparameters  
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate (default: 1e-7)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--optimizer", type=str, default="RAdam",
                        choices=["RAdam", "Adam", "SGD", "AdamW"], help="Optimizer to use (default: RAdam)")
    parser.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau",
                        choices=["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "None"],
                        help="Learning rate scheduler to use (default: ReduceLROnPlateau)")
    parser.add_argument("--max_epoch", type=int, default=10, help="Maximum number of training epochs (default: 10)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Maximum number of training epochs (default: 10)")

    # Device and reproducibility  
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device for training (default: cuda)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Logging and checkpoints  
    parser.add_argument("--wandb_project", type=str, default="test-project",
                        help="WandB project name (default: test-project)")
    parser.add_argument("--wandb_run", type=str, default="run-v1",
                        help="WandB run name (default: run-v1)")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Path to save the best model (default: best_model.pth)")
    parser.add_argument("--pre_train", action="store_true",help="Enable pre-training mode")

    return parser.parse_args()


def main():
    # Parse arguments  
    args = parse_arguments()

    # Print configurations for debugging  
    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

        # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Initialize train and test datasets  
    train_dataset = GeneralDataset(
        data_type="train",
        dataset_name=args.dataset,
        image_size=args.image_size,
        image_path="./datasets/datasets",
        random_seed=args.random_seed
    )
    test_dataset = GeneralDataset(
        data_type="test",
        dataset_name=args.dataset,
        image_size=args.image_size,
        image_path="./datasets/datasets",
        random_seed=args.random_seed
    )
    print(f"Total images in the train dataset: {len(train_dataset)}")
    print(f"Total images in the test dataset: {len(test_dataset)}")
    print(f"Total classes: {train_dataset.num_classes}")

    # Create DataLoaders  
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    if args.backbone == "VGG16":
        backbone_channels = 512
    elif args.backbone == "ResNet18":
        backbone_channels = 128
    attention_module = None
    # Select attention mechanism  
    if args.attention == "CBAM":
        attention_module = CBAMBlock(channel=backbone_channels, reduction=16, kernel_size=7)
    elif args.attention == "BAM":
        attention_module = BAMBlock(channel=backbone_channels, reduction=16, dia_val=2)
    elif args.attention == "scSE":
        attention_module = scSEBlock(channel=backbone_channels)
    elif args.attention == "none":
        attention_module = None
    elif args.attention == "MHA_CBAM":
        attention_module = HMHA_CBAM(channel=512, num_heads=8, reduction=16, kernel_size=7)

     # Select backbone model
    model = None
    if args.backbone == "VGG16":
        model = VGG16(pretrained=args.pre_train, attention=attention_module, num_classes=train_dataset.num_classes)
    elif args.backbone == "ResNet18":
        model = ResNet18(pretrained=args.pre_train, attention=attention_module, num_classes=train_dataset.num_classes)

        # Training configurations
    configs = {
        "device": device,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "lr_scheduler": args.lr_scheduler,
        "max_epoch_num": args.max_epoch,
        "checkpoint_path": args.checkpoint_path,
        "wandb_api_key": "94c506f92aea3ff024c35621dc85e7ee75194d12",
        "project_name": args.wandb_project,
        "run_name": args.wandb_run,
        "early_stopping_patience": args.early_stopping_patience,
    }
    if model is not None:
        print(model)
        # Initialize trainer
        trainer = DatasetTrainer(model, train_loader, test_loader, test_loader, configs, wb=True)

        # Start training
        trainer.train()
    else:
        raise "Model is none"


if __name__ == "__main__":
    main()
