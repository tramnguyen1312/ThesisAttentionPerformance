import argparse
import torch
from trainer import DatasetTrainer
from backbone import ResNet18, VGG16
from attention import CBAMBlock, BAMBlock, scSEBlock
from datasets import GeneralDataset
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import random, numpy as np, torch
from torch.utils.data import WeightedRandomSampler
import time
import psutil
import os

# For computational analysis
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with 'pip install thop' for FLOP analysis.")

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across modules.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_arguments():
    """  
    Parse arguments for training configuration.  
    """
    parser = argparse.ArgumentParser(
        description="Train model with various backbones, attention mechanisms, and configurations.")

    # Dataset and model arguments  
    parser.add_argument("--dataset", type=str, default="HAM10000",
                        choices=["STL10", "Caltech101", "Caltech256", "Oxford-IIIT Pets", "HAM10000", "isic-2018-task-3"],
                        help="Choose dataset to train on (default: STL10)")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for resizing (default: 224)")
    parser.add_argument("--backbone", type=str, default="VGG16",
                        choices=["VGG16", "ResNet18"],
                        help="Choose the backbone model (default: VGG16)")
    parser.add_argument("--attention", type=str, default="CBAM",
                        choices=["CBAM", "BAM", "scSE", "none", "mha_only", "cbam_only", "bam_only", "scse_only"],
                        help="Choose an attention mechanism or none (default: CBAM)")
    
    # Ablation study arguments (based on reviewer feedback)
    parser.add_argument("--fusion_type", type=str, default="multiply",
                        choices=["multiply", "concat"],
                        help="Fusion method for GLA-Block: multiply (A*L) or concat ([A,L]) (default: multiply)")
    parser.add_argument("--enable_computational_analysis", action="store_true",
                        help="Enable computational cost analysis (FLOPs, inference time, memory)")
    parser.add_argument("--cross_dataset_eval", action="store_true",
                        help="Enable cross-dataset evaluation for generalization analysis")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader (default: 0)")

    # Training hyperparameters  
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate (default: 1e-7)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--optimizer", type=str, default="RAdam",
                        choices=["RAdam", "Adam", "SGD", "AdamW"], help="Optimizer to use (default: RAdam)")
    parser.add_argument("--lr_scheduler", type=str, default="ReduceLROnPlateau",
                        choices=["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "CosineWarmup","None"],
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
    parser.add_argument("--wandb_key", type=str, default="run-v1",
                        help="WandB run api key")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth",
                        help="Path to save the best model (default: best_model.pth)")
    parser.add_argument("--pre_train", action="store_true",help="Enable pre-training mode")
    parser.add_argument("--dataset_path", type=str, default="best_model.pth",
                        help="Path to save the best model (default: best_model.pth)")

    return parser.parse_args()


def measure_computational_cost(model, input_tensor, device):
    """
    Measure FLOPs, parameters, inference time, and memory usage
    """
    results = {}
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # 1. FLOPs and Parameters
    if THOP_AVAILABLE:
        try:
            # Clone model for FLOP analysis to avoid modifying original
            model_copy = type(model)(
                attn_type=getattr(model, 'attn_type', 'none'),
                num_heads=8,
                pretrained=False,
                num_classes=model.fc.out_features if hasattr(model, 'fc') else model.classifier[-1].out_features,
                fusion_type=getattr(model, 'fusion_type', 'multiply') if hasattr(model, 'fusion_type') else 'multiply'
            )
            model_copy.eval()
            
            flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            results['flops'] = flops
            results['flops_formatted'] = flops_formatted
            results['params'] = params
            results['params_formatted'] = params_formatted
        except Exception as e:
            print(f"FLOP analysis failed: {e}")
            results['flops'] = "N/A"
            results['flops_formatted'] = "N/A"
            
    # Manual parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    
    # 2. Inference Time
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Measure inference time
    num_runs = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    results['inference_time_ms'] = avg_inference_time
    
    # 3. Memory Usage
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # Convert to MB
        results['memory_usage_mb'] = memory_used
        torch.cuda.reset_peak_memory_stats(device)
    else:
        # CPU memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 ** 2)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        final_memory = process.memory_info().rss / (1024 ** 2)
        memory_used = final_memory - initial_memory
        results['memory_usage_mb'] = memory_used
    
    return results


def run_cross_dataset_evaluation(args, model, device):
    """
    Cross-dataset evaluation: train on one dataset, test on another
    """
    print("\n=== CROSS-DATASET EVALUATION ===")
    
    # Available datasets
    datasets = ["HAM10000", "isic-2018-task-3"]
    
    results = {}
    
    for train_dataset_name in datasets:
        for test_dataset_name in datasets:
            if train_dataset_name == test_dataset_name:
                continue
                
            print(f"\nTrain on {train_dataset_name}, Test on {test_dataset_name}")
            
            # Load train dataset
            train_dataset_obj = GeneralDataset(train_dataset_name, args.dataset_path)
            train_data, val_data = train_dataset_obj.get_splits(
                val_size=0.2, seed=args.random_seed, image_size=args.image_size
            )
            
            # Load test dataset  
            test_dataset_obj = GeneralDataset(test_dataset_name, args.dataset_path)
            _, test_data = test_dataset_obj.get_splits(
                val_size=0.2, seed=args.random_seed, image_size=args.image_size
            )
            
            # Create model with appropriate number of classes
            if args.backbone == "VGG16":
                cross_model = VGG16(
                    pretrained=args.pre_train, 
                    attn_type=args.attention, 
                    num_heads=8, 
                    num_classes=train_dataset_obj.num_classes,
                    fusion_type=args.fusion_type
                )
            else:
                cross_model = ResNet18(
                    pretrained=args.pre_train, 
                    attn_type=args.attention, 
                    num_heads=8, 
                    num_classes=train_dataset_obj.num_classes,
                    fusion_type=args.fusion_type
                )
            
            # Create data loaders
            train_labels = [train_data.lbls[i] for i in range(len(train_data))]
            class_counts = np.bincount(train_labels, minlength=train_data.num_classes)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in train_labels]
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            
            train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            
            # Train model
            configs = {
                "device": device,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
                "lr_scheduler": args.lr_scheduler,
                "max_epoch_num": min(args.max_epoch, 20),  # Reduced epochs for cross-dataset
                "checkpoint_path": f"cross_{train_dataset_name}_{test_dataset_name}.pth",
                "wandb_api_key": None,
                "project_name": f"cross-dataset",
                "run_name": f"{train_dataset_name}-to-{test_dataset_name}",
                "early_stopping_patience": 5,
            }
            
            trainer = DatasetTrainer(cross_model, train_loader, test_loader, test_loader, configs, wb=False)
            trainer.train()
            
            # Store results
            key = f"{train_dataset_name}_to_{test_dataset_name}"
            results[key] = {
                'train_dataset': train_dataset_name,
                'test_dataset': test_dataset_name,
                'accuracy': trainer.best_acc if hasattr(trainer, 'best_acc') else 0.0
            }
            
            print(f"Cross-dataset result: {results[key]['accuracy']:.2f}%")
    
    return results


def main():
    # Parse arguments  
    args = parse_arguments()

    random.seed(args.random_seed)

    # Print configurations for debugging  
    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

        # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Initialize train and test datasets
    dataset = GeneralDataset(args.dataset, args.dataset_path)
    train_dataset, test_dataset = dataset.get_splits(val_size=0.2, seed=args.random_seed, image_size=args.image_size)

    print(f"Total images in the train dataset: {len(train_dataset)}")
    print(f"Total images in the test dataset: {len(test_dataset)}")
    print(f"Total classes: {dataset.num_classes}")

    # Lấy danh sách nhãn từ train_ds
    train_labels = [train_dataset.lbls[i] for i in range(len(train_dataset))]

    class_counts = np.bincount(train_labels, minlength=train_dataset.num_classes)
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[label] for label in train_labels]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # = len(train_ds)
        replacement=True
    )

    # DataLoader cho train với sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers
    )

    #Create DataLoaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers
    # )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
     # Select backbone model
    model = None
    if args.backbone == "VGG16":
        model = VGG16(pretrained=args.pre_train, attn_type=args.attention, num_heads=8, num_classes=dataset.num_classes, fusion_type=args.fusion_type)
    elif args.backbone == "ResNet18":
        model = ResNet18(pretrained=args.pre_train, attn_type=args.attention, num_heads=8, num_classes=dataset.num_classes, fusion_type=args.fusion_type)
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
        "wandb_api_key": args.wandb_key,
        "project_name": args.wandb_project,
        "run_name": args.wandb_run,
        "early_stopping_patience": args.early_stopping_patience,
    }
    if model is not None:
        print(model)
        
        # Computational Analysis (if enabled)
        if args.enable_computational_analysis:
            print("\n=== COMPUTATIONAL ANALYSIS ===")
            input_tensor = torch.randn(1, 3, args.image_size, args.image_size)
            device_obj = torch.device(device)
            
            comp_results = measure_computational_cost(model, input_tensor, device_obj)
            
            print(f"Model: {args.backbone} + {args.attention}")
            print(f"Total Parameters: {comp_results['total_params']:,}")
            print(f"Trainable Parameters: {comp_results['trainable_params']:,}")
            
            if THOP_AVAILABLE and 'flops_formatted' in comp_results:
                print(f"FLOPs: {comp_results['flops_formatted']}")
                print(f"Parameters (thop): {comp_results['params_formatted']}")
            
            print(f"Average Inference Time: {comp_results['inference_time_ms']:.2f} ms")
            print(f"Memory Usage: {comp_results['memory_usage_mb']:.2f} MB")
            
            # Save computational results
            import json
            comp_filename = f"computational_analysis_{args.backbone}_{args.attention}_{args.fusion_type}.json"
            with open(comp_filename, 'w') as f:
                json.dump(comp_results, f, indent=2, default=str)
            print(f"Computational analysis saved to {comp_filename}")
        
        # Cross-dataset evaluation (if enabled) 
        if args.cross_dataset_eval:
            cross_results = run_cross_dataset_evaluation(args, model, torch.device(device))
            
            # Save cross-dataset results
            import json
            cross_filename = f"cross_dataset_results_{args.backbone}_{args.attention}.json"
            with open(cross_filename, 'w') as f:
                json.dump(cross_results, f, indent=2)
            print(f"Cross-dataset results saved to {cross_filename}")
        
        # Regular training
        trainer = DatasetTrainer(model, train_loader, test_loader, test_loader, configs, wb=True)
        trainer.train()
        
        # Print final summary
        print("\n=== TRAINING SUMMARY ===")
        print(f"Best Accuracy: {trainer.best_acc if hasattr(trainer, 'best_acc') else 'N/A'}%")
        print(f"Model Configuration: {args.backbone} + {args.attention}")
        if args.fusion_type != 'multiply':
            print(f"Fusion Type: {args.fusion_type}")
        
    else:
        raise "Model is none"


if __name__ == "__main__":
    main()
