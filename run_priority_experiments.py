#!/usr/bin/env python3
"""
Priority Ablation Experiments - Most Important for Reviewers
Run only the essential experiments to address reviewer feedback
"""

import subprocess
import sys

# Base command template matching your Kaggle format
BASE_CMD = """python train.py \\
  --image_size 224 \\
  --batch_size 64 \\
  --lr 0.005 \\
  --min_lr 1e-6 \\
  --weight_decay 5e-4 \\
  --optimizer SGD \\
  --lr_scheduler ReduceLROnPlateau \\
  --early_stopping_patience 10 \\
  --device cuda \\
  --random_seed 42 \\
  --wandb_key cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b \\
  --dataset_path '/kaggle/input/'"""

def run_cmd(cmd, description):
    """Run command with description"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print("Command:")
    print(cmd)
    print()
    
    # Ask for confirmation
    response = input("Continue with this experiment? (y/n/q): ").lower()
    if response == 'q':
        print("Exiting...")
        sys.exit(0)
    elif response != 'y':
        print("Skipped.")
        return
    
    # Execute
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"✅ COMPLETED: {description}")
    else:
        print(f"❌ FAILED: {description}")

def main():
    print("🎯 PRIORITY ABLATION EXPERIMENTS")
    print("Address key reviewer requirements")
    
    # 1. COMPUTATIONAL ANALYSIS (Fast - 1 epoch)
    print("\n" + "="*60)
    print("📊 SECTION 1: COMPUTATIONAL ANALYSIS (Quick)")
    print("="*60)
    
    experiments = [
        # Computational analysis - fast experiments
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention none "
         f"--enable_computational_analysis --max_epoch 1 "
         f"--wandb_project COMP_ANALYSIS_BASELINE --wandb_run baseline_vgg16",
         "Computational Analysis - Baseline"),
        
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention CBAM "
         f"--enable_computational_analysis --max_epoch 1 "
         f"--wandb_project COMP_ANALYSIS_GLA --wandb_run gla_cbam_vgg16",
         "Computational Analysis - GLA-CBAM"),
        
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention mha_only "
         f"--enable_computational_analysis --max_epoch 1 "
         f"--wandb_project COMP_ANALYSIS_MHA --wandb_run mha_only_vgg16",
         "Computational Analysis - MHA Only"),
        
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention cbam_only "
         f"--enable_computational_analysis --max_epoch 1 "
         f"--wandb_project COMP_ANALYSIS_CBAM --wandb_run cbam_only_vgg16",
         "Computational Analysis - CBAM Only"),
    ]
    
    for cmd, desc in experiments:
        run_cmd(cmd, desc)
    
    # 2. KEY ABLATION STUDIES
    print("\n" + "="*60)
    print("🧪 SECTION 2: CORE ABLATION STUDIES")
    print("="*60)
    
    ablation_experiments = [
        # MHA Only vs Full GLA-Block
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention mha_only "
         f"--max_epoch 90 --wandb_project ABLATION_MHA_ONLY --wandb_run mha_vgg16_ham",
         "Ablation: MHA Only (HAM10000, VGG16)"),
        
        # Lightweight Only 
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention cbam_only "
         f"--max_epoch 90 --wandb_project ABLATION_CBAM_ONLY --wandb_run cbam_vgg16_ham",
         "Ablation: CBAM Only (HAM10000, VGG16)"),
        
        # Fusion Comparison
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention CBAM --fusion_type concat "
         f"--max_epoch 90 --wandb_project ABLATION_FUSION_CONCAT --wandb_run concat_vgg16_ham",
         "Ablation: Concatenation Fusion (HAM10000, VGG16)"),
    ]
    
    for cmd, desc in ablation_experiments:
        run_cmd(cmd, desc)
    
    # 3. CROSS-DATASET EVALUATION
    print("\n" + "="*60)
    print("🔄 SECTION 3: CROSS-DATASET EVALUATION")
    print("="*60)
    
    cross_dataset = [
        (f"{BASE_CMD} --dataset HAM10000 --backbone VGG16 --attention CBAM "
         f"--cross_dataset_eval --max_epoch 30 "
         f"--wandb_project CROSS_DATASET_EVAL --wandb_run cross_eval_gla",
         "Cross-dataset Evaluation"),
    ]
    
    for cmd, desc in cross_dataset:
        run_cmd(cmd, desc)
    
    print("\n🎉 Priority experiments completed!")
    print("Check computational_analysis_*.json and cross_dataset_results_*.json files")

if __name__ == "__main__":
    main()