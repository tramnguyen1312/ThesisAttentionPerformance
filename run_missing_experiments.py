#!/usr/bin/env python3
"""
Script to run missing experiments for ablation study
Based on analysis of existing WandB runs
"""

import subprocess
import time
import json
from datetime import datetime

def run_experiment(cmd_args, description=""):
    """Run a single experiment with given arguments"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"{'='*60}")
    
    # Build command
    cmd = ['python', 'train.py'] + cmd_args
    
    # Print command for verification
    print("Command:", ' '.join(cmd))
    print()
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"✅ COMPLETED in {duration/60:.1f} minutes: {description}")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ FAILED after {duration/60:.1f} minutes: {description}")
        print(f"Error: {e}")
        return False

def main():
    print("🎯 MISSING EXPERIMENTS FOR ABLATION STUDY")
    print("=" * 60)
    
    # Based on analysis, these experiments are missing:
    missing_experiments = [
        # 1. scSE Only (failed in original run)
        {
            'args': [
                '--dataset', 'HAM10000',
                '--backbone', 'VGG16', 
                '--attention', 'scse_only',
                '--batch_size', '64',
                '--lr', '0.005',
                '--min_lr', '1e-6',
                '--weight_decay', '5e-4',
                '--optimizer', 'SGD',
                '--lr_scheduler', 'ReduceLROnPlateau',
                '--max_epoch', '90',
                '--early_stopping_patience', '10',
                '--device', 'cuda',
                '--random_seed', '42',
                '--wandb_key', 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
                '--dataset_path', '/kaggle/input/ham10000/HAM10000',
                '--wandb_project', 'GLSBLOCK_ABLATION_VGG16_scse_only_HAM10000',
                '--wandb_run', 'ablation_scse_only_VGG16_ham10000_full'
            ],
            'description': 'Ablation Study - scSE Only (VGG16, HAM10000)'
        },
        
        # 2. Cross-dataset evaluation (failed in original run)  
        {
            'args': [
                '--dataset', 'HAM10000',
                '--backbone', 'VGG16',
                '--attention', 'CBAM',
                '--cross_dataset_eval',
                '--batch_size', '64',
                '--lr', '0.005',
                '--min_lr', '1e-6',
                '--weight_decay', '5e-4',
                '--optimizer', 'SGD',
                '--lr_scheduler', 'ReduceLROnPlateau',
                '--max_epoch', '30',
                '--early_stopping_patience', '5',
                '--device', 'cuda',
                '--random_seed', '42',
                '--wandb_key', 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
                '--dataset_path', '/kaggle/input/ham10000/HAM10000',
                '--wandb_project', 'CROSS_DATASET_EVAL',
                '--wandb_run', 'cross_eval_gla_cbam'
            ],
            'description': 'Cross-dataset Evaluation - GLA-Block CBAM'
        },
        
        # 3. ISIC-2018 experiments (failed in original run)
        {
            'args': [
                '--dataset', 'isic-2018-task-3',
                '--backbone', 'VGG16',
                '--attention', 'mha_only',
                '--batch_size', '64',
                '--lr', '0.005',
                '--min_lr', '1e-6', 
                '--weight_decay', '5e-4',
                '--optimizer', 'SGD',
                '--lr_scheduler', 'ReduceLROnPlateau',
                '--max_epoch', '90',
                '--early_stopping_patience', '10',
                '--device', 'cuda',
                '--random_seed', '42',
                '--wandb_key', 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
                '--dataset_path', '/kaggle/input/isic-2018-task-3',
                '--wandb_project', 'GLSBLOCK_ABLATION_VGG16_mha_only_ISIC2018',
                '--wandb_run', 'ablation_mha_only_VGG16_isic2018_full'
            ],
            'description': 'Ablation Study - MHA Only (VGG16, ISIC-2018)'
        },
        
        # 4. Full GLA-Block variants for comparison
        {
            'args': [
                '--dataset', 'HAM10000',
                '--backbone', 'VGG16',
                '--attention', 'CBAM',
                '--fusion_type', 'multiply',
                '--batch_size', '64',
                '--lr', '0.005',
                '--min_lr', '1e-6',
                '--weight_decay', '5e-4',
                '--optimizer', 'SGD',
                '--lr_scheduler', 'ReduceLROnPlateau',
                '--max_epoch', '90',
                '--early_stopping_patience', '10',
                '--device', 'cuda',
                '--random_seed', '42',
                '--wandb_key', 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
                '--dataset_path', '/kaggle/input/ham10000/HAM10000',
                '--wandb_project', 'GLSBLOCK_ABLATION_VGG16_CBAM_HAM10000_multiply',
                '--wandb_run', 'ablation_CBAM_VGG16_ham10000_multiply_full'
            ],
            'description': 'GLA-CBAM with Multiply Fusion (VGG16, HAM10000)'
        },
        
        # 5. BAM variants
        {
            'args': [
                '--dataset', 'HAM10000',
                '--backbone', 'VGG16',
                '--attention', 'BAM',
                '--fusion_type', 'concat',
                '--batch_size', '64',
                '--lr', '0.005',
                '--min_lr', '1e-6',
                '--weight_decay', '5e-4',
                '--optimizer', 'SGD',
                '--lr_scheduler', 'ReduceLROnPlateau',
                '--max_epoch', '90',
                '--early_stopping_patience', '10',
                '--device', 'cuda',
                '--random_seed', '42',
                '--wandb_key', 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
                '--dataset_path', '/kaggle/input/ham10000/HAM10000',
                '--wandb_project', 'GLSBLOCK_ABLATION_VGG16_BAM_HAM10000_concat',
                '--wandb_run', 'ablation_BAM_VGG16_ham10000_concat_full'
            ],
            'description': 'GLA-BAM with Concatenation Fusion (VGG16, HAM10000)'
        }
    ]
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'experiments': [],
        'summary': {'success': 0, 'failed': 0}
    }
    
    # Run experiments
    for i, exp in enumerate(missing_experiments, 1):
        print(f"\n📋 EXPERIMENT {i}/{len(missing_experiments)}")
        success = run_experiment(exp['args'], exp['description'])
        
        results['experiments'].append({
            'experiment': exp['description'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            results['summary']['success'] += 1
        else:
            results['summary']['failed'] += 1
            
        # Small delay between experiments
        if i < len(missing_experiments):
            print("⏳ Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    # Save results summary
    results['end_time'] = datetime.now().isoformat()
    with open('missing_experiments_log.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 MISSING EXPERIMENTS COMPLETED")
    print("="*60)
    print(f"✅ Successful: {results['summary']['success']}")
    print(f"❌ Failed: {results['summary']['failed']}")
    print(f"📊 Total: {len(missing_experiments)} experiments")
    print(f"📋 Log saved to: missing_experiments_log.json")
    
    if results['summary']['failed'] > 0:
        print("\n⚠️  Some experiments failed. Check the log for details.")
        print("You may need to rerun failed experiments manually.")
    else:
        print("\n🎯 All missing experiments completed successfully!")
        print("Ablation study data is now complete!")

if __name__ == "__main__":
    main()