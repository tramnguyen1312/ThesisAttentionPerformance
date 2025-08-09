#!/usr/bin/env python3
"""
Ablation Study Experiments Script
Run all required experiments based on reviewer feedback
"""

import subprocess
import time
import json
from datetime import datetime

# Base configuration matching your Kaggle setup
BASE_CONFIG = {
    'image_size': 224,
    'batch_size': 64,
    'lr': 0.005,
    'min_lr': '1e-6',
    'weight_decay': '5e-4',
    'optimizer': 'SGD',
    'lr_scheduler': 'ReduceLROnPlateau',
    'max_epoch': 90,
    'early_stopping_patience': 10,
    'device': 'cuda',
    'random_seed': 42,
    'wandb_key': 'cfa48af5b389548142fc1fcc1ab79cbcfe7fc07b',
    'dataset_path': '/kaggle/input/'
}

def run_experiment(config, description=""):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"{'='*60}")
    
    # Build command
    cmd = ['python', 'train.py']
    for key, value in config.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    # Print command for verification
    print("Command:", ' '.join(cmd))
    print()
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        duration = time.time() - start_time
        print(f"✅ COMPLETED in {duration/60:.1f} minutes: {description}")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ FAILED after {duration/60:.1f} minutes: {description}")
        print(f"Error: {e}")
        return False

def create_config(dataset, backbone, attention, fusion_type=None, 
                 special_flags=None, max_epoch=None, wandb_suffix=""):
    """Create experiment configuration"""
    config = BASE_CONFIG.copy()
    config.update({
        'dataset': dataset,
        'backbone': backbone,
        'attention': attention,
        'wandb_project': f'GLSBLOCK_ABLATION_{backbone}_{attention}_{dataset}',
        'wandb_run': f'ablation_{attention}_{backbone}_{dataset.lower()}_{wandb_suffix}'
    })
    
    if fusion_type:
        config['fusion_type'] = fusion_type
        config['wandb_project'] += f'_{fusion_type}'
        config['wandb_run'] += f'_{fusion_type}'
    
    if max_epoch:
        config['max_epoch'] = max_epoch
        
    if special_flags:
        config.update(special_flags)
        
    return config

def main():
    print("🎯 ABLATION STUDY EXPERIMENTS - REVIEWER REQUIREMENTS")
    print("=" * 60)
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'experiments': [],
        'summary': {'success': 0, 'failed': 0}
    }
    
    # Define experiments based on reviewer requirements
    experiments = [
        # 1. COMPUTATIONAL ANALYSIS (Quick - 1 epoch for measurement)
        {
            'config': create_config('HAM10000', 'VGG16', 'none', 
                                  special_flags={'enable_computational_analysis': True}, 
                                  max_epoch=1, wandb_suffix='comp_analysis'),
            'description': 'Computational Analysis - Baseline VGG16'
        },
        {
            'config': create_config('HAM10000', 'VGG16', 'CBAM', 
                                  special_flags={'enable_computational_analysis': True}, 
                                  max_epoch=1, wandb_suffix='comp_analysis'),
            'description': 'Computational Analysis - GLA-Block CBAM'
        },
        {
            'config': create_config('HAM10000', 'VGG16', 'mha_only', 
                                  special_flags={'enable_computational_analysis': True}, 
                                  max_epoch=1, wandb_suffix='comp_analysis'),
            'description': 'Computational Analysis - MHA Only'
        },
        {
            'config': create_config('HAM10000', 'VGG16', 'cbam_only', 
                                  special_flags={'enable_computational_analysis': True}, 
                                  max_epoch=1, wandb_suffix='comp_analysis'),
            'description': 'Computational Analysis - CBAM Only'
        },
        
        # 2. ABLATION STUDY - MHA Only (Full training)
        {
            'config': create_config('HAM10000', 'VGG16', 'mha_only', wandb_suffix='full'),
            'description': 'Ablation Study - MHA Only (VGG16, HAM10000)'
        },
        {
            'config': create_config('HAM10000', 'ResNet18', 'mha_only', wandb_suffix='full'),
            'description': 'Ablation Study - MHA Only (ResNet18, HAM10000)'
        },
        {
            'config': create_config('isic-2018-task-3', 'VGG16', 'mha_only', wandb_suffix='full'),
            'description': 'Ablation Study - MHA Only (VGG16, ISIC-2018)'
        },
        
        # 3. ABLATION STUDY - Lightweight Only  
        {
            'config': create_config('HAM10000', 'VGG16', 'cbam_only', wandb_suffix='full'),
            'description': 'Ablation Study - CBAM Only (VGG16, HAM10000)'
        },
        {
            'config': create_config('HAM10000', 'VGG16', 'bam_only', wandb_suffix='full'),
            'description': 'Ablation Study - BAM Only (VGG16, HAM10000)'
        },
        {
            'config': create_config('HAM10000', 'VGG16', 'scse_only', wandb_suffix='full'),
            'description': 'Ablation Study - scSE Only (VGG16, HAM10000)'
        },
        
        # 4. FUSION COMPARISON
        {
            'config': create_config('HAM10000', 'VGG16', 'CBAM', fusion_type='concat', wandb_suffix='full'),
            'description': 'Fusion Comparison - Concatenation (VGG16, HAM10000)'
        },
        {
            'config': create_config('HAM10000', 'ResNet18', 'CBAM', fusion_type='concat', wandb_suffix='full'),
            'description': 'Fusion Comparison - Concatenation (ResNet18, HAM10000)'
        },
        
        # 5. CROSS-DATASET EVALUATION (Reduced epochs)
        {
            'config': create_config('HAM10000', 'VGG16', 'CBAM', 
                                  special_flags={'cross_dataset_eval': True}, 
                                  max_epoch=30, wandb_suffix='cross_dataset'),
            'description': 'Cross-dataset Evaluation - GLA-Block CBAM'
        },
        
        # 6. KEY BASELINE COMPARISONS (if not already done)
        {
            'config': create_config('HAM10000', 'VGG16', 'none', wandb_suffix='baseline'),
            'description': 'Baseline - No Attention (VGG16, HAM10000)'
        },
        {
            'config': create_config('HAM10000', 'ResNet18', 'none', wandb_suffix='baseline'),
            'description': 'Baseline - No Attention (ResNet18, HAM10000)'
        },
    ]
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        print(f"\n📋 EXPERIMENT {i}/{len(experiments)}")
        success = run_experiment(exp['config'], exp['description'])
        
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
        if i < len(experiments):
            print("⏳ Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    # Save results summary
    results['end_time'] = datetime.now().isoformat()
    with open('ablation_experiments_log.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 ABLATION STUDY EXPERIMENTS COMPLETED")
    print("="*60)
    print(f"✅ Successful: {results['summary']['success']}")
    print(f"❌ Failed: {results['summary']['failed']}")
    print(f"📊 Total: {len(experiments)} experiments")
    print(f"📋 Log saved to: ablation_experiments_log.json")
    
    if results['summary']['failed'] > 0:
        print("\n⚠️  Some experiments failed. Check the log for details.")
        print("You may need to rerun failed experiments manually.")
    else:
        print("\n🎯 All experiments completed successfully!")
        print("Ready for paper resubmission with reviewer requirements addressed.")

if __name__ == "__main__":
    main()