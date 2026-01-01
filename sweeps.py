"""
sweeps.py - Sensitivity analysis experiments
Runs multiple experiments varying hyperparameters and saves organized results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

# Import core functions from grokking.py
from grokking import (
    generate_data,
    MLP,
    train,
    find_grokking_epoch,
    find_overfitting_epoch,
    analyze_run,
    save_log,
    device
)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# SWEEP FUNCTIONS
# ============================================================================

def sweep_weight_decay(p=97, hidden_dim=128, epochs=5000, save_dir='results'):
    """Sweep over weight decay values"""
    print('\n' + '='*70)
    print('SWEEP 1: Weight Decay Sensitivity')
    print('='*70)
    
    wd_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    for wd in tqdm(wd_values, desc='Weight Decay'):
        # Run experiment
        train_data, test_data = generate_data(p=p)
        model = MLP(2*p, hidden_dim, p).to(device)
        log = train(model, train_data, test_data, epochs=epochs, wd=wd, verbose=False)
        
        # Analyze
        analysis = analyze_run(log)
        
        # Save log
        log_path = f'{save_dir}/logs/wd_{wd:.1f}.json'
        save_log(log, log_path)
        
        results.append({
            'weight_decay': wd,
            'log': log,
            **analysis
        })
    
    return results


def sweep_model_size(p=97, wd=1.0, epochs=5000, save_dir='results'):
    """Sweep over model sizes"""
    print('\n' + '='*70)
    print('SWEEP 2: Model Size Sensitivity')
    print('='*70)
    
    hidden_dims = [32, 64, 128, 256, 512]
    results = []
    
    for hidden_dim in tqdm(hidden_dims, desc='Model Size'):
        # Run experiment
        train_data, test_data = generate_data(p=p)
        model = MLP(2*p, hidden_dim, p).to(device)
        log = train(model, train_data, test_data, epochs=epochs, wd=wd, verbose=False)
        
        # Analyze
        analysis = analyze_run(log)
        num_params = sum(p.numel() for p in model.parameters())
        
        # Save log
        log_path = f'{save_dir}/logs/hidden_{hidden_dim}.json'
        save_log(log, log_path)
        
        results.append({
            'hidden_dim': hidden_dim,
            'num_params': num_params,
            'log': log,
            **analysis
        })
    
    return results


def sweep_operations(p=97, hidden_dim=128, wd=1.0, epochs=5000, save_dir='results'):
    """Sweep over different arithmetic operations"""
    print('\n' + '='*70)
    print('SWEEP 3: Operation Type')
    print('='*70)
    
    operations = ['add', 'subtract', 'multiply']
    results = []
    
    for op in tqdm(operations, desc='Operations'):
        # Run experiment
        train_data, test_data = generate_data(p=p, operation=op)
        model = MLP(2*p, hidden_dim, p).to(device)
        log = train(model, train_data, test_data, epochs=epochs, wd=wd, verbose=False)
        
        # Analyze
        analysis = analyze_run(log)
        
        # Save log
        log_path = f'{save_dir}/logs/op_{op}.json'
        save_log(log, log_path)
        
        results.append({
            'operation': op,
            'log': log,
            **analysis
        })
    
    return results


def sweep_data_fraction(p=97, hidden_dim=128, wd=1.0, epochs=5000, save_dir='results'):
    """Sweep over training data fraction"""
    print('\n' + '='*70)
    print('SWEEP 4: Training Data Fraction')
    print('='*70)
    
    train_fracs = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    for frac in tqdm(train_fracs, desc='Data Fraction'):
        # Run experiment
        train_data, test_data = generate_data(p=p, train_frac=frac)
        model = MLP(2*p, hidden_dim, p).to(device)
        log = train(model, train_data, test_data, epochs=epochs, wd=wd, verbose=False)
        
        # Analyze
        analysis = analyze_run(log)
        
        # Save log
        log_path = f'{save_dir}/logs/frac_{frac:.1f}.json'
        save_log(log, log_path)
        
        results.append({
            'train_fraction': frac,
            'train_size': len(train_data[0]),
            'log': log,
            **analysis
        })
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sweep_comparison(results, param_name, param_key, save_path):
    """Create comprehensive comparison plot for a sweep"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Test accuracy curves
    ax = axes[0, 0]
    for r in results:
        label = f"{param_name}={r[param_key]}"
        ax.plot(r['log']['epoch'], r['log']['test_acc'], label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title(f'Test Accuracy vs {param_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Train accuracy curves
    ax = axes[0, 1]
    for r in results:
        label = f"{param_name}={r[param_key]}"
        ax.plot(r['log']['epoch'], r['log']['train_acc'], label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Train Accuracy', fontsize=11)
    ax.set_title(f'Train Accuracy vs {param_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Grokking time vs parameter
    ax = axes[1, 0]
    param_values = [r[param_key] for r in results]
    grok_times = [r['grok_epoch'] if r['grok_epoch'] is not None else np.nan for r in results]
    ax.plot(param_values, grok_times, 'o-', markersize=10, linewidth=2, color='darkblue')
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Grokking Epoch', fontsize=11)
    ax.set_title(f'Grokking Time vs {param_name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 4: Final test accuracy vs parameter
    ax = axes[1, 1]
    final_accs = [r['final_test_acc'] for r in results]
    ax.plot(param_values, final_accs, 's-', markersize=10, linewidth=2, color='green')
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Final Test Accuracy', fontsize=11)
    ax.set_title(f'Final Test Acc vs {param_name}', fontsize=12, fontweight='bold')
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def plot_operations_comparison(results, save_path):
    """Custom plot for operations sweep (categorical)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test accuracy curves
    for r in results:
        axes[0].plot(r['log']['epoch'], r['log']['test_acc'], 
                     label=r['operation'].capitalize(), linewidth=2.5, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Test Accuracy', fontsize=11)
    axes[0].set_title('Test Accuracy by Operation', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Grokking time bar chart
    ops = [r['operation'].capitalize() for r in results]
    groks = [r['grok_epoch'] if r['grok_epoch'] else 0 for r in results]
    axes[1].bar(ops, groks, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    axes[1].set_xlabel('Operation', fontsize=11)
    axes[1].set_ylabel('Grokking Epoch', fontsize=11)
    axes[1].set_title('Grokking Time by Operation', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary_table(all_results):
    """Print summary table of all sweeps"""
    print('\n' + '='*80)
    print('SUMMARY TABLE')
    print('='*80)
    
    for sweep_name, results in all_results.items():
        print(f'\n{sweep_name}:')
        print('-' * 80)
        
        if 'weight_decay' in results[0]:
            print(f"{'WD':<10} {'Grok Epoch':<15} {'Overfit Epoch':<15} {'Delay':<10} {'Final Test Acc':<15}")
            print('-' * 80)
            for r in results:
                grok = r['grok_epoch'] if r['grok_epoch'] else 'N/A'
                overfit = r['overfit_epoch'] if r['overfit_epoch'] else 'N/A'
                delay = r['delay'] if r['delay'] else 'N/A'
                print(f"{r['weight_decay']:<10.1f} {str(grok):<15} {str(overfit):<15} {str(delay):<10} {r['final_test_acc']:<15.4f}")
        
        elif 'hidden_dim' in results[0]:
            print(f"{'Hidden':<10} {'Params':<15} {'Grok Epoch':<15} {'Delay':<10} {'Final Test Acc':<15}")
            print('-' * 80)
            for r in results:
                grok = r['grok_epoch'] if r['grok_epoch'] else 'N/A'
                delay = r['delay'] if r['delay'] else 'N/A'
                print(f"{r['hidden_dim']:<10} {r['num_params']:<15,} {str(grok):<15} {str(delay):<10} {r['final_test_acc']:<15.4f}")
        
        elif 'operation' in results[0]:
            print(f"{'Operation':<15} {'Grok Epoch':<15} {'Delay':<10} {'Final Test Acc':<15}")
            print('-' * 80)
            for r in results:
                grok = r['grok_epoch'] if r['grok_epoch'] else 'N/A'
                delay = r['delay'] if r['delay'] else 'N/A'
                print(f"{r['operation']:<15} {str(grok):<15} {str(delay):<10} {r['final_test_acc']:<15.4f}")
        
        elif 'train_fraction' in results[0]:
            print(f"{'Frac':<10} {'Train Size':<15} {'Grok Epoch':<15} {'Delay':<10} {'Final Test Acc':<15}")
            print('-' * 80)
            for r in results:
                grok = r['grok_epoch'] if r['grok_epoch'] else 'N/A'
                delay = r['delay'] if r['delay'] else 'N/A'
                print(f"{r['train_fraction']:<10.1f} {r['train_size']:<15} {str(grok):<15} {str(delay):<10} {r['final_test_acc']:<15.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('='*80)
    print('GROKKING SENSITIVITY ANALYSIS')
    print('='*80)
    print(f'Device: {device}')
    
    # Setup
    save_dir = 'results'
    os.makedirs(f'{save_dir}/logs', exist_ok=True)
    os.makedirs(f'{save_dir}/plots', exist_ok=True)
    
    p = 97
    epochs = 5000
    
    print(f'Configuration: p={p}, epochs={epochs:,}')
    print(f'Running 4 comprehensive sweeps...\n')
    
    # Run all sweeps
    all_results = {}
    
    # Sweep 1: Weight Decay (CRITICAL)
    wd_results = sweep_weight_decay(p=p, hidden_dim=128, epochs=epochs, save_dir=save_dir)
    all_results['Weight Decay'] = wd_results
    plot_sweep_comparison(wd_results, 'Weight Decay', 'weight_decay', 
                          f'{save_dir}/plots/sweep_weight_decay.png')
    
    # Sweep 2: Model Size
    size_results = sweep_model_size(p=p, wd=1.0, epochs=epochs, save_dir=save_dir)
    all_results['Model Size'] = size_results
    plot_sweep_comparison(size_results, 'Hidden Dim', 'hidden_dim', 
                          f'{save_dir}/plots/sweep_model_size.png')
    
    # Sweep 3: Operations
    op_results = sweep_operations(p=p, hidden_dim=128, wd=1.0, epochs=epochs, save_dir=save_dir)
    all_results['Operations'] = op_results
    plot_operations_comparison(op_results, f'{save_dir}/plots/sweep_operations.png')
    
    # Sweep 4: Data Fraction
    frac_results = sweep_data_fraction(p=p, hidden_dim=128, wd=1.0, epochs=epochs, save_dir=save_dir)
    all_results['Data Fraction'] = frac_results
    plot_sweep_comparison(frac_results, 'Train Fraction', 'train_fraction', 
                          f'{save_dir}/plots/sweep_data_fraction.png')
    
    # Print summary
    print_summary_table(all_results)
    
    # Save summary
    summary = {}
    for name, results in all_results.items():
        summary[name] = [{k: v for k, v in r.items() if k != 'log'} for r in results]
    
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved summary: {save_dir}/summary.json')
    
    print('\n' + '='*80)
    print('ALL SWEEPS COMPLETE!')
    print('='*80)
    print(f'\nResults saved in {save_dir}/:')
    print('  logs/     - JSON logs for each experiment')
    print('  plots/    - Comparison plots')
    print('  summary.json - Summary of all results')