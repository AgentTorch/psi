"""
grokking.py - Core functions for grokking experiments
Contains: data generation, model definition, training loop, analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_data(p=97, train_frac=0.5, operation='add', seed=42):
    """
    Generate modular arithmetic dataset.
    
    Args:
        p: modulus (prime number recommended)
        train_frac: fraction of data for training
        operation: 'add', 'subtract', or 'multiply'
        seed: random seed for reproducibility
    
    Returns:
        (train_x, train_y), (test_x, test_y)
    """
    np.random.seed(seed)
    inputs, labels = [], []
    
    for a in range(p):
        for b in range(p):
            if operation == 'add':
                result = (a + b) % p
            elif operation == 'subtract':
                result = (a - b) % p
            elif operation == 'multiply':
                result = (a * b) % p
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # One-hot encoding
            x = np.zeros(2 * p)
            x[a] = 1
            x[p + b] = 1
            inputs.append(x)
            labels.append(result)
    
    inputs = np.array(inputs)
    labels = np.array(labels)
    
    # Shuffle and split
    idx = np.random.permutation(len(inputs))
    split = int(len(inputs) * train_frac)
    
    train_x = torch.FloatTensor(inputs[idx[:split]])
    train_y = torch.LongTensor(labels[idx[:split]])
    test_x = torch.FloatTensor(inputs[idx[split:]])
    test_y = torch.LongTensor(labels[idx[split:]])
    
    return (train_x, train_y), (test_x, test_y)


# ============================================================================
# MODEL
# ============================================================================

class MLP(nn.Module):
    """Simple 2-layer MLP for grokking experiments"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# TRAINING
# ============================================================================

def train(model, train_data, test_data, epochs=5000, lr=1e-3, wd=1.0, 
          log_interval=50, device=device, verbose=True):
    """
    Train model and return metrics.
    
    Args:
        model: PyTorch model
        train_data: (train_x, train_y) tuple
        test_data: (test_x, test_y) tuple
        epochs: number of training epochs
        lr: learning rate
        wd: weight decay
        log_interval: how often to log metrics
        device: torch device
        verbose: whether to print progress
    
    Returns:
        Dictionary with training logs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    train_x, train_y = train_data
    test_x, test_y = test_data
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    log = {
        'epoch': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'weight_norm': []
    }
    
    for epoch in tqdm(range(epochs)):
        # Training step
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        
        # Logging
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Train metrics
                train_pred = out.argmax(dim=1)
                train_acc = (train_pred == train_y).float().mean().item()
                
                # Test metrics
                test_out = model(test_x)
                test_loss = criterion(test_out, test_y).item()
                test_pred = test_out.argmax(dim=1)
                test_acc = (test_pred == test_y).float().mean().item()
                
                # Weight norm
                w_norm = sum(p.norm().item()**2 for p in model.parameters())**0.5
                
                # Store
                log['epoch'].append(epoch)
                log['train_acc'].append(train_acc)
                log['test_acc'].append(test_acc)
                log['train_loss'].append(loss.item())
                log['test_loss'].append(test_loss)
                log['weight_norm'].append(w_norm)
    
    return log


# ============================================================================
# ANALYSIS
# ============================================================================

def find_grokking_epoch(log, threshold=0.95):
    """Find when test accuracy crosses threshold"""
    epochs = np.array(log['epoch'])
    test_acc = np.array(log['test_acc'])
    idx = np.where(test_acc >= threshold)[0]
    return int(epochs[idx[0]]) if len(idx) > 0 else None


def find_overfitting_epoch(log, threshold=0.99):
    """Find when train accuracy crosses threshold"""
    epochs = np.array(log['epoch'])
    train_acc = np.array(log['train_acc'])
    idx = np.where(train_acc >= threshold)[0]
    return int(epochs[idx[0]]) if len(idx) > 0 else None


def analyze_run(log):
    """Analyze a single run and return metrics"""
    grok_epoch = find_grokking_epoch(log)
    overfit_epoch = find_overfitting_epoch(log)
    
    analysis = {
        'final_train_acc': log['train_acc'][-1],
        'final_test_acc': log['test_acc'][-1],
        'grok_epoch': grok_epoch,
        'overfit_epoch': overfit_epoch,
        'delay': grok_epoch - overfit_epoch if (grok_epoch and overfit_epoch) else None
    }
    
    return analysis


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_single_run(log, save_path=None, title='Grokking Experiment'):
    """Plot results from a single run"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = log['epoch']
    
    # Find overfitting point
    overfit_epoch = find_overfitting_epoch(log)
    
    # Accuracy
    axes[0, 0].plot(epochs, log['train_acc'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, log['test_acc'], label='Test', linewidth=2)
    if overfit_epoch:
        axes[0, 0].axvline(x=overfit_epoch, color='red', linestyle='--', 
                          alpha=0.6, linewidth=1.5, label=f'Train→100% (epoch {overfit_epoch})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, log['train_loss'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, log['test_loss'], label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Weight norm
    axes[1, 0].plot(epochs, log['weight_norm'], color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].set_title('Weight Norm')
    axes[1, 0].grid(alpha=0.3)
    
    # Test accuracy closeup
    axes[1, 1].plot(epochs, log['test_acc'], linewidth=2.5, color='orange')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    if overfit_epoch:
        axes[1, 1].axvline(x=overfit_epoch, color='red', linestyle='--', 
                          alpha=0.6, linewidth=1.5, label=f'Train→100%')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Grokking Transition')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    return fig


def save_log(log, filepath):
    """Save log to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'Saved log: {filepath}')


def load_log(filepath):
    """Load log from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# MAIN (for standalone execution)
# ============================================================================

if __name__ == '__main__':
    print('='*60)
    print('GROKKING: Modular Arithmetic')
    print('='*60)
    
    # Config
    p = 97
    hidden_dim = 128
    epochs = 10000 #5000
    
    print(f'\nConfig:')
    print(f'  Modulus: {p}')
    print(f'  Hidden dim: {hidden_dim}')
    print(f'  Epochs: {epochs:,}')
    print(f'  Device: {device}\n')
    
    # Generate data
    print('Generating data...')
    train_data, test_data = generate_data(p=p)
    print(f'  Train: {len(train_data[0])} | Test: {len(test_data[0])}\n')
    
    # Create model
    model = MLP(2*p, hidden_dim, p).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {num_params:,} parameters\n')
    
    # Train
    print('Training...')
    log = train(model, train_data, test_data, epochs=epochs)
    
    # Analyze
    analysis = analyze_run(log)
    print('\n' + '='*60)
    print('RESULTS')
    print('='*60)
    print(f'Final train acc: {analysis["final_train_acc"]:.4f}')
    print(f'Final test acc:  {analysis["final_test_acc"]:.4f}')
    
    if analysis['overfit_epoch']:
        print(f'\nOverfitting at epoch: {analysis["overfit_epoch"]}')
        # Find test acc at overfitting point
        epochs = np.array(log['epoch'])
        test_acc = np.array(log['test_acc'])
        idx = np.where(epochs == analysis['overfit_epoch'])[0]
        if len(idx) > 0:
            print(f'Test acc at that point: {test_acc[idx[0]]:.4f}')
    
    if analysis['grok_epoch']:
        print(f'\nGrokking at epoch: {analysis["grok_epoch"]}')
    
    if analysis['delay']:
        print(f'Delay: {analysis["delay"]} epochs ({analysis["delay"]/analysis["overfit_epoch"]:.1f}x)')
    
    print('='*60)
    
    # Plot
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    plot_single_run(log, save_path='results/plots/baseline.png')
    save_log(log, 'results/logs/baseline.json')
    
    print('\nDone!')