import torch
import os
from grokking import generate_data, MLP, analyze_run
from sparsity_analysis import (
    train_with_sparsity_tracking,
    plot_sparsity_evolution,
    plot_weight_distribution_comparison,
    plot_activation_heatmap,
)
from tqdm import tqdm
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('='*70)
print('GROKKING WITH SPARSITY ANALYSIS')
print('='*70)

# Config
p = 97
hidden_dim = 128 #128
epochs = 5000
checkpoint_epochs = [0, 200, 400, 800, 1500, 2300, 3000, 5000]

print(f'\nConfig:')
print(f'  Modulus: {p}')
print(f'  Hidden dim: {hidden_dim}')
print(f'  Epochs: {epochs:,}')
print(f'  Checkpoints: {checkpoint_epochs}')
print(f'  Device: {device}\n')

# Generate data
print('Generating data...')
train_data, test_data = generate_data(p=p)
print(f'  Train: {len(train_data[0])} | Test: {len(test_data[0])}\n')

# Create model
model = MLP(2*p, hidden_dim, p).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'Model: {num_params:,} parameters\n')

# Train with sparsity tracking
print('Training with sparsity tracking...')
log, tracker = train_with_sparsity_tracking(
    model, train_data, test_data,
    checkpoint_epochs=checkpoint_epochs,
    epochs=epochs,
    lr=1e-3,
    wd=1.0,
    device=device
)

print('\n' + '='*70)
print('TRAINING COMPLETE - GENERATING VISUALIZATIONS')
print('='*70)

# Analysis
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

# Create output directory
os.makedirs(f'results/sparsity_{hidden_dim}', exist_ok=True)

# 1. Plot sparsity evolution
print('\n1. Plotting sparsity evolution...')
plot_sparsity_evolution(tracker, save_path='results/sparsity_{hidden_dim}/sparsity_evolution.png'.format(hidden_dim=hidden_dim))

# 2. Plot weight distribution comparison
print('2. Plotting weight distribution comparison...')
comparison_epochs = [0, 400, 2300, 5000]
plot_weight_distribution_comparison(
    tracker, 
    comparison_epochs, 
    save_path='results/sparsity_{hidden_dim}/weight_comparison.png'.format(hidden_dim=hidden_dim)
)

# 3. Plot activation heatmaps for key epochs
print('3. Plotting activation heatmaps...')
for epoch in [0, 400, 2300, 5000]:
    # Load checkpoint
    checkpoint_model = MLP(2*p, hidden_dim, p).to(device)
    tracker.load_checkpoint(checkpoint_model, epoch)
    
    plot_activation_heatmap(
        checkpoint_model,
        train_data[0],
        epoch,
        device=device,
        save_path=f'results/sparsity_{hidden_dim}/activations_epoch_{epoch}.png'.format(hidden_dim=hidden_dim)
    )

print('\n' + '='*70)
print('ANALYSIS COMPLETE!')
print('='*70)
print('\nResults saved in results/sparsity_{hidden_dim}/:'.format(hidden_dim=hidden_dim))
print('  - sparsity_evolution.png')
print('  - weight_comparison.png')
print('  - activations_epoch_*.png')
print('\nCheckpoints saved in checkpoints/')