"""
Grokking Experiment - Barebones Implementation
Modular Arithmetic: (a + b) mod p
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# ============================================================================
# 1. DATA GENERATION
# ============================================================================

def generate_data(p=97, train_frac=0.5):
    """Generate modular addition dataset: (a + b) mod p"""
    inputs, labels = [], []
    
    for a in range(p):
        for b in range(p):
            # One-hot encode: [a_onehot, b_onehot]
            x = np.zeros(2 * p)
            x[a] = 1
            x[p + b] = 1
            inputs.append(x)
            labels.append((a + b) % p)
    
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
# 2. MODEL
# ============================================================================

class MLP(nn.Module):
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
# 3. TRAINING
# ============================================================================

def train(model, train_data, test_data, epochs=30000, lr=1e-3, wd=1.0):
    """Train and log metrics"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    train_x, train_y = train_data
    test_x, test_y = test_data
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    # Storage
    log = {'epoch': [], 'train_acc': [], 'test_acc': [], 
           'train_loss': [], 'test_loss': [], 'weight_norm': []}
    
    for epoch in tqdm(range(epochs)):
        # Train step
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        
        # Log every 100 epochs
        if epoch % 100 == 0:
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
# 4. VISUALIZATION
# ============================================================================

def plot_results(log, id_str='7000'):
    """Plot grokking phenomenon"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = log['epoch']
    
    # Accuracy
    axes[0, 0].plot(epochs, log['train_acc'], label='Train')
    axes[0, 0].plot(epochs, log['test_acc'], label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy (Grokking)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, log['train_loss'], label='Train')
    axes[0, 1].plot(epochs, log['test_loss'], label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Weight norm
    axes[1, 0].plot(epochs, log['weight_norm'], color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].set_title('Weight Norm')
    axes[1, 0].grid(alpha=0.3)
    
    # Test accuracy closeup
    axes[1, 1].plot(epochs, log['test_acc'], linewidth=2, color='orange')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Grokking Transition')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'grokking_results_{id_str}.png'.format(id_str), dpi=150)
    print('\nPlot saved: grokking_results.png')
    plt.show()

# ============================================================================
# 5. ANALYSIS
# ============================================================================

def analyze(log):
    """Print grokking statistics"""
    epochs = np.array(log['epoch'])
    test_acc = np.array(log['test_acc'])
    train_acc = np.array(log['train_acc'])
    
    print('\n' + '='*50)
    print('RESULTS')
    print('='*50)
    print(f'Final train acc: {train_acc[-1]:.4f}')
    print(f'Final test acc:  {test_acc[-1]:.4f}')
    
    # When did we overfit?
    overfit_idx = np.where(train_acc > 0.99)[0]
    if len(overfit_idx) > 0:
        overfit_epoch = epochs[overfit_idx[0]]
        print(f'\nOverfitting at epoch: {overfit_epoch}')
        print(f'Test acc at that point: {test_acc[overfit_idx[0]]:.4f}')
    
    # When did we grok?
    grok_idx = np.where(test_acc > 0.95)[0]
    if len(grok_idx) > 0:
        grok_epoch = epochs[grok_idx[0]]
        print(f'\nGrokking at epoch: {grok_epoch}')
        
        if len(overfit_idx) > 0:
            delay = grok_epoch - overfit_epoch
            print(f'Delay: {delay} epochs ({delay/overfit_epoch:.1f}x)')
    
    print('='*50 + '\n')

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Config
    p = 97
    hidden_dim = 128
    epochs = 9000 #30000
    
    print('='*50)
    print('GROKKING: Modular Arithmetic')
    print('='*50)
    print(f'Modulus: {p}')
    print(f'Hidden dim: {hidden_dim}')
    print(f'Epochs: {epochs:,}')
    print(f'Weight decay: 1.0 (crucial!)\n')
    
    # Generate data
    print('Generating data...')
    train_data, test_data = generate_data(p=p)
    print(f'Train: {len(train_data[0])} | Test: {len(test_data[0])}\n')
    
    # Create model
    model = MLP(2*p, hidden_dim, p).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}\n')
    
    # Train
    print('Training...')
    log = train(model, train_data, test_data, epochs=epochs)
    
    # Analyze
    analyze(log)
    
    # Plot
    plot_results(log, id_str=f'{epochs}')
    
    print('Done!')