"""
sparsity_analysis.py - Tools for analyzing network sparsity and feature evolution
Contains: activation hooks, weight analysis, checkpoint comparison, visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


# ============================================================================
# ACTIVATION HOOK SYSTEM
# ============================================================================

class ActivationCapture:
    """Captures activations from specific layers during forward pass"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names=None):
        """
        Register forward hooks to capture activations.
        
        Args:
            layer_names: List of layer names to hook. If None, hooks all layers.
        """
        # Clear previous hooks
        self.clear_hooks()
        
        if layer_names is None:
            # Hook all Linear layers
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    layer_names.append(name)
        
        for name in layer_names:
            module = dict(self.model.named_modules())[name]
            hook = module.register_forward_hook(self._create_hook(name))
            self.hooks.append(hook)
        
        return layer_names
    
    def _create_hook(self, name):
        """Create hook function for a specific layer"""
        def hook(module, input, output):
            # Apply activation function if ReLU follows
            self.activations[name] = output.detach()
        return hook
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def get_activations(self):
        """Return captured activations"""
        return self.activations


# ============================================================================
# SPARSITY METRICS
# ============================================================================

def compute_weight_sparsity(model, threshold=0.01):
    """
    Compute sparsity of model weights.
    
    Args:
        model: PyTorch model
        threshold: Values below this are considered "zero"
    
    Returns:
        Dictionary with per-layer and global sparsity metrics
    """
    sparsity_info = {
        'per_layer': {},
        'global': {}
    }
    
    total_params = 0
    sparse_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Per-layer stats
            layer_total = param.numel()
            layer_sparse = (param.abs() < threshold).sum().item()
            layer_sparsity = layer_sparse / layer_total
            
            sparsity_info['per_layer'][name] = {
                'sparsity': layer_sparsity,
                'total': layer_total,
                'sparse': layer_sparse,
                'active': layer_total - layer_sparse,
                'mean_abs': param.abs().mean().item(),
                'std': param.std().item(),
            }
            
            # Accumulate global
            total_params += layer_total
            sparse_params += layer_sparse
    
    # Global stats
    sparsity_info['global']['sparsity'] = sparse_params / total_params
    sparsity_info['global']['total'] = total_params
    sparsity_info['global']['sparse'] = sparse_params
    sparsity_info['global']['active'] = total_params - sparse_params
    
    return sparsity_info


def compute_activation_sparsity(activations, threshold=0.01):
    """
    Compute sparsity of activations.
    
    Args:
        activations: Dict of layer_name -> activation tensor
        threshold: Values below this are considered inactive
    
    Returns:
        Dictionary with per-layer activation sparsity
    """
    sparsity_info = {}
    
    for name, act in activations.items():
        # Apply ReLU if needed (some activations might be pre-ReLU)
        act_positive = F.relu(act)
        
        # Compute sparsity
        total = act_positive.numel()
        inactive = (act_positive < threshold).sum().item()
        sparsity = inactive / total
        
        # Effective neurons (averaged over batch)
        effective_per_sample = (act_positive >= threshold).float().sum(dim=1).mean().item()
        
        sparsity_info[name] = {
            'sparsity': sparsity,
            'effective_neurons': effective_per_sample,
            'mean_activation': act_positive.mean().item(),
            'std_activation': act_positive.std().item(),
            'max_activation': act_positive.max().item(),
        }
    
    return sparsity_info


def analyze_model_sparsity(model, data_x, device='cuda', 
                           weight_threshold=0.01, act_threshold=0.01):
    """
    Complete sparsity analysis of model.
    
    Args:
        model: PyTorch model
        data_x: Input data for activation analysis
        device: Device to run on
        weight_threshold: Threshold for weight sparsity
        act_threshold: Threshold for activation sparsity
    
    Returns:
        Dictionary with complete sparsity analysis
    """
    model.eval()
    data_x = data_x.to(device)
    
    # Capture activations
    capture = ActivationCapture(model)
    layer_names = capture.register_hooks()
    
    with torch.no_grad():
        _ = model(data_x)
    
    activations = capture.get_activations()
    capture.clear_hooks()
    
    # Compute sparsity
    weight_sparsity = compute_weight_sparsity(model, weight_threshold)
    activation_sparsity = compute_activation_sparsity(activations, act_threshold)
    
    return {
        'weight': weight_sparsity,
        'activation': activation_sparsity,
        'layer_names': layer_names,
    }


# ============================================================================
# CHECKPOINT COMPARISON
# ============================================================================

class SparsityTracker:
    """Track sparsity evolution during training"""
    
    def __init__(self, checkpoints_dir='checkpoints'):
        self.checkpoints_dir = checkpoints_dir
        self.history = []
        os.makedirs(checkpoints_dir, exist_ok=True)
    
    def save_checkpoint(self, model, epoch, data_x, device='cuda'):
        """
        Save model checkpoint and compute sparsity metrics.
        
        Args:
            model: PyTorch model
            epoch: Current epoch
            data_x: Input data for activation analysis
            device: Device
        """
        # Save model
        checkpoint_path = os.path.join(self.checkpoints_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        
        # Analyze sparsity
        analysis = analyze_model_sparsity(model, data_x, device)
        analysis['epoch'] = epoch
        analysis['checkpoint_path'] = checkpoint_path
        
        self.history.append(analysis)
        
        return analysis
    
    def load_checkpoint(self, model, epoch):
        """Load model from checkpoint"""
        checkpoint_path = os.path.join(self.checkpoints_dir, f'model_epoch_{epoch}.pt')
        model.load_state_dict(torch.load(checkpoint_path))
        return model
    
    def get_history(self):
        """Get complete history of sparsity metrics"""
        return self.history
    
    def compare_epochs(self, epochs):
        """
        Compare sparsity across multiple epochs.
        
        Args:
            epochs: List of epoch numbers to compare
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'epochs': epochs,
            'weight_sparsity': [],
            'activation_sparsity': [],
            'effective_neurons': defaultdict(list),
        }
        
        for epoch in epochs:
            # Find corresponding analysis
            analysis = next((h for h in self.history if h['epoch'] == epoch), None)
            if analysis is None:
                continue
            
            # Extract metrics
            comparison['weight_sparsity'].append(
                analysis['weight']['global']['sparsity']
            )
            
            # Average activation sparsity across layers
            avg_act_sparsity = np.mean([
                info['sparsity'] 
                for info in analysis['activation'].values()
            ])
            comparison['activation_sparsity'].append(avg_act_sparsity)
            
            # Effective neurons per layer
            for layer, info in analysis['activation'].items():
                comparison['effective_neurons'][layer].append(
                    info['effective_neurons']
                )
        
        return comparison


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sparsity_evolution(tracker, save_path=None):
    """
    Plot how sparsity evolves over training.
    
    Args:
        tracker: SparsityTracker object with history
        save_path: Path to save figure
    """
    history = tracker.get_history()
    if not history:
        print("No history to plot")
        return
    
    epochs = [h['epoch'] for h in history]
    
    # Extract metrics
    weight_sparsity = [h['weight']['global']['sparsity'] for h in history]
    
    # Average activation sparsity
    activation_sparsity = []
    for h in history:
        avg_sparsity = np.mean([
            info['sparsity'] 
            for info in h['activation'].values()
        ])
        activation_sparsity.append(avg_sparsity)
    
    # Effective neurons per layer
    layer_names = history[0]['layer_names']
    effective_neurons = {name: [] for name in layer_names}
    for h in history:
        for name in layer_names:
            if name in h['activation']:
                effective_neurons[name].append(
                    h['activation'][name]['effective_neurons']
                )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight sparsity over time
    axes[0, 0].plot(epochs, weight_sparsity, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Weight Sparsity', fontsize=12)
    axes[0, 0].set_title('Weight Sparsity Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # 2. Activation sparsity over time
    axes[0, 1].plot(epochs, activation_sparsity, 'o-', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Activation Sparsity', fontsize=12)
    axes[0, 1].set_title('Activation Sparsity Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Both sparsities together
    axes[1, 0].plot(epochs, weight_sparsity, 'o-', label='Weight Sparsity', linewidth=2, markersize=6)
    axes[1, 0].plot(epochs, activation_sparsity, 's-', label='Activation Sparsity', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Sparsity', fontsize=12)
    axes[1, 0].set_title('Sparsity Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 4. Effective neurons per layer
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    for i, (name, neurons) in enumerate(effective_neurons.items()):
        layer_label = name.split('.')[-1] if '.' in name else name
        axes[1, 1].plot(epochs, neurons, 'o-', label=layer_label, 
                       linewidth=2, markersize=6, color=colors[i])
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Effective Neurons (avg)', fontsize=12)
    axes[1, 1].set_title('Circuit Size Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    return fig


def plot_weight_distribution_comparison(tracker, epochs_to_compare, save_path=None):
    """
    Compare weight distributions across different training stages.
    
    Args:
        tracker: SparsityTracker with history
        epochs_to_compare: List of epoch numbers to compare
        save_path: Path to save figure
    """
    history = tracker.get_history()
    
    n_epochs = len(epochs_to_compare)
    fig, axes = plt.subplots(2, n_epochs, figsize=(5*n_epochs, 8))
    if n_epochs == 1:
        axes = axes.reshape(-1, 1)
    
    for col, epoch in enumerate(epochs_to_compare):
        # Find checkpoint
        analysis = next((h for h in history if h['epoch'] == epoch), None)
        if analysis is None:
            continue
        
        # Collect all weights
        all_weights = []
        layer_weights = {}
        
        for layer_name, layer_info in analysis['weight']['per_layer'].items():
            # Load checkpoint to get actual weights
            model_class = tracker.history[0].get('model_class')  # Need to store this
            # For now, just use the sparsity info
            
            # Extract layer number for plotting
            if 'net.0' in layer_name:
                layer_label = 'Layer 1'
            elif 'net.2' in layer_name:
                layer_label = 'Layer 2'
            elif 'net.4' in layer_name:
                layer_label = 'Layer 3'
            else:
                layer_label = layer_name
            
            layer_weights[layer_label] = layer_info
        
        # Plot 1: Sparsity by layer (bar chart)
        ax1 = axes[0, col]
        layers = list(layer_weights.keys())
        sparsities = [layer_weights[l]['sparsity'] for l in layers]
        
        ax1.bar(layers, sparsities, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Sparsity', fontsize=11)
        ax1.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Mean absolute weight per layer
        ax2 = axes[1, col]
        mean_weights = [layer_weights[l]['mean_abs'] for l in layers]
        
        ax2.bar(layers, mean_weights, color='coral', alpha=0.7)
        ax2.set_ylabel('Mean |Weight|', fontsize=11)
        ax2.set_xlabel('Layer', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Weight Statistics Across Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    return fig


def plot_activation_heatmap(model, data_x, epoch, device='cuda', save_path=None):
    """
    Create heatmap of activations for sample inputs.
    
    Args:
        model: PyTorch model
        data_x: Input data (use subset for visualization)
        epoch: Current epoch (for title)
        device: Device
        save_path: Path to save figure
    """
    model.eval()
    data_x = data_x.to(device)
    
    # Limit to first 50 samples for visualization
    data_x = data_x[:50]
    
    # Capture activations
    capture = ActivationCapture(model)
    layer_names = capture.register_hooks()
    
    with torch.no_grad():
        _ = model(data_x)
    
    activations = capture.get_activations()
    capture.clear_hooks()
    
    # Create heatmaps
    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 6))
    if n_layers == 1:
        axes = [axes]
    
    for idx, name in enumerate(layer_names):
        act = F.relu(activations[name]).cpu().numpy()
        
        # Plot heatmap
        im = axes[idx].imshow(act.T, aspect='auto', cmap='viridis', 
                             interpolation='nearest')
        axes[idx].set_xlabel('Sample', fontsize=11)
        axes[idx].set_ylabel('Neuron', fontsize=11)
        
        layer_label = f"Layer {idx+1}"
        axes[idx].set_title(f'{layer_label}\n(Epoch {epoch})', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=axes[idx], label='Activation')
    
    plt.suptitle('Activation Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    return fig


# ============================================================================
# INTEGRATED TRAINING WITH SPARSITY TRACKING
# ============================================================================

def train_with_sparsity_tracking(model, train_data, test_data, 
                                 checkpoint_epochs=[0, 400, 1000, 2300, 5000],
                                 epochs=5000, lr=1e-3, wd=1.0,
                                 device='cuda'):
    """
    Train model while tracking sparsity at specific checkpoints.
    
    Args:
        model: PyTorch model
        train_data: (train_x, train_y) tuple
        test_data: (test_x, test_y) tuple
        checkpoint_epochs: List of epochs to save checkpoints
        epochs: Total training epochs
        lr: Learning rate
        wd: Weight decay
        device: Device
    
    Returns:
        Tuple of (training log, SparsityTracker)
    """
    from grokking import train  # Import from main module
    from tqdm import tqdm
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    train_x, train_y = train_data
    test_x, test_y = test_data
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    tracker = SparsityTracker()
    
    log = {
        'epoch': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'weight_norm': [],
    }
    
    log_interval = 50
    
    for epoch in tqdm(range(epochs + 1)):
        # Save checkpoint if needed
        if epoch in checkpoint_epochs:
            print(f"\nSaving checkpoint at epoch {epoch}...")
            tracker.save_checkpoint(model, epoch, train_x, device)
        
        if epoch == epochs:
            break
        
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
                train_pred = out.argmax(dim=1)
                train_acc = (train_pred == train_y).float().mean().item()
                
                test_out = model(test_x)
                test_loss = criterion(test_out, test_y).item()
                test_pred = test_out.argmax(dim=1)
                test_acc = (test_pred == test_y).float().mean().item()
                
                w_norm = sum(p.norm().item()**2 for p in model.parameters())**0.5
                
                log['epoch'].append(epoch)
                log['train_acc'].append(train_acc)
                log['test_acc'].append(test_acc)
                log['train_loss'].append(loss.item())
                log['test_loss'].append(test_loss)
                log['weight_norm'].append(w_norm)
    
    return log, tracker


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("Sparsity Analysis Module")
    print("Import this module and use the functions in your training script")
    print("\nExample:")
    print("  from sparsity_analysis import train_with_sparsity_tracking, plot_sparsity_evolution")
    print("  log, tracker = train_with_sparsity_tracking(model, train_data, test_data)")
    print("  plot_sparsity_evolution(tracker, 'sparsity_evolution.png')")