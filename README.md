# Grokking: Extended Analysis

This branch contains extended analysis of the grokking phenomenon in neural networks, focusing on weight decay, activation norms, and sparsity patterns during training.

## Key Insights

### 1. Adam vs AdamW: Weight Decay is Essential

**Finding:** Weight decay on parameters is essential—not just for generalization, but even for memorization.

Training with vanilla Adam (no weight decay) fails to achieve either memorization or grokking. The model simply does not learn the task. This suggests that weight decay plays a regularizing role that is fundamental to the optimization landscape, not merely a generalization trick applied post-memorization.

**Implication:** Weight decay should be considered a core component of the training dynamics, not an optional regularizer.

---

### 2. Norm Decay During Grokking: Capacity-Dependent Patterns

**Finding:** The commonly observed pattern of activation norm decay during grokking is **not universal**—it depends on model capacity.

| Hidden Dim | Memorization Speed | Norm Behavior Post-Memorization |
|------------|-------------------|--------------------------------|
| 32         | Slower            | Median activation **increased** |
| 128        | Faster            | Median activation **decreased** |

For smaller models (32-dim hidden), the median activation value actually *increases* during the grokking phase after memorization. Larger models (128-dim) show the expected decrease. This suggests that the internal representations learned during grokking differ qualitatively based on available capacity.

**Implication:** Theories of grokking that rely on norm compression may not generalize across model scales.

---

### 3. Sparsity Analysis: Uniform Shrinkage, Not Increased Sparsity

**Finding:** Sparsity does **not** decrease during grokking. Instead, we observe uniform shrinkage in model weights.

Contrary to the hypothesis that grokking involves pruning away unnecessary connections (increasing sparsity), our analysis shows:
- Weight magnitudes shrink uniformly across the network
- The fraction of "near-zero" weights remains relatively stable
- Circuit size evolution shows compression without selective pruning

We provide visualizations of:
- Activation sparsity over training
- Weight sparsity evolution
- Circuit size dynamics

**Implication:** Grokking may be better characterized as "weight compression" rather than "sparse circuit discovery."

---

## Repository Structure

```
├── grokking.py           # Main training script
├── sparsity_analysis.py  # Sparsity measurement and visualization
├── sparsity_expt.py      # Sparsity experiment runner
├── checkpoints/          # Model checkpoints at various epochs
├── results/
│   ├── logs/             # Training logs for different configurations
│   ├── plots/            # Training curves and sweep comparisons
│   └── sparsity*/        # Sparsity analysis outputs by model size
└── analyze/              # Additional analysis scripts
```

## Usage

```bash
# Run base grokking experiment
python grokking.py

# Run sparsity analysis on checkpoints
python sparsity_analysis.py

# Run sparsity experiments across model sizes
python sparsity_expt.py
```

## Requirements

See `requirements.txt`. Core dependencies: PyTorch, matplotlib, numpy.
