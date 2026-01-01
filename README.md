# Ψ (Psi)

Experiments in neural network generalization and learning dynamics.

## Experiment 1: Grokking

**Replicating delayed generalization in modular arithmetic**

### Task

Train a simple MLP to learn modular addition: `(a + b) mod 97`

- **Input**: One-hot encoded pairs (a, b) where a, b ∈ {0, 1, ..., 96}
- **Output**: (a + b) mod 97
- **Architecture**: 3-layer MLP (194 → 128 → 128 → 97)
- **Training**: AdamW with weight decay = 1.0, 50% train/test split

### Results

![Grokking Results](results/plots/baseline.png)

The model exhibits classic **grokking** behavior:
- **Overfitting** (~epoch 400): Train accuracy reaches 99%, test accuracy near 0%
- **Grokking** (~epoch 2000): Test accuracy suddenly jumps from ~10% to 99%
- **Delay**: ~1600 epochs between memorization and generalization

### Analysis

1. **Delayed generalization**: The network first memorizes the training data, then discovers the underlying algorithm long after achieving perfect training accuracy.

2. **Weight decay is crucial**: Without strong L2 regularization (wd=1.0), grokking doesn't occur — the model stays stuck in the memorization regime.

3. **Weight norm dynamics**: The L2 norm peaks during memorization, then decreases as the model transitions to a more generalizable solution.

This confirms the findings from [Power et al. (2022)](https://arxiv.org/abs/2201.02177): neural networks can learn generalizable representations far beyond the point of overfitting.

## References

- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)

## Setup

```bash
git clone https://github.com/AgentTorch/psi.git
cd psi
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python grokking_simple_test.py
```
