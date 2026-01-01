# Grokking in Neural Networks

This repository studies the **grokking** phenomenon of delayed generalization in neural networks on small algorithmic datasets, following and extending the modular arithmetic experiments introduced by Power et al. (2022).

The project provides:
- a minimal, reproducible baseline experiment that exhibits grokking, and  
- a set of controlled sensitivity analyses probing how grokking depends on regularization, model capacity, task type, and training data size.

The goal is empirical characterization rather than proposing a definitive mechanistic explanation.

---

## Overview

**Task**
- Modular arithmetic (e.g., `(a + b) mod 97`)
- Full input space: all 9,409 ordered pairs
- One-hot encoded inputs

**Model**
- MLP with two hidden layers  
  `(194 → H → H → 97)`
- ReLU activations
- AdamW optimizer with explicit weight decay

**Key Phenomenon**
- Rapid memorization (perfect train accuracy, chance test accuracy)
- Prolonged plateau with no apparent progress
- Sudden transition to perfect generalization (grokking)

---

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Run Baseline Experiment
```bash
python grokking.py
```

Runs modular addition with default settings:
- `hidden_dim = 128`
- `weight_decay = 1.0`
- `train/test split = 50% / 50%`
- `epochs = 5000`

Outputs:
- `results/plots/baseline.png`
- `results/logs/baseline.json`

Expected behavior:
- Training accuracy reaches ~100% early
- Test accuracy remains near chance
- Grokking transition occurs after ~2000 epochs

---

## Sensitivity Analysis

### Run All Sweeps
```bash
python sweeps.py
```

Runs four controlled sweeps:

1. **Weight Decay**  
   `[0.0, 0.1, 0.5, 1.0, 2.0, 5.0]`

2. **Model Size (hidden units)**  
   `[32, 64, 128, 256, 512]`

3. **Operation Type**  
   `add`, `subtract`, `multiply` (mod 97)

4. **Training Data Fraction**  
   `[0.3, 0.5, 0.7, 0.9]`

Outputs:
- Per-sweep plots in `results/plots/`
- Aggregated metrics in `results/summary.json`

---

## Analysis Notes

A detailed discussion of the sensitivity analyses and interpretations of grokking dynamics—covering regularization strength, model size, task variation, and data coverage—is provided in [commentary.md](./commentary.md). This document serves as a companion analysis for readers interested in the underlying learning dynamics beyond code execution and reproduction.

---

## Key Observations

- **Regularization dependence**  
  Grokking is only observed above a threshold level of weight decay within the training horizon.

- **Model capacity**  
  Larger models memorize much faster, but grok on a similar absolute timescale, increasing the delay between memorization and generalization.

- **Task robustness**  
  Grokking occurs across modular addition, subtraction, and multiplication.

- **Data threshold**  
  Below a minimum training fraction, models memorize but fail to grok despite strong regularization.

---

## Project Structure

```
├── grokking.py        # Core experiment: data, model, training loop
├── sweeps.py          # Sensitivity analyses
├── results/
│   ├── logs/          # JSON logs from runs
│   ├── plots/         # Generated figures
│   └── summary.json   # Aggregated sweep results
└── README.md
```

---

## Relation to Prior Work

This repository reproduces the grokking phenomenon introduced in:

> Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V.  
> **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**  
> arXiv:2201.02177

The analyses here focus on characterizing when grokking occurs and how its timing depends on key experimental factors, rather than providing a definitive mechanistic explanation.

---

## Notes

- All experiments use a fixed random seed for controlled comparisons.
- Metrics are logged every 50 epochs.
- Grokking times are scoped to the training horizon and configurations used in this repository.