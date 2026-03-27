# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

<div align="center">

English | [한국어](./docs/README.ko.md)

</div> 

---

## Overview

This repository contains a **from-scratch implementation** of **Mamba**, a state-of-the-art architecture that replaces Transformer's quadratic attention mechanism with linear-complexity Selective State Space Models (SSMs). Mamba achieves comparable or better performance than Transformers while being significantly more efficient for long sequences.

## Mamba

| Feature | Transformer | Mamba |
|---------|-------------|-------|
| **Complexity** | O(N²) per layer | O(N) per layer |
| **Long Sequences** | Memory bottleneck | Efficient scaling |
| **Inference Speed** | Slow for autoregressive | Fast constant-time steps |
| **Architecture** | Attention + FFN (2 sub-blocks) | Unified single block |

## Project Structure

```text
mamba/
├── ssm_basic.py          # Basic State Space Model (SSM)
├── selective_ssm.py      # Selective SSM (core innovation)
├── mamba_block.py        # Complete Mamba block & model
├── test_mamba.py         # Comprehensive tests & visualization
├── example_usage.py      # Practical examples
└── requirements.txt      # Dependencies
```

## Core Concepts

#### 1. **State Space Models (SSM)**

SSMs model sequences using continuous-time state equations:

```
h'(t) = A·h(t) + B·x(t)    (state transition)
y(t) = C·h(t)               (output)
```

Discretized for digital computation:
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

**Key Parameters**:
- **A**: State transition matrix (how state evolves)
- **B**: Input matrix (how input affects state)
- **C**: Output matrix (how state produces output)
- **Δ (delta)**: Time scale parameter

#### 2. **Selective SSM** (Mamba's Core Innovation)

Unlike traditional SSMs with fixed parameters, **Selective SSM makes B, C, and Δ input-dependent**:

```python
B_t = Linear_B(x_t)      # Input-dependent
C_t = Linear_C(x_t)      # Input-dependent
Δ_t = Softplus(Linear_Δ(x_t))  # Input-dependent time scale
```

**"Selective"**
- Large Δ → Fast changes (attend to current input)
- Small Δ → Slow changes (remember past information)
- Dynamic B, C → Choose what to remember/forget

This achieves **Attention-like context awareness** with **O(N) complexity**!

#### 3. **Mamba Block** (Replaces Transformer Block)

```
┌─────────────────────────────────────────────────────────┐
│ TRANSFORMER BLOCK                                       │
├─────────────────────────────────────────────────────────┤
│ LayerNorm → Multi-Head Attention (O(N²)) → Add         │
│ LayerNorm → FFN (Linear→Act→Linear) → Add              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MAMBA BLOCK (Unified, O(N))                             │
├─────────────────────────────────────────────────────────┤
│ LayerNorm →                                             │
│   ├─ Input Projection (expand)    [FFN-like]           │
│   ├─ 1D Convolution (local info)  [NEW]                │
│   ├─ Selective SSM                [Attention替代]       │
│   ├─ Gating Mechanism              [FFN-like]           │
│   └─ Output Projection (reduce)   [FFN-like]           │
│ → Add (residual)                                        │
└─────────────────────────────────────────────────────────┘
```

**Key Differences**:
- **Attention → Selective SSM**: O(N²) → O(N)
- **Separate Attention + FFN → Unified Block**: More efficient
- **Added 1D Convolution**: Captures local patterns

## Limitations and Trade-offs

Key limitations identified in the Mamba paper and subsequent research:

### 1. Content-Based Reasoning Constraints
Due to SSM's sequential nature, direct comparisons between arbitrary positions are challenging. While Attention computes similarity between all token pairs directly, Mamba only accesses information through compressed hidden states.

### 2. In-Context Learning Performance
Performance degradation compared to Transformers in few-shot learning and prompt-based tasks. This is particularly noticeable in tasks requiring reference to in-context examples.

### 3. Discrete Copying Tasks
Struggles with tasks requiring exact information copying or citation from long sequences. Attention's direct token reference mechanism is better suited for such operations.

### 4. Training Dynamics
- Slower convergence than Transformers on small datasets
- Requires careful initialization strategies and learning rate scheduling
- May need longer warmup phases

### 5. Interpretability
Unlike Attention weights, SSM's hidden states are less intuitive to interpret. Tools for analyzing the model's decision-making process are limited.

### Hybrid Approaches
To address these limitations, hybrid architectures combining Mamba and Attention have been proposed:
- **Jamba** (AI21 Labs): Interleaves Mamba and Attention layers
- **Zamba** (Zyphra): Hybrid design balancing efficiency and performance

## Quick Start

### Installation

```bash
cd /home/jeonghs/workspace/mamba
pip install -r requirements.txt
```

### Run Tests

```bash
# Test individual components
python ssm_basic.py
python selective_ssm.py
python mamba_block.py

# Comprehensive tests with visualization
python test_mamba.py

# Practical examples
python example_usage.py
```

## Performance & Results

The code automatically uses GPU if available (falls back to CPU otherwise).

### 1. Model Performance (tested on GPU environment)

- **Model Size**: 4M parameters (6 layers, 256 dimensions)
- **Inference Speed**: ~3,300 tokens/sec on GPU
- **Memory Efficiency**: 8x faster than Transformer at 4096 sequence length

### 2. Complexity Comparison (d_model=512)

| Seq Length | Transformer Ops | Mamba Ops | Speedup |
|------------|-----------------|-----------|---------|
| 128 | 8.4M | 33.6M | 0.25x |
| 512 | 134.2M | 134.2M | 1.0x |
| 1024 | 536.9M | 268.4M | **2.0x** |
| 2048 | 2.1B | 536.9M | **4.0x** |
| 4096 | 8.6B | 1.1B | **8.0x** |

*Mamba becomes increasingly efficient as sequence length grows!*

### 3. Selective Behavior Visualization

Running `test_mamba.py` generates a visualization showing how Mamba selectively adapts to important vs. unimportant information:

<div align="center">
<img src="https://github.com/user-attachments/assets/42b15f59-6150-4bd5-b0f1-5682c7416ce2" width="512" height="768"></img><br/>
</div> 

The plot demonstrates:

1. **Top**: Input signal with an important region (timesteps 40-60).
2. **Middle**: Delta values increase in the important region (faster adaptation).
3. **Bottom**: Output reflects the selective processing.

## Usage Examples

### Language Modeling

```python
from mamba_block import MambaModel

model = MambaModel(
    d_model=256,
    n_layers=6,
    vocab_size=50000,
    d_state=16
)

# Generate text
tokens = torch.randint(0, 50000, (1, 10))  # prompt
logits = model(tokens)
next_token = logits[:, -1, :].argmax(dim=-1)
```

### Sequence Classification

```python
backbone = MambaModel(d_model=128, n_layers=4, vocab_size=None)
classifier = nn.Linear(128, num_classes)

features = backbone(x)  # (batch, seq_len, d_model)
pooled = features.mean(dim=1)  # global average pooling
logits = classifier(pooled)
```

## Code Highlights

All code includes **detailed comments comparing with Transformer**:

```python
# ============================================================
# 4. Selective SSM - Core Operation
# [Replaces Transformer's Multi-Head Self-Attention]
# Attention: Q@K^T@V with all token interactions (O(N²))
# Selective SSM: Input-dependent state space (O(N))
# ============================================================
y = self.ssm(x_conv)
```

## References

- **Mamba Paper**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **S4 (Structured State Spaces)**: [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- **Official Implementation**: [state-spaces/mamba](https://github.com/state-spaces/mamba)
