# From Loops to Tensors: A Muscle Memory Training Guide

> **Goal:** Replace your default "for-loop" mental model with a "shape-first, axis-aware" mental model for NumPy and PyTorch tensor operations.
>
> **How to use this document:** Work through each level sequentially. For every exercise, FIRST predict the output shape and values on paper, THEN verify in code. This predict-then-verify cycle is what builds the mental model.

Throughout this guide, `import numpy as np` and `import torch` are assumed.

---

## Level 0: Shape Literacy

**The single most important habit:** Before any operation, know the shape of every tensor involved and predict the output shape.

### 0.1 Reading Shapes

```python
a = np.array([1, 2, 3])                    # shape: (3,)
b = np.array([[1, 2], [3, 4], [5, 6]])     # shape: (3, 2)
c = np.zeros((2, 3, 4))                     # shape: (2, 3, 4)
d = np.ones((5, 1, 3, 8))                   # shape: (5, 1, 3, 8)
```

**Mental model for shape tuples — read them as labeled dimensions:**

```
(3,)           →  3 elements                  [vector]
(3, 2)         →  3 rows, 2 cols              [matrix]
(2, 3, 4)      →  2 matrices of 3×4           [batch of matrices]
(5, 1, 3, 8)   →  5 × 1 × 3 × 8             [4D tensor]
```

**The rule that unlocks everything:**
In ML, the convention is almost always `(batch, ..., features)`. The leftmost dims are "outer" (batch, heads, etc.), the rightmost dim is typically the feature/embedding dimension.

### 0.2 Exercises — Shape Reading

Predict the shape, then verify:

```python
# E0.1
x = np.arange(24).reshape(2, 3, 4)
# What is x.shape?
# What is x[0].shape?
# What is x[0, 1].shape?
# What is x[0, 1, 2].shape?  (scalar or array?)

# E0.2
x = torch.randn(8, 10, 512)
# What is x.shape?
# What is x[0].shape?
# What is x[:, 0, :].shape?
# What is x[:, :, 0].shape?

# E0.3
x = np.zeros((4, 1, 3))
# What is x.shape?
# What is x[0].shape?
# What is x[0, 0].shape?

# E0.4 — Real ML shapes. What does each dim likely represent?
embeddings = torch.randn(32, 128, 768)    # (?, ?, ?)
attention   = torch.randn(32, 8, 128, 128) # (?, ?, ?, ?)
conv_feat   = torch.randn(16, 64, 32, 32)  # (?, ?, ?, ?)
```

---

## Level 1: Axis / Dim Reasoning

**Key insight:** Every dimension in a tensor has a meaning. Operations along an axis collapse, transform, or iterate over that dimension.

### 1.1 What "axis" Means

Think of `axis=k` as: "move along dimension k, doing the operation across all entries in that dimension."

```
x.shape = (3, 4)

x = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10,11,12]]

x.sum(axis=0)  →  sum down rows     →  [15, 18, 21, 24]   shape: (4,)
x.sum(axis=1)  →  sum across cols   →  [10, 26, 42]        shape: (3,)
```

**Visual mnemonic — "axis=k collapses dimension k":**

```
shape (3, 4)
         ↑  ↑
      ax=0  ax=1

sum(axis=0): collapse dim 0 (the 3) → result shape (4,)
sum(axis=1): collapse dim 1 (the 4) → result shape (3,)
```

### 1.2 The `keepdim` Parameter

`keepdim=True` keeps the collapsed dimension as size 1 instead of removing it. This is crucial for broadcasting (Level 4).

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)

x.sum(axis=1)                    # shape (2,)    — dim removed
x.sum(axis=1, keepdims=True)     # shape (2, 1)  — dim kept as 1
```

**Why this matters:** When you want to normalize rows:
```python
# Subtract row means
row_means = x.mean(axis=1, keepdims=True)   # (2, 1)
x_centered = x - row_means                   # (2, 3) - (2, 1) → broadcasts to (2, 3)
```

Without `keepdims`, you'd get shape `(2,)` which can't broadcast against `(2, 3)` the way you want.

### 1.3 Multi-axis Operations

You can collapse multiple axes at once:

```python
x = np.ones((2, 3, 4))
x.sum(axis=(0, 2))        # collapse dims 0 and 2 → shape (3,)
x.sum(axis=(1, 2))        # collapse dims 1 and 2 → shape (2,)
```

### 1.4 Exercises — Axis Reasoning

```python
# E1.1 — Predict output shape for each
x = np.ones((8, 10, 512))
# x.sum(axis=0).shape = ?
# x.sum(axis=1).shape = ?
# x.sum(axis=2).shape = ?
# x.sum(axis=(0, 1)).shape = ?
# x.sum(axis=-1).shape = ?
# x.max(axis=1).shape = ?  (note: np.max vs torch.max differ!)

# E1.2 — Predict shapes
x = np.ones((32, 8, 128, 128))
# x.mean(axis=-1).shape = ?
# x.mean(axis=-1, keepdims=True).shape = ?
# x.sum(axis=(2, 3)).shape = ?

# E1.3 — Which axis to reduce?
# Given scores of shape (batch, heads, seq_len, seq_len) = (32, 8, 128, 128)
# To get attention weights, you softmax over the last dim (keys).
# What axis do you pass to softmax? Answer: axis=___

# E1.4 — Normalize each feature across the batch
x = np.random.randn(100, 5)  # 100 samples, 5 features
# Write code to compute:
#   mean per feature (shape should be (5,) or (1,5))
#   std per feature
#   z-scored x (each feature has mean≈0, std≈1)

# E1.5 — Real scenario: layer normalization
x = np.random.randn(2, 3, 4)  # (batch, seq, features)
# LayerNorm normalizes across the LAST dimension (features).
# Compute mean and std across dim=-1 with keepdims, then normalize.

# E1.6 — argmax reasoning
x = np.array([[0.1, 0.7, 0.2],
              [0.3, 0.3, 0.4],
              [0.9, 0.05, 0.05]])
# x.argmax(axis=0) = ?  (predict values, not just shape)
# x.argmax(axis=1) = ?
```

---

## Level 2: Element-wise Operations

**Key insight:** When two tensors have the SAME shape, `+`, `-`, `*`, `/`, `**` all operate element-by-element. No loops needed.

### 2.1 Same-Shape Operations

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b    # [11, 22, 33]
a * b    # [10, 40, 90]
a / b    # [0.1, 0.1, 0.1]
a ** 2   # [1, 4, 9]

# The loop equivalent (never write this):
# result = np.empty(3)
# for i in range(3):
#     result[i] = a[i] + b[i]
```

### 2.2 Scalar Broadcasting (Simplest Case)

A scalar automatically expands to match any shape:

```python
x = np.array([[1, 2], [3, 4]])   # (2, 2)

x + 5     # [[6,7],[8,9]]       scalar → (2,2)
x * 2     # [[2,4],[6,8]]
x / 10    # [[0.1,0.2],[0.3,0.4]]
1 / x     # [[1, 0.5],[0.33, 0.25]]
```

### 2.3 Common Element-wise Functions

```python
np.exp(x)        # e^x for each element
np.log(x)        # ln(x) for each element
np.sqrt(x)       # √x for each element
np.abs(x)        # |x| for each element
np.maximum(x, 0) # ReLU!
np.clip(x, 0, 1) # clamp to [0, 1]
```

**Sigmoid from scratch (no loops!):**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Works on ANY shape:
sigmoid(np.array([0, 1, -1]))          # (3,)
sigmoid(np.random.randn(32, 10))       # (32, 10)
sigmoid(np.random.randn(2, 3, 4, 5))  # (2, 3, 4, 5)
```

### 2.4 Exercises — Element-wise

```python
# E2.1 — Implement ReLU without any loops or if statements
def relu(x):
    pass  # one line

# E2.2 — Implement softmax along axis=-1 (no loops)
# softmax(x)_i = exp(x_i) / sum(exp(x_j))
# Hint: use keepdims
def softmax(x, axis=-1):
    pass

# E2.3 — Implement cross-entropy loss (no loops)
# L = -sum(y_true * log(y_pred)) / N
# y_true: (N, C) one-hot, y_pred: (N, C) probabilities
def cross_entropy(y_true, y_pred):
    pass

# E2.4 — Predict the output
a = np.array([[1, 2], [3, 4]])
b = np.array([[10, 20], [30, 40]])
# a * b = ?
# a ** b = ?  (element-wise power)
# np.maximum(a, 3) = ?

# E2.5 — GELU activation (approximate version)
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
# Implement without loops:
def gelu(x):
    pass

# E2.6 — Given logits of shape (batch, num_classes) = (4, 5):
logits = np.random.randn(4, 5)
# Compute softmax probabilities (should sum to 1 along axis=1)
# Verify: probs.sum(axis=1) should be [1, 1, 1, 1]
```

---

## Level 3: Reshaping Toolkit

**Key insight:** Reshaping never changes the data, only how dimensions are organized. The total number of elements stays the same.

### 3.1 reshape / view

```python
x = np.arange(12)              # shape (12,)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

x.reshape(3, 4)                # (3, 4)
# [[0,  1,  2,  3],
#  [4,  5,  6,  7],
#  [8,  9,  10, 11]]

x.reshape(2, 2, 3)             # (2, 2, 3)
# [[[0, 1, 2],
#   [3, 4, 5]],
#  [[6, 7, 8],
#   [9, 10, 11]]]

x.reshape(4, -1)               # -1 means "infer this dim" → (4, 3)
```

**The -1 trick:** Exactly one dimension can be -1, and NumPy/PyTorch will compute it:
```python
x = np.zeros((2, 3, 4))       # 24 elements
x.reshape(-1)                  # (24,)     — flatten
x.reshape(6, -1)               # (6, 4)    — 24/6 = 4
x.reshape(2, -1)               # (2, 12)   — 24/2 = 12
x.reshape(-1, 2, 3)            # (4, 2, 3) — 24/6 = 4
```

### 3.2 squeeze and unsqueeze

```python
# unsqueeze: add a dimension of size 1
x = np.array([1, 2, 3])       # shape (3,)

x[np.newaxis, :]               # (1, 3) — row vector
x[:, np.newaxis]               # (3, 1) — column vector

# PyTorch equivalent:
t = torch.tensor([1, 2, 3])   # (3,)
t.unsqueeze(0)                 # (1, 3)
t.unsqueeze(1)                 # (3, 1)
t.unsqueeze(-1)                # (3, 1)

# squeeze: remove dimensions of size 1
y = np.zeros((1, 3, 1, 4))    # (1, 3, 1, 4)
y.squeeze()                    # (3, 4)      — remove ALL size-1 dims
np.squeeze(y, axis=0)          # (3, 1, 4)   — remove only dim 0
np.squeeze(y, axis=2)          # (1, 3, 4)   — remove only dim 2
```

### 3.3 transpose and permute

```python
# 2D transpose
x = np.array([[1, 2, 3],
              [4, 5, 6]])      # (2, 3)
x.T                            # (3, 2)

# Higher-D: use transpose/permute to reorder ALL dims
x = np.zeros((2, 3, 4))       # (2, 3, 4)
x.transpose(0, 2, 1)          # (2, 4, 3)  — swap dims 1 and 2
x.transpose(2, 0, 1)          # (4, 2, 3)  — full reorder

# PyTorch:
t = torch.zeros(2, 3, 4)
t.permute(0, 2, 1)            # (2, 4, 3)
t.permute(2, 0, 1)            # (4, 2, 3)
```

**When you need this in practice — multi-head attention:**
```python
# x: (batch, seq_len, num_heads * head_dim)  e.g. (32, 128, 512)
# Goal: reshape to (batch, num_heads, seq_len, head_dim)

x = x.reshape(32, 128, 8, 64)  # (32, 128, 8, 64)  — split last dim
x = x.permute(0, 2, 1, 3)      # (32, 8, 128, 64)  — move heads before seq
#          batch ──┘  │    │  └── head_dim
#               heads─┘    └── seq_len
```

### 3.4 expand and repeat

```python
# expand: broadcast a size-1 dim to larger size (no memory copy)
x = torch.tensor([[1], [2], [3]])  # (3, 1)
x.expand(3, 4)                     # (3, 4) — each row repeated 4 times
# [[1, 1, 1, 1],
#  [2, 2, 2, 2],
#  [3, 3, 3, 3]]

# repeat: actually copies data
x.repeat(1, 4)                     # same result but allocates new memory
x.repeat(2, 3)                     # (6, 3) — 2× along dim0, 3× along dim1
```

### 3.5 Flatten and Unflatten

```python
x = torch.randn(2, 3, 4, 5)

# Flatten specific dims
x.flatten(start_dim=1)             # (2, 60)    — merge dims 1,2,3
x.flatten(start_dim=1, end_dim=2)  # (2, 12, 5) — merge dims 1,2 only
x.flatten(start_dim=0, end_dim=1)  # (6, 4, 5)  — merge dims 0,1

# Unflatten (inverse)
y = torch.randn(2, 60)
y.unflatten(1, (3, 4, 5))          # (2, 3, 4, 5)
```

### 3.6 Exercises — Reshaping

```python
# E3.1 — Predict the output shape
x = np.arange(24)
# x.reshape(2, 3, 4).shape = ?
# x.reshape(6, -1).shape = ?
# x.reshape(-1, 2, 3).shape = ?
# x.reshape(2, 3, 4).reshape(-1).shape = ?

# E3.2 — Predict values
x = np.arange(6).reshape(2, 3)
# x = ?  (write out the array)
# x.T = ?
# x.reshape(3, 2) = ?       ← NOTE: this is NOT the same as x.T!
# Why are x.T and x.reshape(3,2) different?

# E3.3 — Multi-head attention reshape
# Given Q of shape (batch=2, seq=4, d_model=6)
# with num_heads=2, head_dim=3
# Reshape Q to (batch, num_heads, seq, head_dim) = (2, 2, 4, 3)
Q = torch.arange(48, dtype=torch.float).reshape(2, 4, 6)
# Write the reshape + permute:
# Q_heads = ?

# E3.4 — Undo the multi-head reshape
# Given Q_heads of shape (batch=2, num_heads=2, seq=4, head_dim=3)
# Reshape back to (batch=2, seq=4, d_model=6)
# Q_back = ?
# Verify: torch.equal(Q, Q_back)

# E3.5 — Add a batch dimension
x = torch.randn(10, 512)    # (seq_len, d_model) — single sequence
# Add batch dim to get (1, 10, 512)
# Method 1: unsqueeze
# Method 2: reshape
# Method 3: indexing with None

# E3.6 — Image format conversion
img = np.random.randint(0, 255, (224, 224, 3))  # HWC format
# Convert to CHW format: (3, 224, 224)
# img_chw = ?

# E3.7 — Flatten for a linear layer
features = torch.randn(32, 64, 7, 7)  # (batch, channels, H, W)
# Flatten to (32, 64*7*7) = (32, 3136) for a linear layer
# features_flat = ?
```

---

## Level 4: Broadcasting — The Core Skill

**This is the single most important section.** Broadcasting is what replaces most loops.

### 4.1 The Broadcasting Rules

When operating on two tensors with different shapes, NumPy/PyTorch aligns dimensions from the RIGHT and checks compatibility:

**Rule 1:** If tensors have different number of dims, pad the shorter shape with 1s on the LEFT.

**Rule 2:** For each dimension (from right to left), sizes are compatible if they are equal OR one of them is 1.

**Rule 3:** The size-1 dimension gets "stretched" to match the other.

```
Shape A:     (8, 1, 6, 1)
Shape B:        (7, 1, 5)

Step 1 — Pad B:  (1, 7, 1, 5)
Step 2 — Check:
  dim 0:  8 vs 1 → OK (1 stretches to 8)
  dim 1:  1 vs 7 → OK (1 stretches to 7)
  dim 2:  6 vs 1 → OK (1 stretches to 6)
  dim 3:  1 vs 5 → OK (1 stretches to 5)

Result:  (8, 7, 6, 5)
```

### 4.2 Broadcasting Visualized

```python
# Example 1: (3,) + (1,) → (3,)
[1, 2, 3] + [10] = [11, 12, 13]
#  10 stretches to [10, 10, 10]

# Example 2: (3, 1) + (1, 4) → (3, 4)
[[1],      [[10, 20, 30, 40]]     [[11, 21, 31, 41],
 [2],  +                       =   [12, 22, 32, 42],
 [3]]                              [13, 23, 33, 43]]

# Column stretches right, row stretches down

# Example 3: (2, 3) + (3,) → (2, 3)
[[1, 2, 3],     [10, 20, 30]     [[11, 22, 33],
 [4, 5, 6]]  +               =   [14, 25, 36]]
# The (3,) vector is added to EACH ROW

# Example 4: (2, 3) + (2, 1) → (2, 3)
[[1, 2, 3],     [[10],     [[11, 12, 13],
 [4, 5, 6]]  +   [20]]  =   [24, 25, 26]]
# The (2,1) column is added to EACH COLUMN within its row
```

### 4.3 Broadcasting Incompatibility

```
Shape A: (3, 4)
Shape B: (5,)

dim -1: 4 vs 5 → INCOMPATIBLE (neither is 1, and they're not equal)
→ ERROR!
```

### 4.4 The Outer Product Pattern

One of the most powerful broadcasting patterns:

```python
# Computing all pairwise differences
a = np.array([1, 2, 3])[:, np.newaxis]   # (3, 1)
b = np.array([10, 20, 30])[np.newaxis, :] # (1, 3)

a - b
# (3, 1) - (1, 3) → (3, 3)
# [[ -9, -19, -29],
#  [ -8, -18, -28],
#  [ -7, -17, -27]]

# This replaces the nested loop:
# for i in range(3):
#     for j in range(3):
#         result[i,j] = a[i] - b[j]
```

### 4.5 Common Broadcasting Patterns in ML

**Pattern 1: Subtracting mean per row**
```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])          # (2, 3)
mu = x.mean(axis=1, keepdims=True) # (2, 1)
x_centered = x - mu                # (2, 3) - (2, 1) → (2, 3)
```

**Pattern 2: Adding bias to all batch elements**
```python
out = np.random.randn(32, 512)     # (batch, features)
bias = np.random.randn(512)         # (features,)
out + bias                           # (32, 512) + (512,) → (32, 512)
# bias added to every sample in the batch
```

**Pattern 3: Scaling attention scores**
```python
scores = np.random.randn(32, 8, 128, 128)  # (B, H, T, T)
scale = 1 / np.sqrt(64)                      # scalar
scores = scores * scale                       # scalar broadcasts everywhere
```

**Pattern 4: Applying a mask**
```python
scores = np.random.randn(32, 8, 128, 128)   # (B, H, T, T)
mask = np.triu(np.ones((128, 128)), k=1)     # (T, T) upper triangle
scores = scores - 1e9 * mask                  # (B,H,T,T) - (T,T) → broadcasts!
```

### 4.6 Exercises — Broadcasting

```python
# E4.1 — Will these broadcast? If yes, what's the result shape?
# a: (5, 3) + b: (3,)         → ?
# a: (5, 3) + b: (5,)         → ?
# a: (5, 3) + b: (1, 3)       → ?
# a: (5, 3) + b: (5, 1)       → ?
# a: (2, 3, 4) + b: (3, 4)    → ?
# a: (2, 3, 4) + b: (3, 1)    → ?
# a: (2, 3, 4) + b: (2, 1, 1) → ?
# a: (2, 3, 4) + b: (5, 3, 4) → ?
# a: (8, 1, 6, 1) + b: (7, 1, 5) → ?

# E4.2 — Compute the outer product of two vectors WITHOUT loops
a = np.array([1, 2, 3, 4])    # (4,)
b = np.array([10, 20, 30])    # (3,)
# Result should be shape (4, 3) where result[i,j] = a[i] * b[j]
# outer = ?

# E4.3 — Pairwise Euclidean distance matrix
# Given two sets of points:
A = np.random.randn(100, 3)   # 100 points in 3D
B = np.random.randn(50, 3)    # 50 points in 3D
# Compute distance matrix D of shape (100, 50)
# where D[i,j] = ||A[i] - B[j]||
# Hint: reshape A to (100, 1, 3) and B to (1, 50, 3)
# D = ?

# E4.4 — Normalize each row to sum to 1 (without loops)
x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)
# x_normalized = ?
# Verify: x_normalized.sum(axis=1) should be [1, 1, 1]

# E4.5 — Causal mask for attention
# Create a mask where mask[i,j] = 0 if j <= i, else -inf
# For seq_len = 4:
# [[ 0, -inf, -inf, -inf],
#  [ 0,    0, -inf, -inf],
#  [ 0,    0,    0, -inf],
#  [ 0,    0,    0,    0]]
seq_len = 4
# mask = ?
# Hint: np.triu with k=1, then multiply by -inf

# E4.6 — Temperature scaling
# Given logits of shape (batch, vocab_size) = (4, 1000)
# and temperature T (scalar), compute logits / T
# Then apply softmax along axis=-1
logits = np.random.randn(4, 1000)
T = 0.7
# scaled_probs = ?

# E4.7 — Batch-wise weighted sum
# weights: (batch, 3)    — 3 weights per sample
# values:  (batch, 3, dim) — 3 vectors per sample, each of dimension dim
# Goal: weighted sum → (batch, dim)
# result[b, d] = sum_i weights[b, i] * values[b, i, d]
weights = np.random.randn(4, 3)
values = np.random.randn(4, 3, 8)
# Hint: unsqueeze weights to (4, 3, 1), multiply, sum over axis=1
# result = ?

# E4.8 — Predict the output (compute by hand)
a = np.array([[1], [2], [3]])    # (3, 1)
b = np.array([[10, 20, 30]])     # (1, 3)
# a + b = ?
# a * b = ?
# a - b = ?
```

---

## Level 5: Reduction Operations

### 5.1 The Full Toolkit

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])   # (2, 3)

# Reduce to scalar (no axis)
x.sum()           # 21
x.mean()          # 3.5
x.max()           # 6
x.min()           # 1
x.prod()          # 720

# Reduce along axis
x.sum(axis=0)     # [5, 7, 9]      shape (3,)
x.sum(axis=1)     # [6, 15]         shape (2,)
x.max(axis=0)     # [4, 5, 6]      shape (3,)
x.argmax(axis=1)  # [2, 2]          shape (2,) — indices of max

# PyTorch differences
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
t.max(dim=1)      # returns (values, indices) — a named tuple!
t.max(dim=1).values   # tensor([3, 6])
t.max(dim=1).indices  # tensor([2, 2])
```

### 5.2 Cumulative Operations

```python
x = np.array([1, 2, 3, 4, 5])
np.cumsum(x)     # [1, 3, 6, 10, 15]
np.cumprod(x)    # [1, 2, 6, 24, 120]

# Along an axis
x = np.array([[1, 2, 3],
              [4, 5, 6]])
np.cumsum(x, axis=0)   # [[1,2,3], [5,7,9]]
np.cumsum(x, axis=1)   # [[1,3,6], [4,9,15]]
```

### 5.3 Exercises — Reductions

```python
# E5.1 — Implement logsumexp without loops
# logsumexp(x, axis) = log(sum(exp(x), axis))
# But numerically stable version:
# logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
def logsumexp(x, axis=-1, keepdims=False):
    pass

# E5.2 — Predict output (by hand)
x = np.array([[3, 1, 4],
              [1, 5, 9],
              [2, 6, 5]])
# x.max(axis=0) = ?
# x.argmax(axis=0) = ?
# x.max(axis=1) = ?
# x.argmax(axis=1) = ?
# x.sum(axis=0) - x.sum(axis=1) = ?   ← what shape?

# E5.3 — Top-k selection
x = torch.randn(4, 1000)   # (batch, vocab)
# Get top-5 values and indices per sample
# top_vals, top_idx = ?

# E5.4 — Boolean reductions
x = np.array([[1, -2, 3], [-4, 5, -6]])
# How many positive elements total?
# How many positive elements per row?
# Are ALL elements in row 0 positive?
# Is ANY element in row 1 positive?

# E5.5 — Variance from scratch (no np.var)
x = np.random.randn(100, 5)
# Compute variance of each feature (axis=0)
# var = mean((x - mean)^2)
# my_var = ?
# Verify: np.allclose(my_var, x.var(axis=0))

# E5.6 — Attention weights normalization
# Given raw scores (batch, heads, q_len, k_len) = (2, 4, 10, 10)
scores = np.random.randn(2, 4, 10, 10)
# Apply causal mask (upper triangle = -1e9)
# Apply softmax along the LAST axis
# Verify: weights.sum(axis=-1) is all 1s
```

---

## Level 6: Indexing — The Hard Part

### 6.1 Basic Slicing

```python
x = np.arange(10)             # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x[3]          # 3          — single element
x[2:5]        # [2, 3, 4]  — slice [start:stop)
x[:3]         # [0, 1, 2]  — from beginning
x[7:]         # [7, 8, 9]  — to end
x[::2]        # [0, 2, 4, 6, 8]  — every 2nd
x[::-1]       # [9, 8, 7, ..., 0] — reversed
x[-3:]        # [7, 8, 9]  — last 3
```

### 6.2 Multi-dimensional Slicing

```python
x = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

x[1, 2]       # 6              — single element
x[1]          # [4, 5, 6, 7]   — entire row 1
x[:, 2]       # [2, 6, 10]     — entire column 2
x[0:2, 1:3]   # [[1, 2], [5, 6]]  — submatrix
x[:, -1]      # [3, 7, 11]     — last column
```

**The `:` in each position means "take all of this dimension":**
```python
x = np.zeros((2, 3, 4, 5))

x[0]             # shape (3, 4, 5)  — first along dim 0
x[:, 1]          # shape (2, 4, 5)  — second along dim 1
x[:, :, 2]       # shape (2, 3, 5)  — third along dim 2
x[0, :, :, 3]    # shape (3, 4)     — fix dim 0 and dim 3
x[..., 0]        # shape (2, 3, 4)  — ... means "all remaining dims"
```

### 6.3 Boolean Indexing

```python
x = np.array([10, 20, 30, 40, 50])

mask = x > 25          # [False, False, True, True, True]
x[mask]                # [30, 40, 50]
x[x > 25]             # [30, 40, 50]  — same, inline

# 2D
x = np.array([[1, 2], [3, 4], [5, 6]])
x[x > 3]              # [4, 5, 6]  — always returns 1D!

# Setting values with boolean mask
x[x > 3] = 0
# [[1, 2], [3, 0], [0, 0]]
```

**Masking in attention:**
```python
scores = np.random.randn(4, 4)
mask = np.triu(np.ones((4, 4), dtype=bool), k=1)  # upper triangle
scores[mask] = -1e9
# Now scores has -1e9 wherever the mask is True
```

### 6.4 Fancy (Advanced) Indexing

```python
x = np.array([10, 20, 30, 40, 50])

# Index with an array of indices
idx = np.array([0, 3, 4])
x[idx]                 # [10, 40, 50]

# 2D fancy indexing
x = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

rows = np.array([0, 1, 2])
cols = np.array([1, 3, 2])
x[rows, cols]          # [1, 7, 10]  — elements at (0,1), (1,3), (2,2)
```

**Critical ML use case — gather operation (embedding lookup):**
```python
# Embedding table:  (vocab_size, d_model) = (1000, 512)
# Token indices:    (batch, seq_len) = (32, 128)
embedding_table = np.random.randn(1000, 512)
token_ids = np.random.randint(0, 1000, (32, 128))

# Gather embeddings:
embeddings = embedding_table[token_ids]   # shape: (32, 128, 512)
# This replaces:
# for b in range(32):
#     for s in range(128):
#         embeddings[b, s] = embedding_table[token_ids[b, s]]
```

### 6.5 torch.gather and torch.scatter

```python
# gather: select elements along a dim using index tensor
src = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])    # (2, 3)
idx = torch.tensor([[0, 2],
                    [1, 0]])       # (2, 2)

# gather(dim=1): for each row, pick columns specified by idx
torch.gather(src, dim=1, index=idx)
# row 0: pick cols [0, 2] → [1, 3]
# row 1: pick cols [1, 0] → [5, 4]
# result: [[1, 3], [5, 4]]  shape (2, 2)
```

**How to think about gather:**
```
output[i][j] = src[i][ idx[i][j] ]    when dim=1
output[i][j] = src[ idx[i][j] ][j]    when dim=0
```

### 6.6 Exercises — Indexing

```python
# E6.1 — Predict the output
x = np.arange(20).reshape(4, 5)
# x[1:3, 2:4] = ?
# x[:, ::2] = ?
# x[-1] = ?
# x[:, -1] = ?

# E6.2 — Boolean mask
x = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
# x[x < 0] = ?
# Count negative elements per row (no loops):
# neg_per_row = ?

# E6.3 — Embedding lookup
vocab_size, d_model = 50, 8
embedding = np.random.randn(vocab_size, d_model)
sentence = np.array([5, 12, 3, 45, 0])   # 5 token ids
# Look up embeddings (result should be (5, 8))
# emb = ?

# E6.4 — Argmax then gather
logits = torch.randn(4, 10)    # (batch, classes)
predictions = logits.argmax(dim=1)  # (batch,)
# What shape is predictions?
# Get the actual max values using gather:
# max_vals = ?

# E6.5 — Set diagonal to zero
x = np.ones((5, 5))
# Set diagonal to 0 without loops
# Hint: np.diag_indices or boolean mask
# x = ?

# E6.6 — Top-k masking (used in nucleus sampling)
logits = torch.randn(1, 100)
k = 10
top_vals, top_idx = logits.topk(k, dim=-1)
# Create a mask that's True only for top-k positions
# Set all non-top-k logits to -inf
# mask = ?
# masked_logits = ?

# E6.7 — Advanced: Implement batched index select
# Given:
#   x: (batch, seq_len, dim) = (2, 10, 8)
#   idx: (batch, k) = (2, 3)  — 3 selected positions per batch
# Select the specified positions:
#   result: (batch, k, dim) = (2, 3, 8)
x = torch.randn(2, 10, 8)
idx = torch.tensor([[1, 4, 7],
                    [0, 2, 9]])
# result = ?
# Hint: need to expand idx to match dim dimension
```

---

## Level 7: Matrix Multiplication

### 7.1 The @ Operator and matmul

```python
# 2D: standard matrix multiply
A = np.random.randn(3, 4)    # (3, 4)
B = np.random.randn(4, 5)    # (4, 5)
C = A @ B                     # (3, 5)
# C[i,j] = sum_k A[i,k] * B[k,j]

# 1D @ 2D: vector-matrix
v = np.random.randn(4)        # (4,)
A = np.random.randn(4, 5)     # (4, 5)
v @ A                          # (5,)  — treated as row vector

# 2D @ 1D: matrix-vector
A = np.random.randn(3, 4)     # (3, 4)
v = np.random.randn(4)        # (4,)
A @ v                          # (3,)  — treated as column vector
```

### 7.2 Batched Matrix Multiplication

```python
# 3D: batched matmul
A = np.random.randn(32, 10, 64)    # (batch, m, k)
B = np.random.randn(32, 64, 20)    # (batch, k, n)
C = A @ B                           # (32, 10, 20)
# Equivalent to:
# for b in range(32):
#     C[b] = A[b] @ B[b]

# 4D: multi-head attention
Q = np.random.randn(32, 8, 128, 64)   # (B, H, T, d)
K = np.random.randn(32, 8, 128, 64)   # (B, H, T, d)
scores = Q @ K.transpose(0, 1, 3, 2)  # ... wait, let me be precise

# In NumPy: swap last two dims
KT = np.swapaxes(K, -2, -1)           # (32, 8, 64, 128)
scores = Q @ KT                        # (32, 8, 128, 128)

# In PyTorch:
Q = torch.randn(32, 8, 128, 64)
K = torch.randn(32, 8, 128, 64)
scores = Q @ K.transpose(-2, -1)       # (32, 8, 128, 128)
```

**The matmul broadcasting rule:** For tensors with >2 dims, matmul treats the last two dims as the matrix dims and broadcasts over the rest.

```python
A = torch.randn(32, 8, 128, 64)   # (32, 8, 128, 64)
B = torch.randn(64, 128)           # (64, 128)
# A @ B → broadcast B as (1, 1, 64, 128)
# Result: (32, 8, 128, 128)
```

### 7.3 Exercises — Matrix Multiplication

```python
# E7.1 — Predict output shapes
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(5, 2)
# (A @ B).shape = ?
# (A @ B @ C).shape = ?

# E7.2 — Batched matmul shapes
A = torch.randn(16, 10, 32)
B = torch.randn(16, 32, 20)
# (A @ B).shape = ?

# E7.3 — Will this work? If yes, what shape?
A = torch.randn(8, 4, 10, 64)
B = torch.randn(8, 4, 64, 10)
# (A @ B).shape = ?

# E7.4 — Will this broadcast?
A = torch.randn(8, 1, 10, 64)
B = torch.randn(4, 64, 10)
# (A @ B).shape = ?

# E7.5 — Linear layer from scratch (no nn.Linear)
# y = xW^T + b
x = torch.randn(32, 784)           # (batch, in_features)
W = torch.randn(256, 784)          # (out_features, in_features)
b = torch.randn(256)               # (out_features,)
# Compute y of shape (32, 256):
# y = ?

# E7.6 — Attention computation
# Q, K, V: (batch, seq_len, d_model) = (2, 5, 8)
Q = torch.randn(2, 5, 8)
K = torch.randn(2, 5, 8)
V = torch.randn(2, 5, 8)
d_k = 8
# Step 1: scores = Q @ K^T / sqrt(d_k)     → shape?
# Step 2: weights = softmax(scores, dim=-1) → shape?
# Step 3: output = weights @ V              → shape?
# Implement all three steps:

# E7.7 — Bilinear form
# Compute x^T A y for batched inputs
# x: (batch, m) = (4, 3)
# A: (m, n) = (3, 5)
# y: (batch, n) = (4, 5)
# result: (batch,)  where result[b] = x[b] @ A @ y[b]
x = torch.randn(4, 3)
A = torch.randn(3, 5)
y = torch.randn(4, 5)
# result = ?
```

---

## Level 8: Einsum — The Universal Tool

### 8.1 Einsum Basics

`einsum` uses a string notation to describe tensor operations. Think of it as:
"For each combination of indices, multiply the specified elements, then sum over indices not in the output."

```python
# The notation: 'input_indices -> output_indices'
# Repeated index = those dims must match and get summed over

# Matrix multiply:  C[i,j] = sum_k A[i,k] * B[k,j]
np.einsum('ik,kj->ij', A, B)

# Breakdown:
#   'ik'  — A has indices i,k
#   'kj'  — B has indices k,j
#   'ij'  — output has indices i,j
#   k appears in inputs but NOT in output → summed over
```

### 8.2 Einsum as a Loop Translator

**The key mental model: einsum IS the loop you're thinking of, just written compactly.**

```python
# YOUR LOOP THINKING:
# for i in range(M):
#     for j in range(N):
#         C[i,j] = sum over k of A[i,k] * B[k,j]

# EINSUM TRANSLATION:
# Step 1: What are the free indices (appear in output)? → i, j
# Step 2: What are the summed indices (appear only in inputs)? → k
# Step 3: Write it:  'ik,kj->ij'
C = np.einsum('ik,kj->ij', A, B)
```

### 8.3 Common Einsum Patterns

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)
w = np.random.randn(3)

# --- Dot products and traces ---
np.einsum('i,i->', v, v)           # dot product: sum_i v[i]*v[i]
np.einsum('ii->', A[:3,:3])        # trace: sum_i A[i,i] (square matrix)

# --- Matrix operations ---
np.einsum('ij->ji', A)             # transpose
np.einsum('ij->', A)               # sum all elements
np.einsum('ij->i', A)              # sum each row
np.einsum('ij->j', A)              # sum each column
np.einsum('ij,j->i', A, v)        # matrix-vector: A @ v
np.einsum('ik,kj->ij', A, B)      # matmul: A @ B

# --- Outer product ---
np.einsum('i,j->ij', w, v)        # outer product: w[:, None] * v[None, :]

# --- Element-wise multiply then sum ---
np.einsum('ij,ij->', A, A)        # Frobenius norm squared
np.einsum('ij,ij->i', A, A)       # row-wise dot product

# --- Batch operations ---
A = np.random.randn(32, 10, 64)
B = np.random.randn(32, 64, 20)
np.einsum('bik,bkj->bij', A, B)   # batched matmul

# --- Multi-head attention ---
Q = np.random.randn(32, 8, 128, 64)   # (B, H, T, D)
K = np.random.randn(32, 8, 128, 64)
np.einsum('bhid,bhjd->bhij', Q, K)    # attention scores
# "For each (b,h), compute Q[i,:] dot K[j,:]"
```

### 8.4 Exercises — Einsum

```python
# E8.1 — Translate these loops to einsum
# Loop 1: c[i] = sum_j A[i,j] * b[j]
# einsum: ?

# Loop 2: C[i,j] = A[i,j] * B[i,j]  (element-wise, Hadamard product)
# einsum: ?

# Loop 3: c = sum_i sum_j A[i,j]
# einsum: ?

# Loop 4: D[i,j,k] = A[i,j] * B[j,k]  (no sum — outer-product-like)
# einsum: ?

# E8.2 — Predict the output shape
A = np.random.randn(2, 3)
B = np.random.randn(3, 4)
C = np.random.randn(4, 5)
# np.einsum('ij,jk->ik', A, B).shape = ?
# np.einsum('ij,jk,kl->il', A, B, C).shape = ?
# np.einsum('ij,jk->ijk', A, B).shape = ?

# E8.3 — Implement batched bilinear form with einsum
# x: (batch, m), A: (m, n), y: (batch, n)
# result: (batch,)  where result[b] = sum_i sum_j x[b,i] * A[i,j] * y[b,j]
x = np.random.randn(4, 3)
A = np.random.randn(3, 5)
y = np.random.randn(4, 5)
# result = np.einsum(?, x, A, y)

# E8.4 — Multi-head attention with einsum
# Q: (B, H, T, D) = (2, 4, 10, 16)
# K: (B, H, T, D) = (2, 4, 10, 16)
# V: (B, H, T, D) = (2, 4, 10, 16)
Q = np.random.randn(2, 4, 10, 16)
K = np.random.randn(2, 4, 10, 16)
V = np.random.randn(2, 4, 10, 16)
# Step 1: scores[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]
# scores = np.einsum(?, Q, K)
# Step 2: after softmax → weights (B, H, T, T)
# Step 3: output[b,h,i,d] = sum_j weights[b,h,i,j] * V[b,h,j,d]
# output = np.einsum(?, weights, V)

# E8.5 — Tensor contraction
# Given A: (2, 3, 4) and B: (4, 3, 5)
# Compute C[i, l] = sum_j sum_k A[i,j,k] * B[k,j,l]
A = np.random.randn(2, 3, 4)
B = np.random.randn(4, 3, 5)
# C = np.einsum(?, A, B)
# What is C.shape?
```

---

## Level 9: Putting It All Together — Transformer Patterns

### 9.1 Full Self-Attention (No Loops)

```python
import torch
import torch.nn.functional as F

def self_attention(x, W_q, W_k, W_v, num_heads):
    """
    x:    (batch, seq_len, d_model)
    W_q:  (d_model, d_model)
    W_k:  (d_model, d_model)
    W_v:  (d_model, d_model)
    """
    B, T, D = x.shape
    head_dim = D // num_heads

    # Step 1: Linear projections — (B, T, D) @ (D, D) → (B, T, D)
    Q = x @ W_q.T                        # (B, T, D)
    K = x @ W_k.T                        # (B, T, D)
    V = x @ W_v.T                        # (B, T, D)

    # Step 2: Reshape to multi-head — (B, T, D) → (B, T, H, d) → (B, H, T, d)
    Q = Q.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    #    (B, H, T, d)

    # Step 3: Attention scores — (B,H,T,d) @ (B,H,d,T) → (B,H,T,T)
    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)

    # Step 4: Causal mask
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))

    # Step 5: Softmax along last dim — (B,H,T,T)
    weights = F.softmax(scores, dim=-1)

    # Step 6: Weighted sum — (B,H,T,T) @ (B,H,T,d) → (B,H,T,d)
    out = weights @ V

    # Step 7: Concatenate heads — (B,H,T,d) → (B,T,H,d) → (B,T,D)
    out = out.permute(0, 2, 1, 3).reshape(B, T, D)

    return out
```

**Trace the shapes through every line.** This is the exercise.

### 9.2 Layer Normalization (No Loops)

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x:     (batch, seq_len, d_model)
    gamma: (d_model,)
    beta:  (d_model,)
    """
    mean = x.mean(dim=-1, keepdim=True)     # (B, T, 1)
    var = x.var(dim=-1, keepdim=True)        # (B, T, 1)
    x_norm = (x - mean) / torch.sqrt(var + eps)   # (B, T, D) broadcasting
    return gamma * x_norm + beta             # (D,) broadcasts over (B, T, D)
```

### 9.3 Position Encoding (No Loops)

```python
def sinusoidal_pe(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)         # (T, 1)
    dim = torch.arange(0, d_model, 2).float()        # (D/2,)
    angles = pos / (10000 ** (dim / d_model))         # (T, 1) / (D/2,) → (T, D/2)

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)     # even indices
    pe[:, 1::2] = torch.cos(angles)     # odd indices
    return pe                            # (T, D)
```

### 9.4 Exercises — Transformer Components

```python
# E9.1 — Shape tracing
# Given x: (2, 5, 16), num_heads=4, head_dim=4
# Trace ALL shapes through self_attention above.
# Write down the shape after EVERY operation.

# E9.2 — Implement a feed-forward network (no loops)
# FFN(x) = ReLU(x @ W1.T + b1) @ W2.T + b2
# x: (B, T, D), W1: (4D, D), b1: (4D,), W2: (D, 4D), b2: (D,)
# What are ALL the intermediate shapes?

# E9.3 — Implement scaled dot-product attention using ONLY einsum
# Q, K, V: (B, H, T, D)
# No @, only einsum.

# E9.4 — Cross-attention
# Q comes from decoder: (B, T_dec, D)
# K, V come from encoder: (B, T_enc, D)
# Implement cross-attention. What shape are the scores?
# What shape is the output?

# E9.5 — Implement token embedding + positional encoding
# vocab_size=100, d_model=32, max_len=50
# Input: token_ids (B, T) with values in [0, 100)
# Step 1: look up embeddings → (B, T, 32)
# Step 2: add sinusoidal PE → (B, T, 32)
# What broadcasting happens in step 2?

# E9.6 — KV-cache inference
# During autoregressive generation, K and V grow by 1 token each step.
# past_K: (B, H, t, D)   — cached keys from previous steps
# new_k:  (B, H, 1, D)   — key for current token
# Concatenate: K = cat(past_K, new_k, dim=2) → (B, H, t+1, D)
# Q for current token: (B, H, 1, D)
# scores = Q @ K^T → what shape?
# output → what shape?
```

---

## Level 10: Drill Problems — Mixed Practice

These combine multiple skills. Do them on paper first.

```python
# D1 — Implement batch cosine similarity matrix
# Given A: (B, N, D) and B: (B, M, D)
# Compute similarity[b, i, j] = cos_sim(A[b,i], B[b,j])
# cos_sim(u, v) = (u·v) / (||u|| * ||v||)
# Result: (B, N, M)
# No loops!

# D2 — Implement multi-query attention (MQA)
# Q: (B, H, T, D)  — multiple query heads
# K: (B, 1, T, D)  — single key head (shared)
# V: (B, 1, T, D)  — single value head (shared)
# Compute attention output: (B, H, T, D)
# Key question: how does K broadcast against Q in Q @ K^T?

# D3 — Implement rotary position embedding (RoPE)
# Given x: (B, H, T, D) where D is even
# For each position t and each pair (x[..., 2i], x[..., 2i+1]):
#   x_rot[..., 2i]   = x[..., 2i] * cos(θ) - x[..., 2i+1] * sin(θ)
#   x_rot[..., 2i+1] = x[..., 2i] * sin(θ) + x[..., 2i+1] * cos(θ)
# where θ_i = t / 10000^(2i/D)
# Implement WITHOUT any Python loops.

# D4 — Implement grouped query attention (GQA)
# Q: (B, H, T, D) where H = num_heads (e.g., 32)
# K: (B, G, T, D) where G = num_kv_groups (e.g., 8)
# V: (B, G, T, D)
# Each group of H/G query heads shares one KV head.
# Reshape Q to (B, G, H//G, T, D), then compute attention.

# D5 — Implement the SwiGLU activation
# SwiGLU(x) = (x @ W1) * sigmoid(x @ W_gate) then @ W2
# x: (B, T, D), W1: (D, 4D), W_gate: (D, 4D), W2: (4D, D)
# Trace all shapes.

# D6 — Implement RMSNorm (used in LLaMA instead of LayerNorm)
# RMSNorm(x) = x / RMS(x) * gamma
# RMS(x) = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
# x: (B, T, D), gamma: (D,)

# D7 — Implement causal sliding window attention
# Same as regular attention, but each query can only attend to
# the previous W tokens (window size).
# Create the mask as a combination of causal mask and window mask.
# mask[i, j] = True  if j > i  OR  j < i - W + 1
# Implement for (T, T) and apply to scores (B, H, T, T).

# D8 — From loop to vectorized
# Rewrite this WITHOUT any Python loops:
def loop_version(x, w):
    """x: (N, D), w: (D,) → result: (N,)"""
    N, D = x.shape
    result = np.zeros(N)
    for i in range(N):
        for j in range(D):
            result[i] += x[i, j] * w[j]
    return result
# vectorized_version = ?

# D9 — From loop to vectorized (harder)
# Rewrite WITHOUT loops:
def loop_version_2(Q, K, V):
    """Q,K,V: (T, D) → output: (T, D)"""
    T, D = Q.shape
    output = np.zeros((T, D))
    for i in range(T):
        scores = np.zeros(T)
        for j in range(T):
            if j <= i:
                for d in range(D):
                    scores[j] += Q[i, d] * K[j, d]
                scores[j] /= np.sqrt(D)
        # softmax over valid positions
        exp_s = np.exp(scores[:i+1] - scores[:i+1].max())
        weights = exp_s / exp_s.sum()
        for j in range(i + 1):
            for d in range(D):
                output[i, d] += weights[j] * V[j, d]
    return output

# D10 — Implement batched pairwise dot product
# Given X: (B, N, D)
# Compute S: (B, N, N) where S[b,i,j] = X[b,i] · X[b,j]
# Method 1: using @
# Method 2: using einsum
# Method 3: verify they give the same result
```

---

## Appendix A: Quick Reference Card

```
SHAPE PREDICTION RULES:
─────────────────────────────────
Element-wise (+, -, *, /):  same shape or broadcast
Reduction (sum, mean, max):  removes the specified axis (or keeps as 1 with keepdim)
Matmul (@):                  (..., m, k) @ (..., k, n) → (..., m, n)
Einsum:                      indices not in output get summed

BROADCASTING:
─────────────────────────────────
1. Align shapes from the RIGHT
2. Pad shorter shape with 1s on the LEFT
3. Sizes must be equal or one must be 1
4. Size-1 dims stretch to match

RESHAPE TOOLKIT:
─────────────────────────────────
reshape(new_shape)    — reinterpret dims, -1 = auto
view(new_shape)       — same as reshape (PyTorch, requires contiguous)
transpose(-2, -1)     — swap last two dims
permute(dims)         — arbitrary dim reorder
squeeze(dim)          — remove size-1 dim
unsqueeze(dim)        — add size-1 dim
flatten(start, end)   — merge dim range

COMMON PATTERNS:
─────────────────────────────────
Row normalize:    x / x.sum(-1, keepdim=True)
Outer product:    a[:, None] * b[None, :]
Pairwise dist:    ((A[:, None] - B[None, :]) ** 2).sum(-1).sqrt()
Batch matmul:     (B, M, K) @ (B, K, N) → (B, M, N)
Attention:        (B,H,T,d) @ (B,H,d,T) → (B,H,T,T) → softmax → @ V
Embedding:        table[indices]  where table: (V,D), indices: (B,T) → (B,T,D)
Linear layer:     x @ W.T + b
```

## Appendix B: Common Mistakes and Fixes

```
MISTAKE                              FIX
─────────────────────────────────────────────────────────────────────
x.sum(axis=1) then broadcast      →  x.sum(axis=1, keepdims=True)
x.T on 3D+ tensor                 →  x.transpose(-2, -1) or .permute()
reshape vs transpose confusion     →  reshape reinterprets memory layout,
                                      transpose reorders axes
torch.max returns tuple            →  use .values or .indices
forgot to scale attention          →  scores / sqrt(d_k)
mask wrong dtype                   →  use .bool() for masked_fill
softmax on wrong axis              →  almost always dim=-1 (over keys)
einsum indices mismatch            →  count: every tensor needs the right
                                      number of indices matching its ndim
```

---

**How to train with this document:**

1. Start from Level 0 and work forward.
2. For each exercise, write your answer on paper FIRST.
3. Then verify in a Python REPL.
4. When you get one wrong, don't just read the answer — redo it from scratch.
5. Revisit Levels 4 (broadcasting) and 6 (indexing) repeatedly; they take the longest to internalize.
6. After finishing all levels, implement a full Transformer block from scratch using only the patterns in this guide — zero Python loops.
