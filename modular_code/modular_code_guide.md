# Writing Modular Code
### From Scratch Implementations That Scale

> A practical guide with exercises for building muscle memory

---

## Table of Contents

- [0. Before We Begin: The Basics](#0-before-we-begin-the-basics)
- [1. What Modular Code Actually Means](#1-what-modular-code-actually-means)
- [2. The Interface-First Mindset](#2-the-interface-first-mindset)
- [3. Decomposition Strategies](#3-decomposition-strategies)
- [4. Designing Clean Interfaces](#4-designing-clean-interfaces)
- [5. Error Handling That Helps You Debug](#5-error-handling-that-helps-you-debug)
- [6. Testing From-Scratch Implementations](#6-testing-from-scratch-implementations)
- [7. File and Project Structure](#7-file-and-project-structure)
- [8. Patterns for ML-Specific Modularity](#8-patterns-for-ml-specific-modularity)
- [9. Refactoring Workflow](#9-refactoring-workflow)
- [10. Common Anti-Patterns and Fixes](#10-common-anti-patterns-and-fixes)
- [11. Checklist](#11-checklist)

---

# 0. Before We Begin: The Basics

If you have never thought about code organization beyond "does it run?", this chapter is for you. We start from the absolute ground level: what is a module, what is coupling, what is an interface, and why any of this matters.

## 0.1 What Is a Module?

A module is any self-contained unit of code that does one thing. It can be a function, a class, or a file. The size does not matter. What matters is that it has a clear boundary: stuff goes in, stuff comes out, and the outside world does not need to know how it works inside.

Think of a kitchen blender. You put fruit in, press a button, smoothie comes out. You do not need to understand the motor, the blade angle, or the wiring. That is a module. Now imagine a blender where you have to manually hold the blade at the right angle, connect two wires yourself, and pour water into the motor housing to cool it. That is non-modular code.

```python
# This is a module: clear input, clear output, self-contained
def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

# This is NOT modular: reads global state, prints, mutates
temperature = 25
def convert():
    global temperature
    f = temperature * 9/5 + 32
    print(f'Result: {f}')
    temperature = f
```

### Exercises: Identifying Modules

**E0.1** — Write a function `kg_to_pounds(kg)` that converts kilograms to pounds. Then write a NON-modular version that uses a global variable and prints the result. Compare: which one can you test by just calling it and checking the return value?

**E0.2** — You have this code. Identify how many separate "jobs" it does:
```python
def process(path):
    f = open(path)
    lines = f.readlines()
    total = 0
    for line in lines:
        num = int(line.strip())
        if num > 0:
            total += num
    print(f'Sum of positives: {total}')
    with open('result.txt', 'w') as out:
        out.write(str(total))
```
Write down each job on a separate line. How many functions should this be?

**E0.3** — Rewrite the `process()` function above as 3–4 separate functions where each does exactly one job. The main script should call them in sequence.

**E0.4** — Write a function that takes a list of numbers and returns the mean. Then write another that returns the standard deviation. Then write a third that calls both and returns `(mean, std)`. Notice how each function has one job.

**E0.5** — Take any script you have written in the past. Count how many "jobs" the longest function does. Write down what each job is.

---

## 0.2 What Is Coupling?

Coupling is how much one piece of code depends on another piece. Tight coupling means changing one thing forces you to change another. Loose coupling means you can change one thing independently.

```python
# TIGHT COUPLING: knows internal structure of User
def process_data(user):
    name = user._internal_dict['personal']['first_name']
    age = user._birth_year
    return f'{name} is {2026 - age} years old'

# LOOSE COUPLING: only uses simple values
def process_data(name, age):
    return f'{name} is {age} years old'
```

### Exercises: Coupling

**E0.6** — Here is a function. List every external thing it depends on:
```python
import config
DB = None

def get_user_email(user_id):
    if DB is None:
        raise RuntimeError('DB not initialized')
    row = DB.query(f'SELECT * FROM {config.TABLE_NAME} WHERE id={user_id}')
    return row['contact_info']['emails'][0]
```
Count the coupling points. Rewrite it so it takes only what it needs as arguments.

**E0.7** — Write two functions A and B where A is tightly coupled to B (A reaches into B's internals). Then rewrite them so they are loosely coupled (A only uses B's return value).

**E0.8** — You have a function that computes BMI. Version 1 takes a `Person` object and accesses `person.weight_kg` and `person.height_m`. Version 2 takes `weight_kg` and `height_m` as plain numbers. Write both. Which one works if you switch from a `Person` class to a dictionary? Which one works with no changes?

**E0.9** — Write `format_report(data)` that takes a dictionary with keys `'title'`, `'author'`, `'values'` and returns a formatted string. Now write `format_report_v2(title, author, values)` with separate arguments. Which version breaks if you rename the dictionary keys?

---

## 0.3 What Is an Interface?

An interface is the agreement between two pieces of code: "I will give you X, and you will give me back Y." The most basic interface in Python is a function signature.

```python
def calculate_area(width: float, height: float) -> float:
    return width * height
```

This function's interface says: give me two floats, I return one float. The caller does not care whether it uses multiplication, a lookup table, or a GPU kernel internally.

### Exercises: Interfaces

**E0.10** — Write the interface (just the signature and docstring, NO implementation) for each of these:
- (a) a function that checks if a string is a palindrome
- (b) a function that finds the top-k elements in a list
- (c) a function that merges two sorted lists into one sorted list

**E0.11** — Write two DIFFERENT implementations of the same interface: `sort_numbers(nums: list) -> list`. One uses bubble sort, one uses Python's built-in `sorted()`. The caller should not be able to tell which one it is using.

**E0.12** — Write `compute_distance(x1, y1, x2, y2) -> float`. Now write `compute_distance_v2(point1, point2) -> float` where each point is a tuple `(x, y)`. Which is easier to call with a list of point tuples?

**E0.13** — A bad interface: write a function that takes a single string like `'3,4,multiply'` and returns the result. Then rewrite it with a clean interface: `compute(a, b, operation)`. Why is the second one better?

---

## 0.4 The Pain of Non-Modular Code

Here is a concrete example. One big function:

```python
def do_everything(filepath):
    data = open(filepath).readlines()
    cleaned = []
    for line in data:
        line = line.strip().lower()
        if line and not line.startswith('#'):
            cleaned.append(line.split(','))
    values = [float(row[2]) for row in cleaned]
    mean = sum(values) / len(values)
    variance = sum((v - mean)**2 for v in values) / len(values)
    result = f'Mean: {mean:.2f}, Std: {variance**0.5:.2f}'
    print(result)
    return result
```

The modular version:

```python
def read_csv(filepath):
    with open(filepath) as f:
        return f.readlines()

def clean_lines(lines):
    cleaned = []
    for line in lines:
        line = line.strip().lower()
        if line and not line.startswith('#'):
            cleaned.append(line.split(','))
    return cleaned

def compute_stats(values):
    mean = sum(values) / len(values)
    variance = sum((v - mean)**2 for v in values) / len(values)
    return mean, variance**0.5

def format_stats(mean, std):
    return f'Mean: {mean:.2f}, Std: {std:.2f}'
```

### Exercises: Decomposing Monoliths

**E0.14** — Decompose this monolith into 4+ separate functions:
```python
def analyze_grades(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    students = {}
    for line in lines[1:]:
        name, grade = line.strip().split(',')
        grade = float(grade)
        if name not in students:
            students[name] = []
        students[name].append(grade)
    for name in students:
        avg = sum(students[name]) / len(students[name])
        if avg >= 90: letter = 'A'
        elif avg >= 80: letter = 'B'
        elif avg >= 70: letter = 'C'
        else: letter = 'F'
        print(f'{name}: {avg:.1f} ({letter})')
```

**E0.15** — Decompose this into separate functions:
```python
def report(transactions):
    income = sum(t['amount'] for t in transactions if t['type'] == 'income')
    expenses = sum(t['amount'] for t in transactions if t['type'] == 'expense')
    net = income - expenses
    by_category = {}
    for t in transactions:
        cat = t.get('category', 'other')
        by_category[cat] = by_category.get(cat, 0) + t['amount']
    print(f'Income: {income}, Expenses: {expenses}, Net: {net}')
    for cat, total in sorted(by_category.items()):
        print(f'  {cat}: {total}')
```

**E0.16** — Write a monolithic function that: reads a text file, counts word frequencies, filters to words longer than 3 characters, sorts by frequency, and prints the top 10. Then decompose it into 5 separate functions and a main script that composes them.

**E0.17** — Take your E0.16 modular version. Without changing any function, use `compute_stats` from the earlier example on the word frequency values. This should work because your frequency function returns data, not prints it. If it does not work, your decomposition was not clean enough.

---

## 0.5 The Vocabulary You Need

| Term | Definition |
|------|------------|
| **Module** | A self-contained unit of code with a clear boundary |
| **Interface** | The contract between modules: what goes in and what comes out |
| **Coupling** | How much one module depends on the internals of another. *Lower is better.* |
| **Cohesion** | How related the things inside a module are. *Higher is better.* |
| **Side effect** | Anything a function does besides returning a value: printing, writing files, mutating globals |
| **Pure function** | A function with no side effects. Same input always gives same output |
| **Abstraction** | Hiding complexity behind a simple interface |
| **Encapsulation** | Bundling data and operations together, restricting direct access to internals |
| **Separation of concerns** | Each module handles one responsibility |
| **DRY** | Don't Repeat Yourself — extract repeated code into a function. But duplication is better than the wrong abstraction |

### Exercises: Vocabulary Drills

**E0.18** — For each function below, state whether it is pure or has side effects. If side effects, name them:
```python
# (a)
def add(a, b): return a + b
# (b)
def add_and_log(a, b): print(a+b); return a+b
# (c)
def append_to(lst, item): lst.append(item); return lst
# (d)
def square(x): return x**2
```

**E0.19** — Write a class `BankAccount` with high cohesion (all methods relate to account operations). Then write a class `Utilities` with LOW cohesion that has methods: `send_email()`, `calculate_tax()`, `resize_image()`. Explain why the second class is bad.

**E0.20** — Identify the abstraction in `sorted([3,1,2])`. What complexity is hidden? Now identify the abstraction in `np.linalg.inv(matrix)`. What complexity is hidden there?

**E0.21** — Write a function that violates DRY by having the same 3-line calculation in two places. Then fix it by extracting the repeated logic into a helper function.

---

# 1. What Modular Code Actually Means

Modularity is not about splitting files. It is about designing units of code where each unit has **exactly one reason to change**.

When you implement `Conv2d` from scratch, the convolution math, the parameter initialization, and the shape validation are three separate concerns. If they live in one tangled function, changing the initialization strategy forces you to re-read and re-test the convolution logic.

**The core test:** can you replace one piece without touching the others? Can you swap Xavier init for Kaiming init without modifying your forward pass? Can you change your padding strategy without rewriting your backward pass? If yes, you have modularity.

## 1.1 The Three Pillars

| Pillar | Description |
|--------|-------------|
| **Separation of Concerns** | Each module handles one job. A loss function computes loss. It does not also clip gradients. |
| **Explicit Interfaces** | Modules communicate through well-defined inputs and outputs. No reaching into internal state. |
| **Information Hiding** | Internal implementation details stay internal. The outside world sees only the interface. |

> **Key Insight:** Modularity is not an aesthetic preference. It directly determines how fast you can debug, extend, and trust your code. Every hour spent on modular design saves ten hours of debugging.

### Exercises: Three Pillars

**E1.1** — You have a `Linear` layer class. It currently does three things in `forward()`: validates input shape, computes `x @ W + b`, and prints the output shape for debugging. Rewrite it so each concern is a separate method. The print should be removable without touching the math.

**E1.2** — Write a class `TemperatureConverter` with a private method `_convert(value, formula)` and public methods `to_celsius(f)` and `to_fahrenheit(c)`. The user should not need to know that `_convert` exists. This is information hiding.

**E1.3** — Write two classes: `Tokenizer` (splits text into tokens) and `Vocabulary` (maps tokens to integer IDs). Make them work together but ensure neither reaches into the other's internals. `Tokenizer` returns a list of strings. `Vocabulary` takes a list of strings and returns a list of ints. They communicate only through the list of strings interface.

**E1.4** — You have a function that does: load CSV → drop NaN rows → one-hot encode categorical columns → standardize numerical columns → split into train/test. Write it as one monolith, then identify which "concern" each section belongs to and rewrite as separate functions.

**E1.5** — For each pair, state whether changing A forces a change in B (tight coupling) or not:
- (a) A = weight initialization, B = forward pass logic
- (b) A = loss function formula, B = optimizer step size
- (c) A = data loader batch size, B = model architecture
- (d) A = activation function, B = backward pass of that activation

---

# 2. The Interface-First Mindset

Before writing any implementation, define the contract. What goes in? What comes out? What invariants must hold? This is the single most impactful habit for from-scratch implementations.

## 2.1 Shape Contracts

For tensor operations, the shape contract is your interface. Write it before you write the computation.

```python
class Conv2d:
    """
    Input:  (batch, in_channels, H_in, W_in)
    Output: (batch, out_channels, H_out, W_out)

    H_out = (H_in + 2*padding - dilation*(kernel-1) - 1) // stride + 1
    """
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        self.weight = np.random.randn(out_ch, in_ch, kernel, kernel) * 0.01
        self.bias   = np.zeros(out_ch)
```

### Exercises: Shape Contracts

**E2.1** — Write ONLY the shape contract (docstring + `__init__` signature, no implementation) for:
- (a) `Linear` layer
- (b) `BatchNorm1d`
- (c) `MaxPool2d`
- (d) `Embedding` layer

For each, specify exact input shape, output shape, and parameter shapes.

**E2.2** — Write the shape contract for an LSTM cell. Specify: input shape, hidden state shape, cell state shape, output shapes, and all four gate weight matrix shapes (`W_i`, `W_f`, `W_o`, `W_c`).

**E2.3** — You are given input `(B, 3, 32, 32)`. It goes through:
```
Conv2d(3, 16, 3, padding=1) → BatchNorm2d(16) → ReLU → MaxPool2d(2)
→ Conv2d(16, 32, 3, padding=1) → Flatten → Linear(?, 10)
```
Write the shape after each layer on paper. What is the `?` value?

**E2.4** — Write shape contracts for a multi-head attention module. Specify Q, K, V input shapes, the shape after splitting into heads, the attention weights shape, and the output shape. Use variables: `B=batch`, `T=sequence_length`, `D=model_dim`, `H=num_heads`, `D_k=D//H`.

---

## 2.2 Design by Contract

Add assertions that enforce your contract at runtime. These are not optional safety nets — they are **executable documentation**.

```python
def forward(self, x):
    assert x.ndim == 4, f"Expected 4D input, got {x.ndim}D"
    B, C, H, W = x.shape
    assert C == self.in_ch, f"Channel mismatch: {C} vs {self.in_ch}"

    out = self._convolve(x)

    expected_h = (H + 2*self.padding - self.kernel) // self.stride + 1
    assert out.shape == (B, self.out_ch, expected_h, expected_w)
    return out
```

### Exercises: Design by Contract

**E2.5** — Write `matrix_multiply(A, B)` that asserts: both are 2D, A's columns == B's rows, and the output shape matches `(A.rows, B.cols)`. Use informative error messages that print actual shapes.

**E2.6** — Add input/output assertions to this function:
```python
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```
Assertions should check: output shape equals input shape, all outputs are in `[0, 1]`, outputs sum to 1.0 along the correct axis.

**E2.7** — Write assertions for `cross_entropy_loss(predictions, targets)` where `predictions` is `(B, num_classes)` of probabilities and `targets` is `(B,)` of integer class indices. Check: predictions shape, targets shape, predictions are positive, targets are valid indices.

**E2.8** — Write a decorator `@validate_shapes(input_shape, output_shape)` that wraps any function and checks shapes before and after. Use it on three different functions.

**E2.9** — Write assertions for an `im2col` function: `im2col(input, kernel_h, kernel_w, stride, padding) -> columns`. Assert input is 4D, kernel fits within padded input, and output column matrix has the correct number of rows and columns.

---

# 3. Decomposition Strategies

## 3.1 Vertical Decomposition (Layers)

Stack independent layers. Each layer transforms data and passes it forward.

```python
class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```

### Exercises: Vertical Decomposition

**E3.1** — Implement a `Sequential` class that works with any layer that has `forward(x) -> y` and `backward(grad) -> grad`. Test it with at least 3 different layer types (`Linear`, `ReLU`, `Sigmoid`). Verify that forward then backward produces gradients of the same shape as the input.

**E3.2** — Implement a `Parallel` class where `forward(x)` passes `x` through N layers independently and returns a list of outputs. Write `backward` to route each gradient back through its corresponding layer.

**E3.3** — Implement a `ResidualBlock` that wraps any layer and adds a skip connection: `output = layer.forward(x) + x`. The backward should correctly handle the gradient split. Test with a `Linear` layer.

**E3.4** — Write a `Pipeline` class for data processing (not ML) that chains transformations: `Pipeline(lowercase, remove_punctuation, split_words, remove_stopwords)`. Each transform takes a value and returns a value. Process a paragraph through it.

---

## 3.2 Horizontal Decomposition (Concerns)

Within a single module, separate the *what* from the *how*.

### Exercises: Horizontal Decomposition

**E3.5** — Take this monolithic LSTM forward pass and decompose it into separate private methods:
```python
def forward(self, x, h, c):
    combined = np.concatenate([x, h], axis=1)
    gates = combined @ self.W + self.b
    i = sigmoid(gates[:, :H])
    f = sigmoid(gates[:, H:2*H])
    o = sigmoid(gates[:, 2*H:3*H])
    g = np.tanh(gates[:, 3*H:])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new
```
Split into: `_compute_gates()`, `_apply_forget_gate()`, `_apply_input_gate()`, `_compute_output()`. Each should be a pure computation.

**E3.6** — Write a `DataLoader` class with separate methods for: `_load_file()`, `_parse_rows()`, `_validate_schema()`, `_create_batches()`. The forward-facing method `load(filepath, batch_size)` should call them in order.

**E3.7** — Decompose a `GradientDescent` optimizer into: `_compute_update(param, grad, lr)`, `_clip_gradient(grad, max_norm)`, `_apply_update(param, update)`. Make each a pure function, then write the `step()` method that orchestrates them.

---

## 3.3 Functional Core, Imperative Shell

Separate pure computation from side effects. Pure functions take inputs and return outputs with no state mutation. The shell manages state, IO, and sequencing.

```python
# FUNCTIONAL CORE — pure, testable, no state
def convolve_2d(input, weight, stride, padding):
    padded = np.pad(input, ...)
    return output

def compute_grad_weight(input, grad_output, kernel_size):
    return grad_weight

# IMPERATIVE SHELL — manages state, caching, lifecycle
class Conv2d:
    def forward(self, x):
        self._input_cache = x
        return convolve_2d(x, self.weight, self.stride, self.padding)
```

### Exercises: Functional Core / Imperative Shell

**E3.8** — Rewrite a `Linear` layer using this pattern. Write two pure functions: `linear_forward(x, W, b)` and `linear_backward(grad_out, x, W)`. Then write the class that manages state (caching x, storing W and b, accumulating gradients).

**E3.9** — Write pure functions for ReLU: `relu_forward(x) -> (output, mask)` and `relu_backward(grad_out, mask) -> grad_in`. Then write the class wrapper. Test the pure functions directly with numpy arrays (no class needed).

**E3.10** — Write pure functions for BatchNorm: `bn_forward(x, gamma, beta, running_mean, running_var, training)` and `bn_backward(grad_out, cache)`. The cache should be a tuple/dict of everything needed for backward. Then write the class.

**E3.11** — Take any from-scratch implementation you have done before (Conv1d, LSTM, anything). Identify which parts are pure computation and which are state management. Refactor it into functional core + imperative shell. Test the core functions without instantiating any class.

**E3.12** — Write a pure function `softmax_cross_entropy(logits, targets)` that returns `(loss, grad_logits)` in a single call. No class, no state. Compare how easy this is to test versus a class-based version.

---

# 4. Designing Clean Interfaces

## 4.1 The Principle of Least Surprise

Your API should behave the way a user expects without reading the source code. If your `Conv2d` takes `(out_channels, in_channels, kernel_size)`, it should match PyTorch's signature.

## 4.2 Narrow Interfaces

Pass only what is needed. A function that takes 7 arguments almost certainly knows too much about its callers.

## 4.3 Composition Over Inheritance

Prefer composing simple objects over building deep inheritance hierarchies. Inheritance locks you into a structure; composition lets you mix and match.

### Exercises: Clean Interfaces

**E4.1** — Write `compute_accuracy(predictions, labels)`. It should work whether predictions are:
- (a) class indices
- (b) one-hot vectors
- (c) probability distributions

Design the interface to handle all three cleanly. Hint: detect the format from the shape.

**E4.2** — Refactor this function to have a narrow interface:
```python
def compute_metrics(model, dataloader, device, criterion, num_classes, logger):
    # computes accuracy, precision, recall, f1
    ...
```
What does it actually NEED? Rewrite the signature to take only the minimum inputs.

**E4.3** — Write `NeuralNet` using composition: it holds a list of layers and calls them in sequence. Write another version `NeuralNetInheritance` using inheritance where `Dense` inherits from `Layer` inherits from `Module`. Add a `Dropout` layer to both. Which one requires fewer changes?

**E4.4** — Write `normalize(x, method='minmax')` that supports `'minmax'`, `'zscore'`, and `'l2'`. Then rewrite it as three separate functions: `minmax_normalize(x)`, `zscore_normalize(x)`, `l2_normalize(x)`. Which design is easier to extend with a new method? Which is easier to test?

**E4.5** — Design the interface (signatures only) for a `Tokenizer` that supports:
- (a) character-level tokenization
- (b) word-level tokenization
- (c) BPE tokenization

All three should have the same interface: `encode(text) -> list[int]` and `decode(ids) -> str`.

**E4.6** — Design two versions of a learning rate scheduler:
- (a) `lr_schedule(t, schedule_type, **kwargs)` where `schedule_type` is `'constant'`, `'linear_decay'`, `'cosine'`, `'warmup_cosine'`
- (b) Separate classes `ConstantLR`, `LinearDecayLR`, `CosineLR`, `WarmupCosineLR` each with `get_lr(t)`

Implement both. Which is more extensible?

---

# 5. Error Handling That Helps You Debug

When your from-scratch LSTM produces NaN on step 3000, the error message determines whether you fix it in 5 minutes or 5 hours.

## 5.1 Fail Fast, Fail Loud

```python
def forward(self, x, h_prev, c_prev):
    if np.any(np.isnan(x)):
        raise ValueError(
            f"NaN in input at step {self._step_count}. "
            f"Last valid output range: [{self._last_min:.4f}, {self._last_max:.4f}]"
        )
```

## 5.2 Contextual Error Messages

A good error message includes:
- **What** went wrong (actual values)
- **What** was expected (expected values)
- **Where** it happened (step, layer name)
- **Hint** about what might have caused it

### Exercises: Error Handling

**E5.1** — Write a `Linear` layer `forward()` that raises informative errors for:
- (a) wrong input dimensions
- (b) feature size mismatch
- (c) NaN in input
- (d) extremely large values (> 1e6)

Each error message should include actual values, expected values, and a hint about what might have gone wrong.

**E5.2** — Write `safe_divide(a, b)` that: raises `ValueError` if `b` is zero (with both values printed), warns if result is very large (> 1e10), warns if result is subnormal (< 1e-300). Use Python's `warnings` module.

**E5.3** — Add gradient explosion detection to an RNN. Write `check_gradients(grad, step, layer_name)` that: warns if `max|grad| > 100`, raises if any NaN, logs the gradient norm at each step. Include the step number and layer name in every message.

**E5.4** — Rewrite all bare asserts in this code with contextual messages:
```python
def attention(Q, K, V):
    assert Q.shape == K.shape
    assert Q.shape[-1] == V.shape[-1]
    assert Q.ndim == 3
    scores = Q @ K.transpose(0, 2, 1)
    assert scores.shape[-1] == scores.shape[-2]
    weights = softmax(scores)
    assert np.allclose(weights.sum(-1), 1.0)
    return weights @ V
```

**E5.5** — Write a NaN tracker: a decorator `@track_nan` that wraps any function, checks if any output contains NaN, and if so, saves the inputs that caused it to a debug file. Use it on a sigmoid function and trigger it by passing in a value of `1e308`.

**E5.6** — Write a `ShapeTracer` class that wraps a model and prints the shape at every layer boundary during forward pass. Use it like: `traced_model = ShapeTracer(model); traced_model.forward(x)` should print each intermediate shape.

---

# 6. Testing From-Scratch Implementations

You cannot trust a from-scratch implementation without automated tests. Visual inspection does not scale.

## 6.1 Gradient Checking

The gold standard for verifying a backward pass: compare your analytical gradient to a numerically estimated one.

```python
def gradient_check(layer, x, eps=1e-5, tol=1e-5):
    out = layer.forward(x)
    grad_out = np.ones_like(out)
    analytical = layer.backward(grad_out)

    numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy(); x_plus[idx] += eps
        x_minus = x.copy(); x_minus[idx] -= eps
        numerical[idx] = (
            layer.forward(x_plus) - layer.forward(x_minus)
        ).sum() / (2 * eps)
    layer.forward(x)  # restore cache

    rel_error = np.abs(analytical - numerical) / (
        np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_err = np.max(rel_error)
    assert max_err < tol, f"Gradient check failed: max error = {max_err:.2e}"
```

## 6.2 Known-Answer Tests

Compare your output against a trusted reference (PyTorch, scipy, hand-computed values).

```python
def test_conv2d_matches_pytorch():
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8)
    my_conv = MyConv2d(3, 16, 3, padding=1)

    import torch
    pt_conv = torch.nn.Conv2d(3, 16, 3, padding=1)
    pt_conv.weight.data = torch.tensor(my_conv.weight)
    pt_conv.bias.data   = torch.tensor(my_conv.bias)

    my_out = my_conv.forward(x)
    pt_out = pt_conv(torch.tensor(x, dtype=torch.float32))
    assert np.allclose(my_out, pt_out.detach().numpy(), atol=1e-5)
```

## 6.3 Property-Based Tests

These verify invariants that must always hold, regardless of input:

- **Shape preservation:** BatchNorm output shape == input shape
- **Gradient shape:** `backward()` output shape == `forward()` input shape
- **Numerical range:** Softmax output sums to 1.0 along correct axis
- **Symmetry:** `padding='same'` preserves spatial dimensions

### Exercises: Testing

**E6.1** — Implement `gradient_check` as shown above. Run it on:
- (a) `Linear` layer
- (b) `Sigmoid` activation
- (c) `Tanh` activation
- (d) `ReLU` activation (note: ReLU is not differentiable at 0 — how do you handle this?)

Fix any bugs your gradient check reveals.

**E6.2** — Write a known-answer test for your Softmax implementation. Compare against `scipy.special.softmax` or manually computed values for a small 2×3 input matrix.

**E6.3** — Write property-based tests for a `Linear` layer:
- (a) output shape is always `(batch, out_features)`
- (b) with zero weights and zero bias, output is all zeros
- (c) with identity-like weights, output approximates input
- (d) backward gradient shape matches input shape

Run each test with 10 different random inputs.

**E6.4** — Write a test that verifies your `BatchNorm` produces zero mean and unit variance output (approximately) during training. Run it on 5 different input shapes.

**E6.5** — Write a full test suite for a from-scratch MSE loss:
- (a) gradient check
- (b) known-answer test with hand-computed values
- (c) property test: loss is always >= 0
- (d) property test: loss is 0 when prediction == target
- (e) compare against PyTorch's `MSELoss`

**E6.6** — Write `test_layer_roundtrip(layer, x)` that verifies: after forward + backward, the layer's parameter gradients have the correct shapes (same as the parameter shapes). Run it on `Linear`, `Conv2d`, and `BatchNorm`.

**E6.7** — Implement a numerical Jacobian checker for any layer. Given `layer` and input `x`, compute the full Jacobian matrix numerically (`output_dim × input_dim`), then verify your backward produces the correct vector-Jacobian product for 5 random `grad_output` vectors.

**E6.8** — Write tests for an LSTM cell that verify:
- (a) hidden state stays bounded (no explosion) over 100 forward steps with random input
- (b) gradient check passes for a single step
- (c) output shape matches expected `(batch, hidden_dim)`
- (d) cell state shape matches expected

---

# 7. File and Project Structure

Structure follows decomposition. Each file should correspond to one concern.

## 7.1 Structure for a From-Scratch ML Library

```
mytorch/
├── __init__.py
├── nn/
│   ├── module.py          # base Module class
│   ├── linear.py
│   ├── conv.py
│   ├── rnn.py
│   ├── batchnorm.py
│   ├── activation.py
│   ├── loss.py
│   └── container.py       # Sequential
├── optim/
│   ├── sgd.py
│   └── adam.py
├── functional/
│   ├── conv_ops.py        # pure convolution functions
│   ├── rnn_ops.py         # pure RNN step functions
│   └── math_ops.py        # im2col, col2im, etc.
├── utils/
│   ├── grad_check.py
│   └── shape_utils.py
└── tests/
    ├── test_linear.py
    ├── test_conv.py
    └── ...
```

## 7.2 Import Discipline

Dependencies should flow in **one direction**. Lower layers never import from higher layers.

```
functional/    # depends on: numpy only
nn/            # depends on: functional/, numpy
optim/         # depends on: nn/ (for parameter access), numpy
tests/         # depends on: everything (that's fine)
```

### Exercises: Project Structure

**E7.1** — Create the directory structure above (empty files are fine). Write `__init__.py` for each package that imports the public API. Verify you can do: `from mytorch.nn import Linear, Conv2d, Sequential`.

**E7.2** — Move your `Linear` layer implementation into the structure: pure function in `functional/math_ops.py`, class wrapper in `nn/linear.py`, tests in `tests/test_linear.py`. Run the tests to verify everything still works.

**E7.3** — Draw the dependency graph of your project. For each file, list what it imports. Check for circular dependencies. If you find any, fix them by extracting the shared concept into a lower-level module.

**E7.4** — Create `utils/shape_utils.py` with functions: `compute_conv_output_shape()`, `compute_pool_output_shape()`, `validate_tensor_shape()`. Use these across your `Conv2d` and `MaxPool2d` classes instead of duplicating shape calculations.

**E7.5** — Add a new layer type (e.g., `Dropout`) to your library. Notice which files you need to touch. If you need to modify more than 2 files (the new layer file + `__init__.py`), your structure has a coupling problem. Fix it.

---

# 8. Patterns for ML-Specific Modularity

## 8.1 The Training Loop Pattern

```python
def train(model, optimizer, loss_fn, dataloader, epochs, callbacks=None):
    callbacks = callbacks or []
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            pred  = model.forward(batch_x)
            loss  = loss_fn.forward(pred, batch_y)
            grad  = loss_fn.backward()
            model.backward(grad)
            optimizer.step()
            optimizer.zero_grad()
            for cb in callbacks:
                cb.on_batch_end(epoch, loss)
        for cb in callbacks:
            cb.on_epoch_end(epoch)
```

## 8.2 The Registry Pattern

```python
ACTIVATIONS = {
    'relu':    ReLU,
    'sigmoid': Sigmoid,
    'tanh':    Tanh,
    'gelu':    GELU,
}

def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown: '{name}'. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()
```

## 8.3 Configuration Separation

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_dim: int   = 256
    num_layers: int   = 4
    dropout:    float = 0.1
    activation: str   = 'relu'

def build_model(cfg: ModelConfig):
    layers = []
    for i in range(cfg.num_layers):
        layers.append(Linear(cfg.hidden_dim, cfg.hidden_dim))
        layers.append(get_activation(cfg.activation))
    return Sequential(*layers)
```

### Exercises: ML Patterns

**E8.1** — Write the training loop above. Make it work with your from-scratch `Linear` layer, SGD optimizer, and MSE loss on a toy dataset: `x = random (100, 5)`, `y = x @ true_weights + noise`. Verify loss decreases.

**E8.2** — Write three callback classes:
- (a) `PrintLossCallback` — prints loss every N batches
- (b) `EarlyStoppingCallback` — stops training if loss hasn't improved for K epochs
- (c) `GradientNormCallback` — prints the total gradient norm each batch

Plug all three into the training loop.

**E8.3** — Build a registry for loss functions: `'mse'`, `'cross_entropy'`, `'l1'`. Write `get_loss(name)` that returns an instance. Add a new loss function (`'huber'`) with a single line change.

**E8.4** — Create `ModelConfig` and `TrainConfig` dataclasses. Write `build_model(cfg)` and `train_from_config(model_cfg, train_cfg, data)`. Train a model by changing only the config objects, never the function code.

**E8.5** — Write a registry for optimizers: `'sgd'`, `'adam'`, `'rmsprop'`. Each optimizer should have the same interface: `__init__(params, lr, **kwargs)` and `step()`. Build a model and swap optimizers by changing one string.

**E8.6** — Write a `MetricsTracker` callback that records loss, accuracy, gradient norm at each step, and provides methods: `plot_loss()`, `get_best_epoch()`, `summary()`. Use it in training and print the summary at the end.

**E8.7** — Write a `CheckpointCallback` that saves model parameters to a file every N epochs and can restore from a checkpoint. Interface: `save(filepath)` and `load(filepath)`. The training loop should not need to know about file I/O.

---

# 9. Refactoring Workflow

You will not write modular code on the first try. The workflow is: **make it work → make it right → make it fast**.

## 9.1 The Refactoring Loop

| Step | Action |
|------|--------|
| 1 | **Working prototype** — Write the messiest code that produces correct output |
| 2 | **Add tests** — Lock down the behavior before refactoring |
| 3 | **Extract functions** — Identify the pure computation. Pull it out |
| 4 | **Define interfaces** — Write shape contracts and assertions |
| 5 | **Reorganize** — Move related code into modules |

> **Key Insight:** Never refactor without tests. Tests are the safety net that makes refactoring possible.

## 9.2 When NOT to Modularize

For code you will use once and throw away, monolithic code is fine. Apply modular design to code that will be **reused, extended, shared, or maintained**.

### Exercises: Refactoring Practice

**E9.1** — START MESSY ON PURPOSE. Write a single function that: creates random data (100 points, 2D, two classes), trains a logistic regression from scratch (sigmoid, BCE loss, gradient descent), evaluates accuracy, and prints results. Make it work in one function. Then apply the refactoring loop: add tests first, then extract, then define interfaces.

**E9.2** — Take this messy code and refactor it step by step:
```python
def nn(X, Y, h=10, lr=0.01, iters=1000):
    W1 = np.random.randn(X.shape[1], h) * 0.01
    b1 = np.zeros(h)
    W2 = np.random.randn(h, Y.shape[1]) * 0.01
    b2 = np.zeros(Y.shape[1])
    for i in range(iters):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        loss = np.mean((z2 - Y)**2)
        dz2 = 2*(z2 - Y)/Y.shape[0]
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2
        if i % 100 == 0: print(loss)
    return W1, b1, W2, b2
```
Document your refactoring steps: what you extracted first, what tests you wrote, what interfaces you defined.

**E9.3** — Refactor your E9.2 result so you can easily change:
- (a) number of hidden layers
- (b) activation function
- (c) loss function
- (d) optimizer

Each change should require modifying at most one line of the training script.

**E9.4** — Take any old homework/project code of yours. Apply the refactoring loop. Before refactoring, write 3 tests that capture current behavior. After refactoring, verify all 3 tests still pass.

**E9.5** — Write a messy K-means clustering implementation in one function. Then refactor into: `initialize_centroids()`, `assign_clusters()`, `update_centroids()`, `compute_inertia()`, `kmeans()`. Write tests before refactoring.

---

# 10. Common Anti-Patterns and Fixes

## 10.1 God Object

One class that does everything.

**Fix:** Decompose into single-responsibility classes.

## 10.2 Hidden Dependencies

Module A modifies a global that Module B secretly relies on.

**Fix:** Make all dependencies explicit — pass them as arguments.

## 10.3 Premature Abstraction

Building a generic framework for one use case.

**Fix:** The Rule of Three — abstract on the third repetition, not the first.

## 10.4 Leaky Abstractions

Exposing internal state as public attributes.

**Fix:** Provide methods, keep internals private (prefix with `_`).

## 10.5 Stringly-Typed Interfaces

```python
# BAD — typos fail silently or at runtime with confusing errors
conv = Conv2d(3, 16, 3, padding='smae')

# BETTER — invalid values caught at import time with clear errors
from enum import Enum
class PaddingMode(Enum):
    VALID = 'valid'
    SAME  = 'same'
    FULL  = 'full'
```

### Exercises: Anti-Pattern Detection

**E10.1** — Write a `GodObject` class called `MLPipeline` that: loads data, preprocesses, builds model, trains, evaluates, plots, saves. Then decompose it into at least 5 separate classes, each with one job.

**E10.2** — Write two modules with a hidden dependency (Module B reads a global that Module A sets). Demonstrate the bug that occurs when Module A is not called first. Then fix it by making the dependency explicit.

**E10.3** — You have only `Conv2d`. Resist the urge to write `ConvNd`. Instead, when you later implement `Conv1d`, notice what is actually shared vs different. Only then extract the common pattern. Write `Conv2d` first, then `Conv1d`, then refactor to share code.

**E10.4** — Write a `BatchNorm` class where `running_mean` and `running_var` are public. Write code that directly modifies them from outside. Now make them private and add `train_mode()` and `eval_mode()` methods. Verify the external code no longer needs to touch internals.

**E10.5** — Find 3 examples of stringly-typed interfaces in code you have written or used (e.g., `padding='same'`, `activation='relu'`, `mode='train'`). Rewrite each using an `Enum`. How does this change the error messages when you pass an invalid value?

**E10.6** — Write a class with a leaky abstraction: a `Matrix` class that exposes its internal `self._data` list directly. Write client code that sorts `self._data` in place. Now change the internal representation from list to numpy array. Watch the client code break. Fix the abstraction by providing proper methods.

**E10.7** — Code smell hunt: take any 100+ line script you have. Identify and label every anti-pattern from this chapter. For each one, write a one-sentence fix. Do not implement the fixes yet; just identify and plan.

---

# 11. Checklist

Use this checklist before considering any implementation complete.

- [ ] Every public function/method has a shape contract or type annotation
- [ ] Assertions validate inputs at every module boundary
- [ ] Error messages include actual values, expected values, and context
- [ ] Pure computation is separated from state management
- [ ] Each file has one clear responsibility
- [ ] Dependencies flow in one direction (no circular imports)
- [ ] Gradient check passes for every differentiable module
- [ ] At least one known-answer test exists against a reference implementation
- [ ] Configuration is separate from logic
- [ ] No function exceeds ~50 lines (if it does, decompose it)
- [ ] You can explain what each module does in one sentence

### Capstone Exercises

**E11.1** — Build a complete from-scratch 2-layer MLP library with: `Linear`, `ReLU`, `Sigmoid`, MSE loss, CrossEntropy loss, SGD optimizer, `Sequential` container. Follow every principle from this document. Structure it using the project layout from Chapter 7. Run the checklist above on every module.

**E11.2** — Add `Conv2d` and `MaxPool2d` to your library. Use functional core / imperative shell. Write gradient checks and known-answer tests for both. Verify the checklist passes.

**E11.3** — Add an LSTM cell and a simple RNN to your library. Implement BPTT. Write gradient checks for sequences of length 1, 3, and 10. Use the registry pattern for activation functions within the gates.

**E11.4** — Write a complete training script that uses your library to classify MNIST. The script should use: config dataclasses, the training loop pattern, at least 2 callbacks, and the registry pattern for selecting model architecture. Changing from `'mlp'` to `'cnn'` should require changing one config value.

**E11.5** — Give your library to a friend (or your future self after 2 weeks). Ask them to add a GRU cell using only the public interfaces and the project structure. If they need to read any internal implementation details to do it, your interfaces are not clean enough. Fix them.
