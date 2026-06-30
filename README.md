# nanoGPT - Transformer from Scratch

nanoGPT is a stripped down transformer library made for clarity and transparency. This project is to show the mechanisms in a transformer using CuPy (GPU-accelerated NumPy), it is for students and hobby learners that are interested in generative AI.

## Features

- Custom autograd engine (ReverseGrad)
- Embedding & Projection layers
- Multi-head Attention with causal masking
- Layer Normalization
- Feed-forward networks
- Adam Optimizer with gradient clipping
- Softmax with backward pass
- Train transformer models on custom datasets
- Generate text from trained models
- Inspect internals of each layer
- Experiment with different hyperparameters
- Compare performance across configurations

## Dependencies
 
- CuPy (with CUDA) - GPU-accelerated array library
- [ReverseGrad](https://github.com/Lucien2468/ReverseGrad) - Custom autograd engine
- [White-Box-ML](https://github.com/Lucien2468/White-Box-ML) - A transparent, interpretable machine learning library
- No PyTorch or TensorFlow needed.
## Architecture

### Model Overview

Takes token indices as input and outputs logits. The model processes tokens through embedding, transformer blocks (with multi-head attention and feed-forward layers), layer normalization, and projection to produce final logits. Users can apply argmax for greedy generation or softmax with temperature sampling for diverse outputs.

### Specifications

For optimal performance with limited RAM, recommended configuration:
- Embedding dimension: 384
- Number of layers: 4
- Number of attention heads: 4
- Vocabulary size: ~12,000
- Context window (block_size): 64

(All values are tunable based on available resources)

### Core Components

1. **Embedding** - Takes token indices and converts them into dense vectors of shape (n_tokens, embed_dim)

2. **Multi-Head Attention** - Uses multiple attention heads in parallel. Each head calculates attention scores between tokens to determine relationships and dependencies. Includes causal masking to prevent attending to future tokens.

3. **Layer Normalization** - Stabilizes the data flow by centering and scaling activations across the embedding dimension.

4. **Feed Forward** - Adds non-linearity and complexity to the representation through dense transformations.

5. **Projection** - Converts the transformer block output from embedding dimension back to vocabulary size to produce logits.

## Training

**Dataset:** 0.4GB of tinystories

**Training Details:**
- Maximum iterations: 100+ (tested, memory stable)
- Learning rate: 0.001
- Batch size: 1
- Optimizer: Adam with gradient clipping (max_norm=2.5)

**Performance:**
- Loss at iteration 0: 9.35
- Loss at iteration 50: 9.30
- Loss at iteration 99: 8.96
- Training time: <1 minute on GPU

## Implementation Highlights

### Custom Autograd

Autograd (ReverseGrad) contains a Tensor class that stores:
- **data** attribute: the actual data (float, numpy array, etc.)
- **gradient** attribute: automatically updated during operations
- **_children** attribute: stores the outputs of operations

Methods implemented include: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__softmax__`, `__var__`, and more.

### Technical Challenges Solved

See [`TECHNICAL_CHALLENGES.md](https://github.com/Lucien2468/NanoGPT/blob/main/TECHNICAL_CHALLENGES.md)

## Known Limitations

- Very simple loss function

Everything is written in CuPy (GPU-accelerated NumPy), so it's easy to fix limitations or bugs you find.

## Usage

### Training

```python
from nanogpt import Transformer, Tokenizer, Indicer
from nanogpt.loss_functions import CrossEntropyLoss
from nanogpt import Optimizer
import cupy as cp

path_to_text = "data/input.txt"
with open(path_to_text, 'r') as f:
    text = f.read()

text = text.lower()
punctuation = ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\']
for char in punctuation:
    text = text.replace(char, ' '+char + ' ')

tokenizer = Tokenizer(vocab_size=10000)
token_map = tokenizer.sequentialize(text)
indicer = Indicer(vocab_size=10000)
indicer.fit(list(token_map))
encoded_tokens = indicer.encode(list(token_map))
vocab_size = 10000

def get_batch(data, block_size=64):
    i = cp.random.randint(0, len(data) - block_size - 1)
    x = data[i : i + block_size]
    y = data[i + 1 : i + block_size + 1]
    return x, y

def train_loop(model, data, iters=100, lr=0.001):
    loss_func = CrossEntropyLoss()
    optimizer = Optimizer(loss_func, model)
    
    for i in range(iters):
        optimizer.zero_grad()
        x, y = get_batch(data)
        output = model.forward(x)
        loss = loss_func.forward(output, y)
        loss.backward()
        optimizer.step(lr)
        
        print(f"Iteration {i}, loss: {loss.data}")
        
        del loss 
        del output
        del x
        del y

model = Transformer(vocab_size, 384, 2, 3072, 2)
train_loop(model, encoded_tokens, iters=350, lr=0.001)
```

### Generation

```python
def generate(model, input_text, indicer, max_tokens=20, temperature=0.001):
    encoded_input = indicer.encode(list(input_text))
    text = list(encoded_input)
    
    for _ in range(max_tokens):
        output = model.forward(text).data
        logits = output[-1]
        
        scaled_logits = logits / temperature
        exps = cp.exp(scaled_logits - cp.max(scaled_logits))
        probabilities = exps / cp.sum(exps)
        token = cp.random.choice(len(probabilities), p=probabilities)
        
        text.append(token)
    
    return indicer.decode(text)

input_text = "to be or not to be"
generated_text = generate(model, input_text, indicer, max_tokens=50, temperature=0.0007)
print(generated_text)
```

## Sample Output

After training on TinyStories:

```
Input: "once upon a time"
Output: "once upon a time , there was a little girl named lily . she loved to play with her toys all day long . one day , her..."
```

## Project Structure

```
nanoGPT/
├── nanogpt/
│   ├── transformer.py
│   ├── optimizer.py
│   ├── tokenizer.py
│   ├── sliding_window.py
│   ├── layers/
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── layernorm.py
│   │   ├── embedding.py
│   │   ├── projection.py
│   │   └── positional_encoding.py
│   └── loss_functions/
│       └── loss_functions.py
├── reversegrad/
│   └── tensor.py
└── README.md
```

## What I Learned

From this project, I learned the basics of backpropagation, memory handling, and the layers of a transformer, especially the attention mechanism.

## Notes

ReverseGrad at [ReverseGrad](https://github.com/Lucien2468/ReverseGrad) had just been updated to work with NanoGPT. The code for NanoGPT will not work without the update.

**All memory problems have been solved**

## Developer

**Lucien** - 11 years old  
Building transformers from scratch to understand how they work.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2025 Lucien

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
