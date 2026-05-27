## Technical Challenges Solved

### 1. Circular References in Computational Graph

**Problem:**
Tensor objects created circular references in the computational graph through `_children`. Neither `del` alone or `gc.collect()` alone could break and delete these references.
**Solution:**
Used both `del` to explicitly remove references AND `gc.collect()` together. The circular references happened in the computational graph through the `_children` of the `Tensor` class.

---

### 2. Attention Heads Accumulation in Forward Pass

**Problem:**
I noticed that memory was increasing linearly during train, this usually means that something was accumulating in the memory between iterations. I found out that each time I called forward, the `MultiheadAttention` class was appending heads to `attn_heads`.

**Solution:**
Moved the initialization of `attn_heads` from the `forward()` method to the `__init__()` method. This way, the attention heads are created once when the layer is initialized, not recreated every forward pass.

---

### 3. Memory Leak from Optimizer Creating New Tensors

**Problem:**
In `optimizer.step()`, I was doing `weight -= Tensor(...)`. This created a new Tensor object every iteration, and the old weight tensors were never deleted. Initially I thought that using .data would break the computational graph but I realised the graph wasn't being used during optimization.

**Solution:**
Changed from `weight -= Tensor(...)` to `weight.data -= ...`. Using `.data` breaks the connection to the computational graph, so the old tensor objects can be freed. Since optimization doesn't need to track gradients for the weight updates themselves, we don't need the graph at this step.

---

### 4. Missing Causal Mask in Attention

**Problem:**
The attention mechanism had no causal mask, which meant tokens could attend to future tokens. In a language model, this is cheating during training as the transformer can look at the next token that it wouldn't have during deployment.

**Solution:**
Added causal masking in the attention mechanism. Before applying softmax, future positions are set to negative infinity (-1e+10), so after softmax they have zero attention weight. This forces each token to only attend to itself and previous tokens.
