# NanoGPT Build Log

## The Goal
The goal is to get nanoGPT to output coherent and grammatical english.

## Before / After

**First real output:**
```
expect expect expect expect expect....
```

**Output now:**
```
once upon a time , there was a little girl named lily . she loved to play with her toys all day long . one day , her...
```

## The Five Hidden Bugs

### 1. Causal mask
The mask was hiding the diagnal, that means that no token can attend to it self, I quickly fixed this by adding `k=1`
### 2. Missing W_O (output projection)
At multi-head attention, I was just gluing unrelated attention heads side by side, I need a way to mix the heads together, so I used W_O, its a linear layer that takes in the glued head outputs and outputs everything mixed
### 3. Softmax backward
In the softmax backward, dot = np.sum(s*g, axis=-1, ...). Instead of -1, I used 1, it only works with 2D, but it doesn't work with 3D(batching), so I used -1 to handle both cases.
### 4. LayerNorm mean gradient
Instead of dividing by the size of the axis used, I divided by the total size, so my gradient was smaller by seq_len(256)
### 5. The two-bug cancellation (loss + projection softmax)
During training, the softmax was actually not present in the loss function, so it could have been a disaster, but there was a stray softmax in output projection, so the 2 bugs cancle out in the training. But during generation, there was temperature, and it uses a rigged softmax, so I had double softmax during generation, at first, I thought it was no big deal and looked for other bugs causing the word salad, but at one point, I swapped the place of softmax from projection to loss, and it just worked. I wanted to prove that softmax matters a lot, so I ran the model with double softmax and it produced word salad, and then I ran the model with single softmax and it produced grammar, so my conclusion is that even if we upgrade and fix the model into a harvester, we need the softmax fix to bring all the power to the surface
## The Biggest Lesson
Lesson 1: Try to invent a bug that doesn't nessesarily exist, for example, I suspected that there was a bug in loss and that it was evaluating on logits, that lead me to the actual bug in output projection(a stray softmax that was supposed to be in the loss function)
Lesson 2: LOSS IS LYING, when loss is coming down hard, it doesn't nessesarily mean that your model will work out of the box. And when loss looks like its not moving, do not conclude anything unless you looked at the graph of loss and it is really flat, or you actual tested your model(by generating a sample).
