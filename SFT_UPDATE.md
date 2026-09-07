# SFT on a from-scratch NanoGPT — Slice 0 write-up

## Setup

A base model with 6 heads, 8 layers, RoPE, RMSNorm, embed_size = 768, ff_expand = 4,
batch_size = 3, trained on TinyStories. The SFT training loop runs on Gemma 4 e4b pairs.

## Implementation

Each pair has 3 parts:

```
<|storystart|> [tinystories story] <|storyend|>
<|instructionstart|> [instruction: retell, complete, or request] <|instructionend|>
<|modelstart|> [model's response] <|modelend|>
```

We only do backpropagation on the story, because that's the model's natural ability, and on
the response, because that's the model's new ability to learn — but not on the instruction,
because it is given. So we mask the instruction when computing loss.

I had to cap the pairs at 400 tokens for two reasons: my VRAM was not enough, and my model's
sequence length is only 256, so RoPE degrades as we go past the ceiling. I dropped 56
sequences out of 500. Batches were sorted by length before batching, so later batches contain
systematically longer sequences.

## Proof that SFT training improved the model's loss on pairs

I computed the noise floor by running 2 identical base models through the SFT training loop
using the same pairs:

| metric | value |
|---|---|
| mean difference in loss | −0.00034 |
| std | 0.00416 |
| max absolute difference | 0.0156 |

Then I ran 2 models through the SFT training loop: one control model with `lr = 0`, and one
effect model with `lr = 0.00015`.

LR has to be smaller at SFT because embedding matrix rows 0–10000 are already trained, and
rows 10000–10008 are what actually need training. A large step can ruin the already-trained
weights.

The difference was big:

| metric | value |
|---|---|
| mean difference in loss | **−0.231** |
| effect model lower than control | **146 / 148 iterations** |

The noise floor is necessary because if I only look at one raw list of loss, it can bounce
from 3.5 to 4.5 between iterations — unreliable without a second run.

## Evaluation

I then constructed 5 pairs by Gemma and 5 pairs by me (I constructed them before seeing any
output), using the same stories — the stories are the eval set, not the training set. Run
with `max_tokens = 30` and `temperature = 0.5`.

Results:

- SFT emitted `<|modelend|>` at prompt 8
- SFT emitted `<|storyend|>` at prompt 1 (in the wrong place)
- The base model output `<|endoftext|>` and started a new story; the SFT model didn't
- **Neither model responded to the actual instruction** — only TinyStories generation, though
  still with coherent grammar and moderate word connection

This localises the effect: SFT changed the model, but that effect is not whole at this scale.
The model could either be undertrained or just too small for the task. More testing is needed
to isolate them.

## The interesting bit

Loss improved but behaviour barely moved. I suspect that more training or a bigger model
(I don't know which) should make behaviour catch up.