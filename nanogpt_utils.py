"""Shared helpers for the NanoGPT SFT / reward-model notebooks.

Everything here was duplicated across nanogpt.ipynb and reward_model.ipynb.
Import from here instead of redefining.

Note on boundary tokens: several functions need token ids (storystart, pad, ...)
that only exist once an Indicer has been fitted. Rather than import them as
globals, they are passed in explicitly. See set_boundary_tokens() below if you
prefer the global style.
"""

import os
import re
import math

import numpy as np
import cupy as cp
import requests

from reversegradGPU import Tensor


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

PUNCTUATION = ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')',
               '[', ']', '{', '}', '-', '_', '/', '\\']


def preprocess_text(text):
    """Lowercase and space out punctuation, matching the TinyStories format."""
    text = text.lower()
    for char in PUNCTUATION:
        text = text.replace(char, ' ' + char + ' ')
    return text


def split_on(lst, sep):
    """Split a flat token list into sublists on a separator token."""
    result, current = [], []
    for item in lst:
        if item == sep:
            result.append(current)
            current = []
        else:
            current.append(item)
    result.append(current)
    return result


# --------------------------------------------------------------------------
# Autograd / GPU housekeeping
# --------------------------------------------------------------------------

def collect(node, visited=None):
    """Tear down an autograd graph so its tensors can be freed.

    Call after every backward pass, and after every forward pass done outside
    training (e.g. generation) -- otherwise each forward leaves a live graph.
    """
    if visited is None:
        visited = set()
    if id(node) in visited:
        return
    visited.add(id(node))
    for child in node._children:
        collect(child, visited)
    node._children = []
    node._backward = lambda: None


def sync():
    """Block until queued GPU work finishes. Needed before timing anything."""
    cp.cuda.Stream.null.synchronize()


def free_pool():
    """Return CuPy's cached blocks to the driver."""
    cp.get_default_memory_pool().free_all_blocks()


def pool_usage():
    """(live bytes, pool-held bytes) in GB. `live` is real tensors."""
    pool = cp.get_default_memory_pool()
    return pool.used_bytes() / 1e9, pool.total_bytes() / 1e9


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------

def save_model(model, path):
    """Save weights as .npy. Refuses to overwrite."""
    if os.path.exists(path):
        raise FileExistsError(f"Model already exists at {path}")
    raw_weights = [cp.asnumpy(w.data) for w in model.weights]
    np.save(path, np.array(raw_weights, dtype=object), allow_pickle=True)


def load_model(untrained_model, path, reward_model=False):
    """Load weights into a freshly constructed model, in place.

    Mutates and returns the model passed in -- so build a NEW Transformer for
    each model you load, or you will end up with two names for one object.

    reward_model=True skips weight index 1 (the scalar head, which has a
    different shape from the LM head it replaces).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model does not exist at {path}")
    raw_weights = np.load(path, allow_pickle=True)
    for i, w in enumerate(untrained_model.weights):
        if reward_model and i == 1:
            continue
        w.data = cp.asarray(raw_weights[i])
    return untrained_model


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def get_batches(data, batch_size, block_size=256):
    """Random contiguous windows for base-model training. Returns (x, y) on GPU."""
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        i = np.random.randint(0, len(data) - block_size - 1)
        batch_x.append(data[i: i + block_size])
        batch_y.append(data[i + 1: i + block_size + 1])
    return (cp.array(np.stack(batch_x, axis=0)),
            cp.array(np.stack(batch_y, axis=0)))


def pad_sequences(sequences, pad_token):
    """Pad a list of token sequences to the longest one."""
    max_len = max(len(s) for s in sequences)
    return [s + [pad_token] * (max_len - len(s)) for s in sequences]


def generate_mask(sequence, instructionstart, instructionend, modelend):
    """1 where the loss should apply, 0 elsewhere.

    Zeroes the instruction span (it is given, not predicted) and everything
    after modelend (padding). Story and response stay at 1.
    """
    mask = cp.ones(len(sequence))
    mask[sequence.index(instructionstart): sequence.index(instructionend) + 1] = 0
    mask[sequence.index(modelend) + 1:] = 0
    return mask


# --------------------------------------------------------------------------
# Gemma
# --------------------------------------------------------------------------

GEMMA_URL = "http://localhost:12434/api/generate"
GEMMA_MODEL = "gemma4:e4b-it-qat"


def call_gemma(prompt, retries=10, url=GEMMA_URL, model=GEMMA_MODEL, timeout=60):
    """POST to a local Ollama instance. Raises after `retries` failures.

    Prints the reason for each failure rather than swallowing it -- a timeout
    and a dead container look identical otherwise.
    """
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json={"model": model, "prompt": prompt,
                      "stream": False, "think": False},
                timeout=timeout,
            )
            data = response.json()
            if "response" in data:
                return data["response"]
            print(f"Attempt {attempt + 1}: unexpected response: {data}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    raise RuntimeError("Gemma failed after all retries")


PAIR_PATTERN = r"Question\[(.+?)\]:\s*(.+?)\s*Response:\s*(.+?)(?=Question\[|$)"
REWARD_PATTERN = (r"Question\[(.+?)\]:\s*(.+?)\s*Good Response:\s*(.+?)"
                  r"\s*Bad Response:\s*(.+?)(?=Question\[|$)")


def _parse(text, pattern, n_groups):
    """Extract tuples, keeping only the first of each instruction type."""
    seen, parsed = [], []
    for match in re.findall(pattern, text, re.DOTALL):
        if match[0] not in seen:
            seen.append(match[0])
            parsed.append(tuple(match[:n_groups]))
    return parsed


def parse_pairs(text):
    """(type, instruction, response) tuples from a Gemma SFT response."""
    return _parse(text, PAIR_PATTERN, 3)


def parse_pairs_reward(text):
    """(type, instruction, good, bad) tuples from a Gemma preference response."""
    return _parse(text, REWARD_PATTERN, 4)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}


def _z(confidence_level):
    if confidence_level in _Z_TABLE:
        return _Z_TABLE[confidence_level]
    raise ValueError(f"no z value for {confidence_level}; "
                     f"use one of {sorted(_Z_TABLE)}")


def proportion_ci(p_hat, n, confidence_level=0.95):
    """Confidence interval for an observed proportion, e.g. RM accuracy.

    Normal approximation -- unreliable when n*p_hat or n*(1-p_hat) is under ~5.
    """
    if not 0 <= p_hat <= 1:
        raise ValueError("p_hat must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be positive")
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    margin = _z(confidence_level) * se
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def n_to_beat_chance(p_hat, confidence_level=0.95, baseline=0.5):
    """Trials needed for the CI around p_hat to exclude `baseline`.

    Note the squared gap in the denominator: halving the effect you want to
    detect quadruples the samples needed.
    """
    gap = p_hat - baseline
    if gap == 0:
        raise ValueError("p_hat equals baseline; no n can separate them")
    n = (_z(confidence_level) ** 2) * p_hat * (1 - p_hat) / (gap ** 2)
    return math.ceil(n)


def paired_diff(a, b):
    """Summary of a paired comparison. Returns a dict.

    Both runs must have seen the same data in the same order, or the pairing
    is meaningless.
    """
    d = np.asarray(a) - np.asarray(b)
    return {
        "n": len(d),
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "negative": int((d < 0).sum()),
    }
def generate_sft(model,modelend, prompt_seq, max_tokens=60, temperature=0.5):
    text = list(prompt_seq)
    for _ in range(max_tokens):
        out = model.forward(text)
        logits = cp.asnumpy(out.data[-1])
        collect(out)
        del out
        logits = logits / temperature
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        tok = int(np.random.choice(len(p), p=p))
        text.append(tok)
        if tok == modelend:
            break
    return text[len(prompt_seq):]