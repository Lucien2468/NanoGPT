# Modern Architecture Upgrades — Projects 1–4

Taking my from-scratch nanoGPT and testing each component of the modern LLM stack against my own measurements, one variable at a time.

---

# Project 1 — RoPE vs ALiBi

My custom CuPy-only nanoGPT uses ALiBi as internal positional encoding, so I derived (I had to start over 3 times) the RoPE function, made a class and a `_rotate()` method in attention, and used a gradient check function — and it passed.

I ran it against ALiBi using a `rope=True/False` flag, using the same hyperparameters. A paired t-test run by a tool showed p < 0.001 — meaning if the null hypothesis were true (RoPE and ALiBi are identical, no real difference), the probability of seeing a gap this large just by chance is less than 0.1%. Mean = 0.099.

So there is a **small real average gap favoring RoPE**, but not fully attributable to it because I didn't seed. *(This confound was later closed — see Project 2: the RoPE effect is ~9× the measured noise floor.)*

The plot also showed an upward slope suggesting the gap might grow with more iterations, but when tested, that trend was too weak to trust — mostly noise.

---

# Project 2 — Pre-LN vs Post-LN

## Part 1 — Derivation, implementation, and prediction

I derived why Pre-LN *should* be better and more stable than Post-LN, and I converted my transformer from Post-LN to Pre-LN.

Pre-LN should be better because of the residual connections. Even if the attention/FF has gradient vanishing, `x` will boost the gradient by exactly one. In Post-LN, that residual connection goes through a LayerNorm, so the gradient could vanish in the LayerNorm and get even more vanished across the 8 layers of the transformer. In Pre-LN, the residual connection never touches LayerNorm in backpropagation.

Pre-LN needed a final projection LayerNorm, because the `x` that the projection was taking is unnormalized. This is crucial: if it isn't there, the unnormalized residual sum reaches the projection, then it goes into softmax. The first operation in softmax is exponents, and we are feeding an unnormalized input into it, so it is very likely to explode. So we have to use a third LayerNorm to stabilize it before it goes into the projection and softmax. It will show its importance in one of the later tests.

**My prediction:** Pre-LN will be better than Post-LN by more than RoPE vs ALiBi's gap, and that difference will become bigger as we go more iterations.

## Part 2 — Establishing the noise floor (the seed test)

I was ready to run the comparison test between the 2 models when I remembered something I had set aside. Back at RoPE, the 2 tutors who were teaching me were telling me to consider seed, and I managed to talk out of it and put it on hold for now. So when I was ready to run the tests, I thought about the seed again. Both tutors were telling me to use it, and they said I can test it to end the argument. I decided to test it.

**My prediction:** the mean diff will be under 0.02 and the p will be > 0.5.

I set up 2 models, exact same hyperparameters, but random initialization. I ran them for 200 iterations, and then started to build a paired t-test. One of the tutors stopped me — he said that a paired t-test with 200 samples is very powerful and would see noise as impossible to be exactly that, and return a low p. I completely disagreed and committed to a falsifiable prediction anyway, and let the data settle it.

**Results:** p = 0.4923, mean diff = 0.0112, t = 0.688.

The results proved the tutor right about testing, and that test happened to prove me right. The tutor used the general rule for paired t-tests; what he didn't account for was variance — the loss oscillates a lot. Because of the high variance, the t-stat is 0.688, nowhere near the significance level.

My predicted gap was also right: the gap was 0.0112, that's under 0.02 and indistinguishable from 0. Also the slope is -0.00033, so random initialization doesn't have a difference here that gets bigger as we go farther either.

I just saved myself from spending hours writing a seed parameter for the transformer. I also proved to the second tutor that seed is unnecessary at my scale for effects above ~0.011.

I also can safely say about the RoPE project that the improvement of RoPE over ALiBi is no longer uncertain because of seed — the RoPE effect is ~9× the noise floor.

The conclusion is that random initialization does not influence cause any difference between runs of the same model down to mean diff=0.011 at my scale(6 heads, 8 layers, embed_dim = 768, batch = 3, expand = 4)

## Part 3 — The comparison, and the hunt for why

I started the Pre/Post-LN test using a toggle I made. The results came out completely opposite of my prediction: **Post-LN beat Pre-LN by a gap of 0.279, p < 0.001.** It was the largest effect I have ever seen — 3× the RoPE effect, 24× the noise floor.

I thought it was a head start, so I ran it for 800 iterations, but it did not catch up (the slope was indistinguishable from 0). So the head-start hypothesis is down (*Post-LN's early-training lead is a warmup artifact; Pre-LN learns faster underneath and will catch up given time*).

This reasoning (*that if the model learns better than Post-LN, it should eventually cross the other model in loss*) also brings down the LayerNorm hypothesis — that the model had to learn 2 more weights. But if it had to learn 2 more weights, it should finish tuning those weights and catch up. **The reasoning brings down an entire class of hypotheses: that Post-LN had some sort of starting advantage.**

The third hypothesis is that the new LayerNorm in the forward pass is worsening Pre-LN. I tested it by removing it. After a few iterations it started to spit out NaNs and INF (initial loss 33.3 vs the normal ~9.3, then inf, then NaN by iteration 4) — see Part 1 for the reason. It turns out that the new LayerNorm is actually mandatory for the model, as shown by the tests. The new layernorm is also **not** worsening Pre-LN, by reasonning: If the new layernorm is bad for the model, then removing it would make the model better, not crash.

The final hypothesis is that 8 layers wasn't enough for the abilities of Pre-LN to really show. So I tried 12 layers. My prediction was that the gap will go down but Post-LN will still be better — and I was right: the gap went down to 0.12, but Post-LN is still better. This doesn't prove the hypothesis completely, but is consistent with it.

At this point I figured out that going more iterations or more layers isn't really worth it, so I stopped here.

## Part 4 — The lesson

Instead of just stopping or running even more tests, I made falsifiable hypotheses, and found arguments that use reasoning and tests to bring down individual hypotheses AND entire classes of hypotheses. I used predictions before every test, and admitted failure only after rigorously shooting down hypotheses — instead of trying to make the original prediction true.

There are 2 sorts of scenarios here, and here is how I respond to both.

**First scenario: I have absolutely no evidence or proof, only reasoning.** How I act: I refuse to be talked out of it, I hold onto my reasoning, and finally run the test — even though the test looked sure to fail. The test proved me right, with actual numbers. I refuse to accept anything (defeat, failure, victory) unless I have actual NUMBERS and RESULTS.

**Second scenario: I have a test sitting under my nose.** Instead of refusing to admit failure, I accept it. Then, instead of stopping, I go further — to *why* was my prediction wrong (or right). I make hypotheses, shoot them down, and only then stop, without testing even more to prove my prediction right, because I already have numbers in front of me.

---

# Project 3 — RMSNorm vs LayerNorm

RMSNorm is a variation of LayerNorm. LayerNorm first subtracts by the mean, then divides by std. RMS drops the subtraction by mean completely and divides by `rms(x)`, which is `sqrt(mean(x**2))`. I think the drop doesn't affect much because the mean-centering is mostly redundant — scale does most of the job, and the downstream weights (attention, FF...) are robust and can affect `x` as much as they want.

I gradient tested the implementation, and it passed tighter than I expected (24/24 at 4-5 significant digits).

Then I went to testing with **768 embed_dim, 8 layers, 6 heads, RoPE, Post-LN**. My prediction was that RMS would get us the exact same performance, but faster and a tiny bit less memory heavy.

After I tested it, my prediction was quantitatively right:

- **Mean diff: −0.0104** — just under the noise floor (0.011)
- **t = −0.666, p = 0.5060** — statistically indistinguishable
- **No slope**
- **Time per iteration: RMS is faster by 0.0459s — that's 7.2% faster**
- **Memory dropped by 80 MB**

---

# Project 4 — SwiGLU vs ReLU

## Part 1 — What SwiGLU is, and the prediction

SwiGLU uses a gating mechanism. It first calculates a value from `x @ W_value`, then uses a gate: `swish(x @ W_gate)`. The gate matrix is a bit like get-item — it can be 0 in some places, discarding features, or it can be positive, keeping or amplifying some features.

The good thing is that the gate is computed with `x`, so it is **data dependent**. ReLU has a fixed rule (negative trash, keep positive as is); SwiGLU *learns* what to keep and what to throw away.

Swish is `x * sigmoid(x)`. The derivative of `sigmoid(x)` is `exp(-x)/(1+exp(-x))^2`, which simplifies to `sigmoid(x) * (1 - sigmoid(x))`.

Last it does a `W_out`. So in the ReLU network there were 2 matrices and 2 vectors; now there are 3 (new `W_gate`), so memory will increase a bit (200 MB) as shown later in tests.

**My prediction:** SwiGLU will perform better than ReLU, with the gap bigger than RoPE vs ALiBi.

## Part 2 — Implementation

I started building the implementation when I realized I don't have a sigmoid method. I went to build one and used the `check_gradient` function — my prediction that they should be tight was correct, tight to 4 decimal places.

Then I found something weird: a manual gradient part in the old FeedForward. It was making a Tensor into a cp array, then applying max, then back into a Tensor, then it manually updated some attributes such as `_children` and `grad`. I suddenly understood it — back then, when there was ReverseGrad but no CuPy, I asked somebody else to make everything into CuPy, and because there was no max, he used a manual method. Now I still don't have max, but I can swap it into `clip(0, inf)`.

So I built it, and built the toggle. I gradient tested a transformer block and everything was tight (3-4 decimal places).

## Part 3 — The comparison, and the lying graph

I started the test with **6 heads, 8 layers, expand = 4, pre_ln = False, rms = True**. After 200 iterations the results were against my prediction. ReLU appears to start at 8 and SwiGLU started at 9, then ReLU appears to have a head start; SwiGLU quickly caught up after 20 iterations, but it still didn't close the gap. The memory increased by about 200 MB.

**Mean diff: −0.2804, t-stat: −17.416, p-value: 0.0000** — that is 25 times stronger than random initialization. (I first tested random initialization by running it 2 times, but nothing changed.)

I wondered why ReLU started at 8. The first hypothesis was that there were 3 matrices all at the 0.1 level, where in ReLU there were 2, so the output of SwiGLU was one smaller. I increased the most direct matrix, `W_out`, to `*1` instead of `*0.1` — but the graph is exactly the same.

But then I noticed that in the **raw** loss graph, both started at 9. The tutor was focusing on the smoothed graph, which averages over 10 iterations — **so the smoothed graph was lying.** Because ReLU descends faster, the average output a lower loss than the starting point.

Now the conclusion was that they start at the same point, but ReLU had some sort of advantage and learns faster. After 25 iterations SwiGLU nearly catches up, but it doesn't quite catch up.

One hypothesis is that SwiGLU needs more iterations to overcome ReLU — but **the entire class of "improvement over more iterations" hypotheses was ruled out** after running both for 500 iterations and reasonning: If SwiGLU was better but was a bit slow at the start (head start by ReLU), it would catch up over more iterations and the gap would decrease, but the gap stayed the same. SwiGLU 3.827 vs ReLU 3.630 at iterations 450–500.

## Part 4 — Finding and the pattern across all four projects

**Finding:** SwiGLU appears to be worse than ReLU at this scale. One hypothesis that is untested is that multiplying 2 random matrices gives roughly the product of the 2 variances. 500 iterations tested, but it holds much further since the slope is indistinguishable from 0. At 6 heads, 8 layers, expand = 4, 0.4GB of TinyStories — SwiGLU is worse at this scale. It can absolutely be possible that SwiGLU will overcome ReLU with a bigger model.

**Pattern:** The 2 working projects are almost scale independent — that means they work regardless of how big the training is. For the 2 failed projects, one of them is scale dependent: Pre-LN is tested on scale and it partially improved, so it could overcome Post-LN if we increase the scale even more; and SwiGLU is not tested.

All 4 projects worked. 2 actually improved the model, 1 can improve the model if we increase the scale, and 1 is not yet tested on scale. **(THEY ARE STILL SUCCESS because they produced a real, informative result.)**

**Deriving why something should help tells you nothing about whether it helps here.**
