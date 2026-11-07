## Findings

The recipes that steered in bf16 (`runs/methods`), rerun on Qwen3.5-4B quantized three ways with bitsandbytes on MPS, 6 November 2026:

- **NF4** and **FP4:** 4-bit weights; activations stay bf16.
- **int8** (LLM.int8): 8-bit weights, and each layer's input activations rounded to 8 bits per token, except outliers above 6, which stay in 16-bit.

Each recipe ran twice:

- **Transfer:** the bf16 run's own neurons and edits, on each format. This answers "is the quantized model as responsive?"
- **Found again (NF4 only):** discovery rerun on the NF4 model, then the same evaluation.

Every run uses the same settings, prompts and seed as its bf16 run, and measures its own no-edit baseline and random-neuron control. The samples are as small as in bf16, so differences under about 0.1 are noise. PCN, sNeuron-TST, Gender, Emotion and IRiS didn't steer in bf16 and weren't rerun.

**Short answer:** most neuron edits work about as well on the quantized models as in bf16. The personality methods, Contrastive Pair Search and Knowledge Neurons respond within noise of bf16 in all three formats, and NF4 finds largely the same neurons again. The exceptions:

- AurA weakens with 4-bit weights.
- Steering through one sentiment unit is unreliable in every format.
- Safety Neurons doesn't reproduce on any quantized model, though its bf16 effect is itself only about half the first estimate.

**Responds as in bf16:**

- **NPTI:** log-prob shift up/down averaged over the five traits:

  | | bf16 | NF4 | int8 | FP4 |
  |---|---|---|---|---|
  | up / down | +1.00 / −1.40 | +1.00 / −1.37 | +0.99 / −1.39 | +1.00 / −1.44 |

  The judge moves the same way in every format, and the perplexity cost is the same (replies 12–14 against about 2.5).
- **DPN-LE:**

  | | bf16 | NF4 | int8 | FP4 |
  |---|---|---|---|---|
  | up / down | +0.44 / −0.74 | +0.42 / −0.71 | +0.45 / −0.73 | +0.48 / −0.75 |

- **HedwigAI's single neurons:** the mean shift along each neuron's named trait (evil, arrogant, humorous, cheerful) is +1.21 in bf16, +1.19 on NF4, +1.20 on int8 and +1.04 on FP4. The random-neuron spread is ±0.31 to ±0.40.
- **Contrastive Pair Search:** ablating the 269 neurons cuts refusal of harmful requests from 0.94 to 0.26 in bf16, 0.90 to 0.28 on NF4, 0.94 to 0.22 on int8, and 0.92 to 0.10 on FP4. Random neurons never move it. Doubling them raises refusal of benign requests in every format.
- **Knowledge Neurons:** zeroing a fact's neurons cuts the answer probability by 84% in bf16, 90% on NF4, 82% on int8 and 74% on FP4. Doubling raises it by a median of 306%, 142%, 290% and 216%. The random control is noisier on the 4-bit models (−13% on NF4, +11% on FP4, against +1% in bf16).
- **AtP:** the neurons that matter are the same in every format. On pools of 600 neurons over 4 country pairs, on average 8.5 to 9 of each pair's 10 largest exact effects are among bf16's 10. On NF4 and FP4, AtP agrees with exact patching as well as in bf16: recall@100 of 0.74 and 0.86, against 0.82, with Pearson 0.98 in all three.

  int8 is the exception: recall@100 0.36 and sign agreement 0.74. Patching any single neuron on int8 moves the logit difference by about 0.2: the median exact effect is 0.19, against 0.035 in bf16. That fits its 8-bit activation rounding, where changing one neuron can change how its whole row is rounded. That buries the small true effects, while the large ones still come through (3.59 against 3.86 for the top neuron).

**Weaker or inconsistent:**

- **AurA:** toxicity falls from 0.39 to 0.29 in bf16 and from 0.43 to 0.27 on int8. On the 4-bit models it barely moves: 0.39 → 0.37 on NF4, and 0.34 → 0.31 on FP4 (random 0.32). Found again on NF4, the experts are almost the same units (Jaccard 0.96), and they still give only 0.34. With 64 prompts the standard error is about 0.04, so this is suggestive rather than certain. A guess at the cause: AurA damps most of its 560k units only slightly, and a change that small and spread out may sit inside 4-bit weight error.
- **Sentiment unit:** reading the unit holds up. NF4 finds the same best unit, L18N1324, with the same accuracy (0.77), and the L1 probe reaches 0.90. Steering through it doesn't hold up:
  - Clamping the unit high at 4× moves reviews from 0.15 to 0.34 positive in bf16, but from 0.29 down to 0.16 on NF4, and from 0.28 to 0.45 on int8.
  - The bf16 top-16 set beats its random control on NF4 and int8 too (0.47 against 0.19, and 0.56 against 0.33).
  - On NF4, the top 16 found again share about 7 units with bf16's (Jaccard 0.28) and don't steer (0.19 against a 0.29 baseline).
  - FP4's raw review continuations are already 87% positive, which leaves no room upward. Pushing the 16 units down at 4× does take it to 0.23.
- **Safety Neurons:** in bf16, patching the instruct model's values into the Base model closed 94% of a 0.57 harm gap on 20 prompts, against 21% for random neurons. No quantized format reproduces it:
  - int8 has an even larger gap, 0.93, but closes only 11% of it (random 24%).
  - NF4 and FP4 make the patched replies no less harmful: 2.94 → 3.21 on NF4 (random 3.28) and 2.38 → 2.65 on FP4 (random 2.48).
  - On NF4, the judge (the NF4 instruct model) barely tells Base from Instruct, with a gap of 0.19.

  The neurons themselves are stable: found again on NF4, they overlap bf16's with Jaccard 0.77 (chance 0.026).

  The bf16 result also shrank when rerun on 60 prompts (`bf16-n60/safety`): the patch closes 54% of a 0.27 gap, while random neurons close none (−2%). So the effect is real in bf16 but smaller than 20 prompts suggested. The 20-prompt quantized runs can't rule out an effect of that size.

**Found again on NF4:** five recipes find largely the same neurons, against chance overlaps of 0.03 or less. The new sets steer about as well as bf16's:

| recipe | Jaccard with bf16 |
|---|---|
| NPTI | 0.69 |
| DPN-LE | 0.70 |
| Contrastive Pair Search | 0.70 |
| Knowledge Neurons | 0.71 (8 of bf16's 9 facts still known) |
| Safety Neurons | 0.77 |

### What it costs on this laptop (M4 Pro, 24 GB, MPS)

| format | memory | forward pass | generation | 8 transfer runs | 7 rediscovery runs |
|---|---|---|---|---|---|
| bf16 | 7.8 GiB | 1× | 1× | about 70 min | — |
| NF4 | 2.85 GiB | 1.5× | 1.35× | 93 min | 162 min |
| FP4 | 2.85 GiB | 1.5× | 1.35× | 93 min | — |
| int8 | 4.5 GiB | 3× | 3.6× | 206 min | — |

bitsandbytes has no fused kernels on MPS. On CUDA, the quantized formats would be much closer to bf16 speed.

### Caveats

- **Judges are the same format:** the personality judge and the Safety harm rating are the model under test, so a quantized format is also judged by a quantized judge. The log-prob shifts, refusal rates, answer probabilities and classifier scores don't depend on that.
- **FP4 is the most degraded model:** neutral-text perplexity is 147, against 130–133, and its raw review continuations are 87% positive, against 15–29%.
- **One seed, small samples:** as in bf16.
