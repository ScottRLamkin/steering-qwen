## Findings: does surviving quantization measure what a neuron means?

**The question.** Could how much of a neuron's steering survives quantization be used to judge how strong or clean its meaning is?

**Short answer: no.**
- **Steering survives:** almost all of each neuron's steering survives NF4, int8 and FP4.
- **The differences are real but don't track meaning:** neurons do differ in how much they keep, and those differences repeat across formats. They don't follow any bf16 measure of meaning: the judge's rating of the trait, coherence, consistency, selectivity, or whether the neuron breaks the model.
- **What they follow instead:**
  - mostly, which trait the neuron was found for;
  - after that, the neuron's bf16 effect size: bigger effects lose proportionally more;
  - weakly, its layer and push size.

**Verdict:** quantization tests the format and the trait's answer set, not the neuron. I won't add an "int3 fragility" column to the search.

The numbers are in [REPORT.md](REPORT.md), and the definitions are in `src/steering_llms/neurons/robustness_report.py`.

### Setup

**The panel:** 95 neurons, 5 per trait across 19 traits.
- The traits are HedwigAI's 9, taken from `runs/torch-bf16`, and our 10, from `runs/search/new-traits`.
- Per trait, the screen's top 3 and 2 weaker ones (at screen ranks about 8 and 20).
- Plus 20 random neurons, each with a residual push the same size as a real neuron's.

**The edit:** each neuron is pushed alone, with the same bf16 `down_proj`-input value in every format.

**The formats:**
- real: NF4, int8 and FP4, through bitsandbytes;
- fake: int4 and int3, which round the bf16 weights to the nearest level in groups of 64.

**The main measure is held-out specificity.** It is the log-prob shift on the trait's answer set minus the mean shift on the other traits' sets.
- It covers the first 32 tokens of answers to 10 new questions.
- The answers were written once in bf16, so every format scores the same text.
- Retention R = specificity in the format / specificity in bf16, for the 75 neurons whose bf16 interval excludes 0.

**Replies:** generated in each format and rated by the bf16 model as judge, for trait and for coherence, so the judge is the same in every format.

### 1. Steering survives every real format

| format | median R | 10–90% of R | spread beyond noise | random push: coherence of replies | top neurons: judged trait gain (r with bf16) |
|---|---|---|---|---|---|
| bf16 | 1.00 | – | – | 3.34 | +1.55 |
| int8 | 1.00 | 0.95–1.05 | 0.04 | 3.26 | +1.66 (0.97) |
| NF4 | 0.97 | 0.85–1.14 | 0.11 | 3.15 | +1.61 (0.93) |
| FP4 | 0.97 | 0.72–1.15 | 0.25 | 3.07 | +1.66 (0.94) |
| fake int4 | 1.00 | 0.76–1.13 | 0.15 | – | – |
| fake int3 | 0.74 | 0.19–1.28 | 0.50 | 2.15 | +1.60 (0.85) |

- **Spread beyond noise** is the sd of the per-neuron change in specificity, after removing each neuron's bootstrap noise.
- **Coherence of the unsteered replies** is 3.85 in bf16 and 3.78 in fake int3.
- **Trait gain** is for each trait's top neuron, judged by the bf16 model against the unsteered replies of the same format.

What the table shows:
- **Neurons found in bf16 work in NF4, int8 and FP4.**
  - Their effect on answer log-probs and the trait the judge reads in their replies are both kept.
  - That holds for the neurons the blog shows: L17N7426+, L5N4530−, L17N8623+ and L20N7450− keep 0.89–1.03 of their effect in each real format.
  - It holds for ours too.
- **A search in those formats would find most of them again.** Of the 54 significant top-3 neurons, this many are still in the top 15 of that format's attribution for their trait:

  | format | bf16 | int8 | NF4 | FP4 | fake int3 |
  |---|---|---|---|---|---|
  | in the top 15 | 33 | 32 | 27 | 27 | 21 |

- **Fake int4 behaves like real 4-bit.**
  - It keeps as much as they do (median R 1.00).
  - Its spread, 0.15, sits between NF4's 0.11 and FP4's 0.25.
  - Its ranking of the neurons correlates at ρ 0.33 with NF4's and 0.51 with FP4's.
  - So fake int3 is a fair, harsher version of the same rounding.
- **Fake int3 makes the model fragile to any push, not to particular neurons.**
  - Its unsteered replies are about as coherent as bf16's.
  - A random push the size of a real neuron's drops coherence from 3.34 to 2.15.
  - The top neurons' trait gains mostly survive (+1.60), but so does the judge's reading of trait in random pushes: +0.64 ± 0.54, against +0.16 in bf16. Garbled replies get read as trait-like.

### 2. Neurons differ in how much they keep, and the differences repeat

**Different formats rank the neurons alike.** The mean Spearman between formats is 0.33, from 0.16 for int8–FP4 to 0.51 for FP4–fake int4.

**Averaging the five rankings gives a stable "robustness" score.**
- Its reliability is about 0.71 (Spearman–Brown).
- So there is a real neuron-level property to explain.

### 3. What it tracks, and what it doesn't

Spearman ρ with the averaged robustness, over 75 neurons; partial ρ controls for the bf16 effect size (REPORT §3b):

| bf16 measure | ρ | partial ρ |
|---|---|---|
| judge's trait gain | −0.05 | +0.01 |
| coherence of its replies | −0.05 | −0.04 |
| broken (1) or fluent (0) | +0.02 | +0.04 |
| consistency across questions | −0.16 | +0.06 |
| selectivity | −0.28 | −0.14 |
| z against random neurons | −0.31 | −0.03 |
| found again by the MLX 8-bit and torch NF4 searches (HedwigAI's traits, n=38) | −0.34 | −0.27 |
| **bf16 effect size** | **−0.31** | – |
| layer | +0.25 | +0.20 |
| push size, \|down_in value\| | +0.27 | +0.21 |
| column rounding error | +0.11 | +0.10 |

**Meaning doesn't predict it.** None of the measures of meaning predicts robustness, alone or once effect size is removed.

**Broken neurons survive as well as fluent ones.**
- These are the neurons whose bf16 replies are gibberish or loops.
- Their median R is 0.99 against 0.95 in NF4, and 0.75 against 0.74 in fake int3 (REPORT §4b).
- L23N81+, a generic persona push that loops "k.k. k.k.", is among the most robust neurons in the panel.

**Effect size works the wrong way for the hypothesis.**
- Neurons with large bf16 effects keep proportionally less, not more.
- Several of the largest are already past their peak at dose 1:
  - cheerful L5N6794−: specificity +3.39 at 0.5×, +2.12 at 1×;
  - robotic L5N6959+;
  - angry L5N6449−.
- Their effect sits on a falling slope that any perturbation moves.

**Layer and push size matter a little.** Early-layer neurons with large columns lose more: layers 4–8 average robustness 0.46, layers 17 and up 0.56. Column rounding error barely matters: the column itself survives.

### 4. The trait matters more than the neuron

**Grouping the neurons by trait explains 57% of the variance in robustness.** Chance would give about 23% with 18 traits; permutation p = 0.0005. The pattern holds in every format except NF4: η² of R by trait is 0.46–0.60 in int8, FP4, fake int4 and fake int3. It holds even in int8, where a neuron changes by only a few percent.

- **Fragile:** every humorous neuron (mean 0.20), cheerful (0.23), sycophantic (0.25), terse (0.34).
- **Robust:** melancholic (0.73), paranoid (0.70), anxious (0.65).

These neurons sit in different layers and pull in different directions, so the shared factor is most likely each trait's answer set, not the neurons.

- **How it would work:** quantization moves the unsteered model's likelihood of each trait's answers relative to the others, and that changes how much room a push has.
- **Example:** terse answers become relatively more likely under quantization (+0.34 log-prob contrast, averaged over the five formats), the largest change of any trait.
- **Evidence across traits:** the change in contrast correlates at −0.42 with the trait's mean robustness, over 18 traits. That's suggestive, not conclusive.

### What this means for the search

- **Searching in a quantized format is fine.**
  - NF4, int8 and FP4 keep almost all of every neuron's effect.
  - Fake int4 confirms that 4-bit rounding moves the neurons only modestly.
  - Edits found in bf16 can be shipped to a 4-bit model.
- **To judge a neuron's meaning, use the rate stage and the dose sweep, not quantization.**
  - The rate stage's coherence and trait ratings are what separated fluent neurons from broken ones in `runs/search/NOTES.md`.
  - A neuron already past its peak at dose 1 is a warning sign.
- **Trait-level differences in retention say more about the measurement than about the neurons.**
  - Comparing neurons across traits by retention would mostly compare the traits.

### Caveats

- **Few questions:** 10 held-out questions, and only the first 32 answer tokens. A neuron's interval is often ±0.2.
- **Noisy ratios:** retention ratios are noisy for weak neurons, so the analysis keeps only the 75 significant ones.
- **One dose:** the dose sweep covers only each trait's top neuron, and only in bf16 and fake int3.
- **A small judge:** the judge is Qwen3.5-4B itself, rating 1–5. It reads some garbled replies as trait-like, which is visible in fake int3's random baseline.
- **Fake quantization covers only the decoder layers:** it rounds every `nn.Linear` in them, not the embeddings or the LM head. bitsandbytes formats also leave those in 16-bit.
- **One model:** everything here is Qwen3.5-4B on MPS.

### Files

**Per format** (`<format>/`):
- `score_<study>.json`: per-question log-probs, unsteered and for every push;
- `replies.json`;
- `judge.json`, `coherence.json` and `fluency.json`: ratings by the bf16 judge;
- `attrib.json`;
- `cols.pt` (not committed).

**Shared:**
- `panel.json`: the neurons and their values;
- `_shared/heldout_<study>.json`: the held-out answer sets;
- `summary.json`: one record per neuron with every measure, including `robustness`.

**Logs:** `run.sh` runs the whole queue; `run.log` and `<step>.log` are its logs.
