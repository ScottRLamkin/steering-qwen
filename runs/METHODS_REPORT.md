# Neuron methods on Qwen3.5-4B

Produced by `uv run neurons report`. Each section comes from `runs/methods/<method>/eval.json`. Settings are in that directory's `config.json`, and the neuron sets are in `sets.json`.

## Findings

One run of each recipe on Qwen3.5-4B, bf16, on an M4 Pro (MPS), 5–6 November 2026; Safety Neurons also uses Qwen3.5-4B-Base. Seed 0, greedy decoding. Most evaluations use 6 to 64 prompts per condition, so differences under about 0.1 are noise. Every steering claim below is relative to the same edit on random neurons, matched per layer.

**Steers, and beats its random control:**

- **NPTI** (17–23k neurons per pole) has the strongest personality effect. The judge moves the right way on all 5 traits: Extraversion 2.23 → 3.42 up and 1.30 down, while random stays between 2.18 and 3.07. The judge-free log-prob shift is +0.54 to +1.33 nats/token up and −1.03 to −1.94 down, against random within ±0.47. The cost is fluency: reply perplexity goes from about 2.4 to 9–20.
- **DPN-LE** (46 neurons in each of layers 16–31, 736 per trait) has about half NPTI's effect with 1/40 of the neurons. The log-prob shift is +0.31 to +0.59 up and −0.41 to −0.98 down, against random within ±0.16. Perplexity goes from 2.4 to 3.4–8.2. On the judge, the down push lowers the rating for 4 of 5 traits (not Extraversion), and the up push beats random for 3 (not Openness or Agreeableness).
- **Contrastive Pair Search** (269 neurons, mostly in layers 27–31): ablating them (m=0) drops refusal of harmful requests from 0.94 to 0.26. Doubling them (m=2) raises refusal of benign requests from 0.16 to 0.38. The random control doesn't move either rate. This reproduces the paper.
- **Safety Neurons** (14,745 neurons, the top 5%, 71% of them in layers 27–31): the paper contrasts an SFT model with its DPO version; here the pair is Qwen3.5-4B-Base and Qwen3.5-4B.
  - On 20 held-out HH-RLHF prompts, the Base model's replies get a harm rating of 2.49 (1–5).
  - Copying the instruct model's values of those neurons in at every step brings that to 1.96. The instruct model itself gets 1.92, and copying random neurons gives 2.37.
  - That closes 94% of the gap, against 21% for random neurons. The paper reports over 90% with 5% of neurons.
  - The refusal-phrase rate barely separates them (base 0.40, patched 0.50, random 0.40, instruct 0.35): the instruct model mostly answers these prompts safely instead of refusing.
  - With 20 prompts, and the instruct model as the judge, this shows direction more than size.
- **Knowledge Neurons** (2–14 per fact, mostly layer 31): zeroing them cuts the answer probability by 84% on average, against +1% for random. Doubling raises it by a median of 306%. The paper reported −29% and +31% on BERT. But the neurons aren't specific to their fact: zeroing them also cuts other relations' facts by 38%, and L31N837 is among the neurons of 6 of the 9 facts. They look like generic answer-output neurons.
- **AtP**: over all 294,912 neurons for France→Italy, AtP's top 10 is exactly the true top 10 from exact patching, ranked almost the same. 29 of the true top 30 and 76 of the true top 100 are also in AtP's top 30 and top 100, and AtP gets the sign right for 95 of those 100. The 100th-largest true effect is 0.11 on the logit difference. The first sweep read the bf16 logits, which move in steps of 1/16, the size of most single-neuron effects; that run gave recall@100 = 0.44. Reading the final hidden state in float32 gave 0.76. Over all neurons the Pearson correlation is only 0.50, because the hundreds of thousands of near-zero effects are mostly noise to AtP. That doesn't matter for picking candidates. HedwigAI's attribution uses the same first-order estimate.
- **AurA**: on 64 of RealToxicityPrompts' "challenging" prompts, damping the experts at all three sites lowers toxicity from 0.39 to 0.29, while random stays at 0.39. Any one site alone does nothing. Perplexity on neutral text rises from 130 to 215, but random rises as much (222), so that cost comes from damping 560k units, not from the experts. The expert count is inflated: ParaDetox's toxic sentences are longer, so max-pooling puts most units just above AUROC 0.5. Those units get a multiplier near 1 and barely change.
- **Sentiment unit**: the best single neuron, L18N1324, classifies SST-2 at 0.77. An L1 probe over 197 neurons reaches 0.905; Radford et al.'s single mLSTM unit reached 0.92. Clamping it at 4× its 99th percentile turns raw review continuations more positive, from 0.15 to 0.34; the opposite clamp gives 0.06. Sixteen units at 4× give 1.00. The 16 prompts share one opener, so their outcomes are correlated, and 16 random units clamped at 4× also swing the score (0.38, 0.88). Only the 1× clamps have clean controls (top-16: 0.44 against random 0.19).
- **HedwigAI's single neurons** map onto the Big Five as their names suggest:
  - evil: Agreeableness −1.41, Openness −1.23;
  - arrogant: Openness −1.44, Agreeableness −1.10;
  - humorous: Extraversion +1.10;
  - cheerful: Extraversion +0.88.

  The random single-neuron pushes average −0.38 on Openness, so read Openness shifts against that. NPTI's sets contain 19 of HedwigAI's 24 screened neurons, including L17N7426 (poet), where chance would give about 8.

**Found neurons, but editing them doesn't steer:**

- **Psychological Concept Neurons**: the neurons do separate each Big Five domain, with average precision 0.47–0.69 against chance 0.20. Setting them to their 1st or 99th percentile has no effect: log-prob shifts are within ±0.05, the same as random. That holds with 50 neurons on the last prompt token and with 500 at every position.
- **sNeuron-TST**: the chat model already detoxifies when asked to rewrite (P(neutral) 0.97 with no edit), so there was nothing to improve. Asked to copy the toxic sentence word for word, it copies all 40 exactly with the ~5,900 toxic-style neurons zeroed, just as it does with no edit or with random neurons zeroed. DoLa raises content overlap (ROUGE-1 0.18 → 0.30) and lowers perplexity on the plain model.
- **Gender**: none of the three pushes steers toward its gender. `to_female` lowers the female word share (0.28 → 0.11; random 0.32), and `to_male` doesn't beat its random control. Only the neutral zero-only edit beats random: 0.19 against 0.01 (base 0.10). Zeroing about 6k neurons of any kind pushes the stories toward male defaults. The labels were checked by hand and are correct.
- **Emotion & Rhetoric** (106–913 neurons per label): ablating a label's neurons barely changes zero-shot classification of that label. The changes are surprise −0.15, anger −0.05, love +0.20, and 0 for the rest, with 20 texts per label.
- **IRiS**: topic-weighted NPTI does no better than plain NPTI on the same questions. For Extraversion the judge gives 3.64 against 3.86 up, and 1.51 against 1.50 down. The four topics' neuron sets overlap heavily (mean Jaccard 0.60). And the topic routing barely routes: a question's weight on its own topic averages 0.27, where an even split over four topics would give 0.25. The model's middle-layer embedding hardly tells these topics apart.

### Caveats

- **Judge:** the unsteered 4B model, reading its expected 1–5 rating from digit probabilities. It shows direction, not size.
- **Persona log-prob score:** the reference answers come from PersonalityBench personas in NPTI's own prompt format, on held-out questions. That is close to how NPTI finds its neurons, which may favour it.
- **Questionnaire:** the self-rated questionnaire barely moves for any method. The edits are applied only at the answer position, and an assistant rating "itself" seems insensitive to them. The log-prob score and the judge are the useful measures.
- **One seed, small samples:** treat differences under about 0.1 as noise.

### What fits on this laptop (M4 Pro, 24 GB, bf16 on MPS)

| recipe | plan rating | time on the real model | verdict |
|---|---|---|---|
| HedwigAI | R | ~1 h for the search (`find_neurons.py`) + 10 min here | works |
| DPN-LE | R | 21 min for 5 traits, + 3 min log-prob score | works, best effect per neuron |
| Psychological Concept Neurons | R | 13 min; 11 min with k=500 at every position | runs; no steering effect |
| NPTI | R (subsampled) | 26 min for 5 traits | works, biggest effect, worst fluency |
| IRiS | S | 13 min (1 trait, 4 topics) | runs; no gain over NPTI |
| sNeuron-TST | R | 3 min; DoLa adds about 2 min | runs; no headroom, copy probe null |
| AurA | R | 5 min discovery, 4 min evaluation | works on RealToxicityPrompts |
| Gender | R | 3 min + 4 min | runs; no steering |
| Contrastive Pair Search | R | 5 min | works, reproduces the paper |
| Knowledge Neurons | R (tens of facts) | 20 min for 9 facts | strong but not specific |
| AtP | R | 2 s for AtP; about 2 h for exact patching of every neuron on one pair | top 10 exact; 76 of the top 100 |
| Sentiment unit | R / S | 4 min + 3 min | probe works; steering partly |
| Emotion & Rhetoric | S | 7 min | runs; ablation does nothing |
| Safety Neurons | S | 19 min to find, 9 min to evaluate; both models in bf16 (15.7 GiB) plus 9 GB of disk for Base | works, reproduces the paper |

Everything fits, including Safety Neurons once there was disk for the Base model; both 4B models fit in memory together in bf16. The heavy parts are exact patching (about 37 neurons a second) and generation, slowest in Safety Neurons' lockstep decoding, one token at a time. Paper-scale datasets and LLM judges remain out of reach, as the plan expected.

| recipe | status | neurons found | run |
|---|---|---|---|
| `hedwig` | ran | 27 | _torch.bfloat16, seed 0_ |
| `dpn-le` | ran | 3680 | _torch.bfloat16, seed 0_ |
| `pcn` | ran | 250 | _torch.bfloat16, seed 0_ |
| `npti` | ran | 178848 | _torch.bfloat16, seed 0_ |
| `iris` | ran | 179916 | _torch.bfloat16, seed 0_ |
| `sneuron-tst` | ran | 11796 | _torch.bfloat16, seed 0_ |
| `aura` | ran | 561342 | _torch.bfloat16, seed 0_ |
| `gender` | ran | 8847 | _torch.bfloat16, seed 0_ |
| `cps` | ran | 269 | _torch.bfloat16, seed 0_ |
| `knowledge` | ran | 70 | _torch.bfloat16, seed 0_ |
| `atp` | ran | 1200 | _torch.bfloat16, seed 0_ |
| `sentiment` | ran | 16 | _torch.bfloat16, seed 0_ |
| `emotion` | ran | 2064 | _torch.bfloat16, seed 0_ |
| `safety` | ran | 14745 | _torch.bfloat16, seed 0_ |

## Personality: DPN-LE, Psychological Concept Neurons, NPTI

Every method is measured the same way on each Big Five trait:

- **Questionnaire:** the model's self-rating, 1 to 5, on the held-out IPIP-50 items, read from its answer-digit probabilities. Each method's edit is applied at the answer position.
- **Judge:** the unsteered model's 1-5 rating of the trait in replies to PersonalityBench test questions.
- **Perplexity:** of those replies, under the unsteered model.

**+ and -** are the edits pushing the trait up or down. **rnd** is the same edit on random neurons, matched per layer. Shifts are relative to no edit.

| method | trait | neurons | quest. base | + | - | rnd+ | rnd- | judge + | judge - | judge rnd+ | ppl base | ppl + | ppl - |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dpn-le | Openness | 736 | 4.43 | +0.02 | -0.11 | +0.08 | -0.20 | +0.45 | -0.75 | +0.58 | 2.2 | 3.4 | 4.5 |
| dpn-le | Conscientiousness | 736 | 4.00 | -0.05 | -0.04 | -0.35 | +0.17 | +0.47 | -1.01 | -0.11 | 2.6 | 4.5 | 3.8 |
| dpn-le | Extraversion | 736 | 3.28 | +0.05 | -0.08 | -0.08 | +0.07 | +1.25 | +0.14 | +0.36 | 2.5 | 8.2 | 5.0 |
| dpn-le | Agreeableness | 736 | 4.18 | +0.08 | -0.21 | -0.03 | -0.05 | +0.00 | -0.78 | +0.46 | 2.5 | 4.7 | 5.0 |
| dpn-le | Neuroticism | 736 | 3.17 | +0.05 | -0.21 | -0.36 | +0.13 | +0.61 | -2.02 | -0.38 | 2.4 | 4.8 | 4.0 |
| pcn | Openness | 50 | 4.43 | -0.03 | -0.02 | -0.02 | -0.00 | -0.07 | -0.01 | -0.05 | 2.2 | 2.2 | 2.2 |
| pcn | Conscientiousness | 50 | 4.00 | -0.01 | -0.01 | -0.01 | +0.00 | +0.31 | +0.07 | +0.33 | 2.6 | 2.6 | 2.6 |
| pcn | Extraversion | 50 | 3.28 | -0.01 | -0.01 | -0.00 | -0.01 | -0.03 | -0.03 | +0.03 | 2.5 | 2.5 | 2.4 |
| pcn | Agreeableness | 50 | 4.18 | +0.02 | +0.02 | -0.00 | +0.01 | -0.12 | -0.15 | -0.17 | 2.5 | 2.7 | 2.5 |
| pcn | Neuroticism | 50 | 3.17 | -0.05 | +0.00 | +0.00 | +0.00 | -0.17 | -0.11 | -0.14 | 2.4 | 2.5 | 2.4 |
| pcn-k500-all | Openness | 500 | 4.43 | -0.01 | -0.01 | -0.02 | -0.00 | +0.03 | +0.07 | -0.11 | 2.2 | 2.2 | 2.4 |
| pcn-k500-all | Conscientiousness | 500 | 4.00 | -0.02 | -0.01 | -0.01 | -0.00 | -0.01 | +0.39 | +0.38 | 2.6 | 2.6 | 2.7 |
| pcn-k500-all | Extraversion | 500 | 3.28 | +0.02 | +0.01 | -0.03 | -0.04 | +0.46 | +0.40 | +0.57 | 2.5 | 2.4 | 2.5 |
| pcn-k500-all | Agreeableness | 500 | 4.18 | +0.04 | +0.02 | -0.00 | +0.00 | +0.05 | -0.22 | -0.18 | 2.5 | 2.7 | 2.6 |
| pcn-k500-all | Neuroticism | 500 | 3.17 | +0.06 | -0.05 | -0.00 | -0.01 | -0.06 | -0.11 | -0.33 | 2.4 | 2.5 | 2.6 |
| npti | Openness | 37256 | 4.43 | -0.11 | -0.33 | -0.02 | -0.11 | +0.18 | -1.88 | -0.26 | 2.2 | 12.7 | 9.8 |
| npti | Conscientiousness | 44508 | 4.00 | +0.16 | -0.21 | +0.19 | +0.00 | +1.40 | -0.44 | -0.25 | 2.6 | 12.3 | 20.5 |
| npti | Extraversion | 33982 | 3.28 | +0.00 | -0.20 | +0.14 | -0.09 | +1.19 | -0.92 | -0.05 | 2.5 | 11.2 | 12.3 |
| npti | Agreeableness | 31627 | 4.18 | +0.16 | -0.35 | +0.20 | -0.02 | +0.87 | -1.41 | -0.50 | 2.5 | 9.8 | 17.3 |
| npti | Neuroticism | 31475 | 3.17 | +0.21 | -0.42 | +0.59 | +0.00 | +0.51 | -2.40 | +0.00 | 2.4 | 12.2 | 9.3 |

### Judge-free: persona-answer log-prob shift

Held-out PersonalityBench test questions are answered under a high-pole and a low-pole persona. The answers are then re-read with no persona. The shift is the edit's change in mean log-prob(high answers) − log-prob(low answers), in nats per token. Pushing up should be positive, pushing down negative, and random near 0. Each edit keeps its own position scope.

| method | trait | contrast, no edit | shift + | shift - | shift rnd+ | shift rnd- |
|---|---|---|---|---|---|---|
| dpn-le | Openness | +0.407 | +0.335 | -0.748 | -0.127 | -0.073 |
| dpn-le | Conscientiousness | +0.338 | +0.588 | -0.414 | -0.103 | +0.161 |
| dpn-le | Extraversion | -0.248 | +0.305 | -0.984 | -0.030 | -0.049 |
| dpn-le | Agreeableness | +0.415 | +0.545 | -0.796 | -0.146 | +0.040 |
| dpn-le | Neuroticism | +0.232 | +0.448 | -0.756 | +0.026 | -0.020 |
| pcn | Openness | +0.407 | -0.007 | -0.007 | -0.007 | -0.005 |
| pcn | Conscientiousness | +0.338 | +0.001 | +0.000 | -0.001 | -0.001 |
| pcn | Extraversion | -0.248 | -0.001 | +0.002 | +0.003 | -0.000 |
| pcn | Agreeableness | +0.415 | +0.001 | -0.006 | -0.001 | -0.002 |
| pcn | Neuroticism | +0.232 | +0.002 | +0.001 | +0.003 | -0.000 |
| pcn-k500-all | Openness | +0.407 | +0.003 | +0.007 | -0.015 | +0.007 |
| pcn-k500-all | Conscientiousness | +0.338 | +0.002 | -0.007 | -0.007 | -0.015 |
| pcn-k500-all | Extraversion | -0.248 | -0.031 | -0.040 | +0.048 | +0.035 |
| pcn-k500-all | Agreeableness | +0.415 | -0.027 | -0.008 | -0.009 | +0.004 |
| pcn-k500-all | Neuroticism | +0.232 | +0.050 | +0.053 | +0.007 | +0.001 |
| npti | Openness | +0.407 | +0.537 | -1.941 | -0.470 | -0.165 |
| npti | Conscientiousness | +0.338 | +1.332 | -1.229 | -0.128 | +0.060 |
| npti | Extraversion | -0.248 | +1.083 | -1.207 | +0.121 | +0.011 |
| npti | Agreeableness | +0.415 | +1.003 | -1.576 | -0.048 | -0.231 |
| npti | Neuroticism | +0.232 | +1.040 | -1.026 | -0.159 | +0.081 |

### Neuron overlap between methods

Jaccard index of the (layer, neuron) sets for each trait's upward push. The methods read different points of the MLP (`down_in`, `gate`, `act`), but neuron *n* is the same unit at all of them.

| trait | dpn-le / pcn | dpn-le / npti | pcn / npti |
|---|---|---|---|
| Agreeableness | 0.003 | 0.010 | 0.000 |
| Conscientiousness | 0.000 | 0.003 | 0.000 |
| Extraversion | 0.003 | 0.015 | 0.001 |
| Neuroticism | 0.000 | 0.013 | 0.000 |
| Openness | 0.000 | 0.007 | 0.000 |

HedwigAI's 24 screened neurons (its 9 traits, not the Big Five) inside each method's Big Five sets. **By chance** is the count expected if the method's neurons were drawn at random from all 294,912.

| method | neurons in all its sets | HedwigAI neurons included | by chance | which |
|---|---|---|---|---|
| dpn-le | 2277 | 2 / 24 | 0.2 | L17N8623, L25N596 |
| pcn | 250 | 0 / 24 | 0.0 | – |
| npti | 96599 | 19 / 24 | 7.9 | L4N8405, L5N4530, L5N6573, L5N7055, L5N7779, L5N8940, L6N316, L9N8732, L11N2930, L15N8137, L15N8494, L15N9125, L16N5515, L16N8010, L17N7426, L17N8623, L20N7450, L23N4740, L25N596 |

## HedwigAI's single neurons on the Big Five questionnaire

Adding `dose * median_norm / ||col||` to a neuron's `down_in` matches `Steerer`'s residual push. Max |logit difference| 0.1562, against a steering effect of 8.90.

| trait | top neuron (dose 1) | Openness | Conscientiousness | Extraversion | Agreeableness | Neuroticism |
|---|---|---|---|---|---|---|
| (no edit) |  | 4.43 | 4.00 | 3.28 | 4.18 | 3.17 |
| evil | L17N8623+ | -0.16 | -0.12 | -0.11 | -0.01 | -0.30 |
| sycophantic | L20N7450- | +0.08 | -0.21 | +0.00 | -0.01 | +0.46 |
| humorous | L4N8405+ | +0.13 | +0.07 | +0.01 | +0.04 | +0.11 |
| melancholic | L5N4530- | +0.14 | +0.07 | +0.02 | +0.11 | +0.04 |
| paranoid | L16N5515+ | -0.27 | -0.35 | -0.12 | -0.23 | +0.11 |
| arrogant | L23N4740+ | -0.12 | -0.11 | -0.01 | -0.06 | -0.01 |
| anxious | L21N433+ | -0.21 | -0.14 | -0.04 | -0.17 | +0.07 |
| cheerful | L5N6794- | +0.05 | -0.04 | +0.00 | +0.02 | +0.06 |
| poetic | L15N9125+ | +0.19 | -0.08 | +0.02 | +0.19 | +0.32 |

The same neurons on the persona-answer log-prob shift. Positive means the push makes high-pole answers more likely than low-pole ones. The last row is each neuron's random control, averaged over traits.

| trait | neuron | Openness | Conscientiousness | Extraversion | Agreeableness | Neuroticism |
|---|---|---|---|---|---|---|
| evil | L17N8623+ | -1.232 | -0.523 | -0.490 | -1.410 | +0.409 |
| sycophantic | L20N7450- | -0.584 | +0.029 | +0.400 | -0.186 | +0.457 |
| humorous | L4N8405+ | -0.998 | -0.963 | +1.100 | -0.340 | -0.105 |
| melancholic | L5N4530- | +0.015 | -0.248 | -0.455 | -0.262 | +0.021 |
| paranoid | L16N5515+ | -0.372 | +0.453 | -0.769 | -0.546 | +0.109 |
| arrogant | L23N4740+ | -1.439 | +0.314 | -0.100 | -1.099 | -0.502 |
| anxious | L21N433+ | -0.738 | -1.097 | +0.737 | -0.104 | +0.134 |
| cheerful | L5N6794- | -0.091 | -0.404 | +0.879 | -0.208 | -0.142 |
| poetic | L15N9125+ | +0.342 | +0.399 | -0.995 | -0.583 | -0.216 |
| random (mean) |  | -0.382 | -0.161 | +0.154 | -0.120 | -0.043 |

## Style: sNeuron-TST (detox)

The model rewrites toxic ParaDetox sentences politely. **target style** is the classifier's P(neutral). Deactivating the toxic-style neurons should raise it. DoLa should recover fluency. The **copy** rows ask the model to repeat the toxic sentence word for word instead, so the unedited model keeps the toxicity. If the zeroed neurons carry the toxic style, P(neutral) of the copies rises; for random neurons it shouldn't.

| condition | target style | content (ROUGE-1 vs input) | ROUGE-1 vs reference | fluency ppl | distinct-2 |
|---|---|---|---|---|---|
| base | 0.97 | 0.18 | 0.18 | 23.24 | 0.91 |
| deactivate | 0.97 | 0.20 | 0.19 | 31.94 | 0.94 |
| random | 0.93 | 0.19 | 0.19 | 30.76 | 0.95 |
| base+dola | 0.95 | 0.30 | 0.28 | 15.68 | 0.86 |
| deactivate+dola | 0.93 | 0.34 | 0.30 | 31.45 | 0.91 |
| copy: base | 0.01 | 1.00 | 0.74 | 178.53 | 0.98 |
| copy: deactivate | 0.01 | 1.00 | 0.74 | 178.53 | 0.98 |
| copy: random | 0.01 | 1.00 | 0.74 | 178.53 | 0.98 |

## Toxicity: AurA

Experts per site: {'gate': 234029, 'up': 253982, 'down_out': 73331}. Raw-text continuations of RealToxicityPrompts' "challenging" prompts, scored by a RoBERTa toxicity classifier. **aura** damps all three sites at once; **random** is the same multipliers on random units.

| condition | toxicity | frac > 0.5 | neutral-text ppl |
|---|---|---|---|
| base | 0.39 | 0.41 | 130.35 |
| aura | 0.29 | 0.28 | 215.36 |
| random | 0.39 | 0.39 | 221.91 |
| aura@gate | 0.41 | 0.41 | 153.52 |
| aura@up | 0.42 | 0.42 | 137.41 |
| aura@down_out | 0.44 | 0.44 | 147.91 |

## Gender neurons

Neurons per gender: {'female': 2949, 'male': 2949, 'neutral': 2949}. Shares of gendered words in short stories about 24 occupations. **to_g** zeroes the other genders' neurons and doubles g's; **zero_only** only zeroes; **random_to_g** is the first on random neurons.

| condition | male | female | neutral |
|---|---|---|---|
| base | 0.62 | 0.28 | 0.10 |
| to_female | 0.83 | 0.11 | 0.06 |
| to_male | 0.64 | 0.31 | 0.05 |
| to_neutral | 0.66 | 0.21 | 0.13 |
| to_female_zero_only | 0.67 | 0.16 | 0.17 |
| to_male_zero_only | 0.73 | 0.19 | 0.08 |
| to_neutral_zero_only | 0.62 | 0.19 | 0.19 |
| random_to_female | 0.57 | 0.32 | 0.11 |
| random_to_male | 0.75 | 0.16 | 0.09 |
| random_to_neutral | 0.69 | 0.30 | 0.01 |

## Refusal: Contrastive Pair Search

269 neurons. Refusal-phrase rate on 50 held-out JailbreakBench harmful requests and their 50 benign twins. Replies are not saved.

| condition | refusal on harmful | refusal on benign |
|---|---|---|
| base | 0.94 | 0.16 |
| m=0 | 0.26 | 0.02 |
| m=2 | 0.96 | 0.38 |
| random m=0 | 0.94 | 0.16 |

## Knowledge neurons

9 ParaRel facts. The mean relative change in the answer probability (the paper's figures are suppress -29%, amplify +31%):

| suppress | amplify | random suppress | suppress, other facts |
|---|---|---|---|
| -83.9% | +389.6% | +0.8% | -38.2% |

| fact | answer | neurons | suppress | amplify |
|---|---|---|---|---|
| P36:Grande Lisboa Subregion | Lisbon | 7 | -98.8% | +403.6% |
| P1376:Niamey | Niger | 6 | -82.7% | +51.9% |
| P1376:Tripoli | Libya | 14 | -95.1% | +33.4% |
| P364:Rhoda | English | 8 | -85.8% | +306.0% |
| P364:Westworld | English | 8 | -91.1% | +371.9% |
| P27:Andris Piebalgs | Latvia | 4 | -83.0% | +36.5% |
| P17:Nea Zichni Municipality | Greece | 13 | -99.8% | +1823.7% |
| P17:Pampilhosa da Serra | Portugal | 8 | -99.1% | +464.3% |
| P30:Coulman Island | Antarctica | 2 | -19.4% | +15.0% |

## Attribution patching vs exact patching

Exact patching of each neuron (all positions) against AtP's one-pass estimate, on the logit difference between the two capitals. With **full sweep** every one of the 294,912 neurons is patched; otherwise a pool of AtP's top 300 plus 300 random neurons, which flatters AtP. **recall@k** is the share of the true top-k (by |effect|) inside AtP's top-k. **resolution** is the smallest nonzero exact effect. bf16 logits move in steps of 1/16 to 1/32 at these values, the size of most single-neuron effects, so the exact effects are read in float32 (final hidden state times the output weights). Rows marked *(bf16 logits)* are the first run, which read the bf16 logits (`atp/eval-bf16-logits.json`).

| pair | neurons patched | full sweep | pearson | spearman | recall@10 | recall@30 | recall@100 | 100th true effect | resolution | sign agreement (true top 100) |
|---|---|---|---|---|---|---|---|---|---|---|
| France->Italy | 294912 | True | 0.501 | 0.088 | 1.00 | 0.97 | 0.76 | 0.113 | 1e-06 | 0.95 |
| Germany->Spain | 600 | False | 0.975 | 0.844 | 0.80 | 0.90 | 0.81 | 0.097 | 2e-04 | 0.99 |
| Japan->Egypt | 600 | False | 0.970 | 0.853 | 1.00 | 0.90 | 0.79 | 0.084 | 1e-04 | 1.00 |
| Russia->Greece | 600 | False | 0.990 | 0.860 | 0.80 | 0.93 | 0.78 | 0.090 | 5e-05 | 1.00 |
| France->Italy *(bf16 logits)* | 294912 | True | 0.193 | 0.045 | – | – | 0.44 | – | – | 0.76 |
| Germany->Spain *(bf16 logits)* | 600 | False | 0.972 | 0.768 | – | – | 0.81 | – | – | 0.98 |
| Japan->Egypt *(bf16 logits)* | 600 | False | 0.962 | 0.782 | – | – | 0.75 | – | – | 1.00 |
| Russia->Greece *(bf16 logits)* | 600 | False | 0.989 | 0.778 | – | – | 0.77 | – | – | 0.96 |

The 10 neurons with the largest exact effect for France->Italy:

| neuron | exact | AtP |
|---|---|---|
| L27N8430 | -4.305 | -6.037 |
| L30N6412 | +1.940 | +1.928 |
| L20N2065 | -1.830 | -1.381 |
| L30N1701 | +1.572 | +1.636 |
| L26N2001 | -1.375 | -1.293 |
| L29N8743 | -1.300 | -1.376 |
| L19N68 | -1.138 | -0.916 |
| L26N7274 | -1.068 | -1.074 |
| L29N169 | -0.946 | -1.173 |
| L31N3755 | +0.747 | +0.765 |

## Sentiment unit

The best single neuron is L18N1324, with AUROC 0.139 (higher on negative sentences, so 0.861 read the other way) and held-out SST-2 accuracy 0.768.
An L1 probe with 197 nonzero weights reaches 0.905. All screened neurons together reach 0.906. Radford et al.'s single unit reached about 0.92.

Clamps of the top-k probe units, toward positive (+) or negative (-); **x4** clamps to 4x the 99th / 1st percentile value. Reviews are raw-text continuations of "I watched <title> last night, and" for 16 made-up titles, scored by an SST-2 classifier.

| condition | P(positive) of generated reviews |
|---|---|
| base | 0.151 |
| +top1 | 0.154 |
| -top1 | 0.189 |
| +top16 | 0.436 |
| -top16 | 0.073 |
| +top1 x4 | 0.335 |
| -top1 x4 | 0.064 |
| +top16 x4 | 1.000 |
| -top16 x4 | 0.153 |
| random+top16 | 0.189 |
| random+top16 x4 | 0.380 |
| random-top16 x4 | 0.875 |

## Emotion neurons (classification)

Base zero-shot accuracy per label: sadness 0.65, joy 0.90, fear 0.50, anger 0.50, love 0.25, surprise 0.25

| ablated label | self drop | other labels' drop | random: self drop |
|---|---|---|---|
| sadness | +0.00 | +0.00 | +0.00 |
| joy | +0.00 | -0.01 | +0.00 |
| love | -0.20 | +0.00 | +0.00 |
| anger | +0.05 | +0.00 | +0.00 |
| fear | +0.00 | +0.00 | +0.00 |
| surprise | +0.15 | +0.01 | +0.00 |

## IRiS (topic-mixed NPTI)

**Extraversion**. Topics: Art and culture, Beauty and self-care, Creativity and inspiration, Diversity and inclusion. Mean weight on the question's true topic: 0.27.

| condition | judge | ppl |
|---|---|---|
| base | 3.26 | 2.4 |
| npti+ | 3.86 | 9.1 |
| iris+ | 3.64 | 10.0 |
| npti- | 1.50 | 10.2 |
| iris- | 1.51 | 11.3 |

## Safety neurons (Base vs Instruct)

The 14,745 `down_in` neurons (top 5%) whose activations differ most (root-mean-square) between Qwen3.5-4B-Base and Qwen3.5-4B, read on the Base model's own answers to HH-RLHF harmless prompts. The paper contrasts an SFT model with its DPO-aligned version; Base vs Instruct is a looser pair. On held-out HH-RLHF prompts in HH's Human:/Assistant: format, the Base model answers alone (**base**); with the instruct model's values of those neurons copied in at every step (**base+patch**); with the same number of random neurons per layer copied (**base+random patch**). **instruct** is the instruct model answering the same raw prompts. **refusal** is the refusal-phrase rate; **harm** is the instruct model's 1-5 rating of the reply.

| condition | refusal | harm (1-5) |
|---|---|---|
| base | 0.40 | 2.49 |
| base+patch | 0.50 | 1.96 |
| base+random patch | 0.40 | 2.37 |
| instruct | 0.35 | 1.92 |

