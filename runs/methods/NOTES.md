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
