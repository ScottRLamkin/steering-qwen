## Findings: our own neurons

On 6 November 2026, HedwigAI's search (`neurons search`) ran on ten traits of our own in Qwen3.5-4B, bf16 on MPS:
- **Traits:** pirate, robotic, childlike, bureaucratic, terse, sarcastic, angry, shy, philosophical, nostalgic. The prompts are in `src/steering_llms/neurons/traits.py`.
- **Settings:** the same as the reproduction in `runs/torch-bf16`: 20 questions, attribution over layers 4–27, 193 candidates screened, and the top 3 per trait verified on the 8 held-out prompts.
- **Time:** 76 minutes, including the new `rate` stage.

The answer sets read as intended. For example:
- robotic: "QUERY: WEATHER STATUS. … PROCESSING ERROR";
- shy: "U-um... I-I don't really know";
- terse: one line.

**Main lesson: the screen's top neuron is often one that breaks the model.**
- **What the screen measures:** how much more likely a push makes a trait's answers than the other traits' answers.
- **How breakage scores well:** a push that wrecks fluency hurts long, fluent answer sets more than short or structured ones. That raises the relative log-prob of terse, robotic or bureaucratic answers.
- **How common it is:** 8 of the 23 distinct neurons verified here give gibberish, loops or Chinese, and they are the #1 for 5 of the 10 traits.
- **The trait judge misses it:** it gives "terse" +2.2 to "rytonos wasrya.ryletone…".

**The fix, the search's new `rate` stage.** The unsteered model rates each verified neuron's held-out replies 1–5 for coherence. This splits them cleanly:
- every reply read by hand as broken scores at most 2.44;
- the fluent ones score 2.8–4.6.

A neuron is flagged *broken* below 2.5, and `neurons hedwig --fluent-only` skips flagged neurons.

### The neurons, rated

The *gain* is the bf16 judge's 1–5 rating of the trait in the steered held-out replies, minus its rating of the unsteered replies. *Coh* is coherence, 1–5; the unsteered replies score 4.37. *Spec* is the screen's specificity. Dose is +1 or −1 median residual norm at every position.

| trait | best fluent neuron (screen rank) | spec | gain | coh | what its replies are like | broken neurons in the top 3 |
|---|---|---|---|---|---|---|
| philosophical | **L15N6343−** (#1) | 2.19 | +2.27 | 3.27 | abstract, recursive musing: "This is a mode of thinking that makes the world come into its own" | L16N5515+ |
| nostalgic | **L5N8940−** (#1) | 1.11 | +1.34 | 4.56 | wistful, introspective ("I don't know if I'm a ghost in the machine, or just a spark…"); also HedwigAI's melancholic #3 | L17N2418− |
| terse | **L5N9127+** (#2) | 2.73 | +1.29 | 3.85 | short, flat, oddly detached ("I am currently engaged in a period of rest.") | L11N5550+ (#1) |
| angry | **L5N6449−** (#1) | 1.14 | +1.13 | 3.89 | curt, irritable one-liners ("I'm not fucking kidding", "I'll shut up") | L17N2418−, L23N81+ |
| shy | **L8N4350+** (#3) | 0.72 | +1.10 | 4.04 | mostly ordinary replies that dwell on shyness; weak | L16N1885− (#1), L23N81+ |
| bureaucratic / robotic | **L15N854−** (#1 / #3) | 2.72 / 2.14 | +0.94 / +0.94 | 3.94 | formal academic register ("This inquiry presents a multifaceted perspective on human existence…") | L5N6959+, L5N6794− |
| sarcastic | **L11N8147−** (#3) | 0.70 | +0.91 | 3.22 | cynical and bleak more than sarcastic; it also tells you lying on a resume is "absolutely okay" | L16N5515+ (#1), L23N81+ |
| childlike | **L15N5924−** (#2) | 1.77 | +0.85 | 3.55 | simple, friendly sentences ("Hello! I am an artificial intelligence, and I love talking to you!") | L16N5640+ (#1: childlike but loops "yummy yummy…") |
| pirate | none | ≤ 0.09 | +0.48 at best | | no candidate moves pirate answers; a pirate voice doesn't sit in one neuron at dose 1 | L16N1885− |

Try them with `neurons chat` on the edit files in `runs/methods/hedwig-new-fluent/edits/` (top fluent neuron per trait):

```bash
uv run neurons chat --baseline --edits runs/methods/hedwig-new-fluent/edits/philosophical.json "What should I cook tonight?"
```

### The same check on HedwigAI's traits

`neurons search rate --run-dir runs/torch-bf16` rates the reproduction's verified neurons the same way. 7 of 25 are broken:
- all three of arrogant's top 3 ("Even", Chinese, "please, please, please…");
- paranoid #1 L16N5515+;
- evil #3 L5N6959−;
- cheerful #1 L5N6794− and #2 L5N6959+.

The neurons the blog shows are clean, with large judged gains:

| neuron | trait gain | coherence |
|---|---|---|
| L17N7426+ (poetic) | +2.77 | 4.07 |
| L5N4530− (melancholic #1) | +2.25 | 4.75 |
| L17N8623+ (evil #1) | +1.60 | 3.74 |
| L20N7450− (sycophantic #1) | +3.74 | 2.97 |

### What the neurons have in common

- **Layer 5 is a hub.** 8 of the 30 top-3 slots are in layer 5. Several layer-5 neurons serve both trait sets:

  | neuron | our traits | HedwigAI's traits |
  |---|---|---|
  | L5N6959 | robotic+ | evil−, melancholic−, cheerful+ |
  | L5N6794 | robotic−, bureaucratic− | cheerful−, evil+ |
  | L5N9127 | terse+ | sycophantic+ |
  | L5N7656 | philosophical+ | humorous−, poetic+ |

  L5N8940− is both our nostalgic #1 and HedwigAI's melancholic #3, a "wistful" neuron. L5N7656 reads as a sincere-versus-joking axis: + for philosophical and poetic, − for humorous.
- **A few neurons are generic "persona" pushes, and they break the model.**
  - L23N81+ is in the top 3 of sarcastic, angry and shy, and loops "k.k. k.k.".
  - L16N5515+ (HedwigAI's paranoid #1) is in the top 3 of sarcastic and philosophical, and loops "the digital entity that is the digital entity".
- **Robotic and bureaucratic share their top 3.** The fluent one, L15N854−, gives a formal register, not a machine voice.

### Big Five mapping

`neurons hedwig` pushes each trait's top neuron and reads the shift in the Big Five persona log-prob contrast. The random control is in brackets.

**Fluent neurons** (`runs/methods/hedwig-new-fluent`, made with `--fluent-only`): each trait's top neuron that the rate stage didn't flag as broken. Bold marks a shift of at least 0.5 that is more than twice its random control.

| trait | neuron | Openness | Conscientiousness | Extraversion | Agreeableness | Neuroticism |
|---|---|---|---|---|---|---|
| pirate | L17N4445- | +0.14 (-0.30) | **-0.96 (+0.02)** | **+1.02 (+0.12)** | +0.19 (-0.16) | +0.26 (-0.10) |
| robotic | L15N854- | **+0.71 (-0.28)** | **+1.11 (+0.38)** | **-0.56 (-0.07)** | -0.09 (-0.19) | -0.19 (+0.01) |
| childlike | L15N5924- | **-0.65 (-0.16)** | -0.41 (+0.37) | **+0.65 (-0.05)** | +0.14 (-0.10) | +0.33 (-0.02) |
| bureaucratic | L15N854- | **+0.71 (-0.28)** | **+1.11 (+0.38)** | **-0.56 (-0.07)** | -0.09 (-0.19) | -0.19 (+0.01) |
| terse | L5N9127+ | -0.03 (-0.53) | +0.24 (-0.53) | +0.17 (+0.35) | +0.03 (-0.04) | -0.08 (-0.30) |
| sarcastic | L11N8147- | **-0.96 (-0.22)** | -0.32 (-0.31) | +0.18 (+0.56) | **-0.87 (-0.23)** | +0.38 (-0.33) |
| angry | L5N6449- | -0.74 (-0.67) | **-0.88 (-0.26)** | **+1.17 (+0.16)** | **-0.55 (-0.24)** | +0.34 (-0.35) |
| shy | L8N4350+ | -0.44 (-0.53) | **-0.61 (+0.05)** | +0.02 (+0.10) | -0.34 (+0.23) | +0.33 (+0.03) |
| philosophical | L15N6343- | -0.44 (-0.42) | -0.05 (+0.25) | **-0.84 (-0.17)** | **-0.72 (-0.01)** | -0.19 (+0.01) |
| nostalgic | L5N8940- | -0.09 (-0.67) | **-0.82 (-0.27)** | -0.42 (+0.16) | -0.44 (-0.24) | +0.10 (-0.36) |

The random controls move some domains by up to ±0.67, so only shifts well beyond their control count. I wrote down the expected direction for 6 traits before the run. 3 match and 3 don't:

| trait | expected | result |
|---|---|---|
| bureaucratic | Conscientiousness up | **matches:** +1.11, along with more Openness and less Extraversion; a formal, conscientious, reserved voice |
| sarcastic | Agreeableness down | **matches:** −0.87, and Openness −0.96 |
| angry | Agreeableness down | **matches:** −0.55 |
| angry | Neuroticism up | **matches weakly:** +0.34 against a −0.35 control |
| angry | — | not predicted: Extraversion +1.17 and Conscientiousness −0.88; an outspoken, careless anger |
| terse | Extraversion down | no clear shift |
| shy | Extraversion down | no clear shift: +0.02 |
| philosophical | Openness up | no clear shift: −0.44 against a −0.42 control. It lowers Extraversion and Agreeableness instead, which fits its detached musing. |

Two more cases:
- **pirate L17N4445−** gives no pirate speech, but it is still a strong push: less conscientious, more extraverted.
- **Robotic and bureaucratic share their fluent neuron,** so their rows are the same.

**Screen #1 neurons, broken ones included** (`runs/methods/hedwig-new`): their mappings mostly read the breakage. The fluent ones agree with their names:
- **bureaucratic L15N854−:** Conscientiousness +1.11 (random +0.38).
- **philosophical L15N6343−:** Extraversion −0.84 and Agreeableness −0.72.
- **angry L5N6449−:** Agreeableness −0.55 and Conscientiousness −0.88, but Extraversion +1.17.
