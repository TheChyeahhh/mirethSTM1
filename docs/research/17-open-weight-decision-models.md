# 17. Open-weight decision models: GLiNER2.5-Decide, Mapika decider and others

Question from the founder: other open models make typed decisions locally (fastino/GLiNER2.5-Decide and relatives). Can we own the weights, and can we plug such models into MirethSTM1?

Tags: VERIFIED (read in a card, code or config, or measured in our own run), UNVERIFIED (vendor claim or not checked), CONTRADICTED (a card or earlier note was wrong). Date of research: 2026-10-01.

Method notes. All runs were CPU only, float32, on a Ryzen 9 5900X (12 cores), Windows 11, torch 2.11 (CPU use enforced), transformers 5.18, gliner2 2.0.0. No GPU code was run and no paid API was called. Accuracy rows are 300 per task, seed 0, drawn with the repo's committed sampler (commit 18c2455). The MirethSTM1 columns are read from the benchmark's own per-row files for exactly the same rows (VERIFIED: gold labels match on all 1,500 rows). A CPU run of the committed engine matched its GPU file on 100 of 100 rows.

## 1. What each model is, its license, and what "owning the weights" means

"Owning" here means: you can download the file, keep it forever, run it offline, fine-tune it and ship it, with no one able to switch it off. It does not mean you own the training data or can retrain it from scratch.

| Model | What it is | License (source) | What you get |
| --- | --- | --- | --- |
| fastino/GLiNER2.5-Decide | Encoder (DeBERTa-v3-large) classifier with a trained label head. Not a language model: it cannot write or follow free instructions. | Apache-2.0 (VERIFIED, card metadata and HF API). The HF repo has no LICENSE file (VERIFIED, 404), so the grant lives only in the metadata. Not gated. | Weights (1.95 GB fp32), fine-tuning supported by the library. Training data and code unpublished (VERIFIED absent). Base-encoder license not read (UNVERIFIED). One secondary source says part of the training data was GPT-4 synthetic (UNVERIFIED); that could matter for resale. |
| fastino/GLiNER2.5-multi-Decide | Multilingual sibling, mDeBERTa-v3-base encoder. | Apache-2.0 (VERIFIED), same caveats. | 1.15 GB. Same rights. |
| fastino/GLiNER2.5-Decide-1B | Larger variant on a 1B ModernBERT-style encoder. | Apache-2.0 (VERIFIED), same caveats. | 4.76 GB fp32. |
| Mapika/decider-2b (v11), decider-4b | Trained decision models, Qwen3.5-Base fine-tunes (causal LM, hybrid linear attention). Open reproduction of TypeSafe's wire format. | Apache-2.0 (VERIFIED). Training data is about 95 public datasets plus model-written rows; per-dataset licenses not audited (UNVERIFIED). | Weights 3.8 GB (2B) and 8.4 GB (4B) bf16, plus a Python package and training code on GitHub. Most "ownable" of the group: weights, inference code and recipe are all public. |
| StrandsAgents hobson-v19 | LoRA adapter plus readout head on Qwen3.5-2B-Base. | Apache-2.0 (VERIFIED). Training sets include ShareAlike data (UNVERIFIED legal effect). | Adapter and eval receipts; you also need the base model. |
| perplexity-ai/pplx-decider-v1-27b | 27B decision model. | Apache-2.0 (VERIFIED). | About 49 GiB: does not fit a 12 GB card. Quality reference only. |
| bev-decider-0.4B, StartLux-Decision family | Small trained decision models. | CC-BY-NC-4.0 (VERIFIED): non-commercial. | Not usable in a free product that others may use commercially. Avoid. |
| SurjoLabs LFM2.5-350m-Decide | Encoder. | No license, gated (VERIFIED). | Avoid. |

Sweep result: about 40 new decider-style repos appeared since mid-September; most are quantizations, ports or small specialists of the Mapika family (VERIFIED by HF API search, not each one read).

Card errors found: the GLiNER2.5-Decide card says 340M parameters, we counted 486.4M (CONTRADICTED); Decide-1B is 1,188.8M by our count; multi-Decide is 287.4M (matches).

## 2. Comparison with MirethSTM1's measured numbers

CPU versus GPU, plainly: MirethSTM1's own published speed (37 ms for 1 to 10 questions, 98 ms for 28, RTX 5070) is GPU. Every speed number for the other models below is CPU, measured by us, so the two are not comparable. The only like-for-like speed rows are the ones where we ran our own engine on the same CPU. No GPU number exists for any of the other models from our work. Vendor speeds (for example "3.2 ms on a B300" for Mapika, "38 to 47 ms on T4/L4/A100" and "167 ms on a 48-vCPU Xeon" for Fastino) are vendor claims, UNVERIFIED.

### Accuracy, n=300 per cell (about +/-0.05 per cell), all measured by us

| Task | GLiNER2.5-Decide | Decide-1B | multi-Decide | MirethSTM1 Qwen2.5-1.5B | MirethSTM1 Qwen3-4B |
| --- | --- | --- | --- | --- | --- |
| AG News (4 options) | 0.740 | 0.737 | 0.750 | 0.810 | 0.860 |
| Banking77 (77 options) | 0.700 | 0.663 | 0.577 | 0.537 | 0.690 |
| SST-2 yes/no | 0.877 | 0.823 | 0.837 | 0.887 | 0.887 |
| SST-2 choice | 0.887 | 0.847 | 0.823 | 0.913 | 0.887 |
| Yelp 5 levels | 0.560 | 0.480 | 0.473 | 0.380 | 0.413 |
| Mean | 0.753 | 0.710 | 0.692 | 0.705 | 0.747 |

Same-row paired differences, GLiNER2.5-Decide against our 1.5B: AG News -0.070, Banking77 +0.163, Yelp +0.180 (all clear); SST-2 not clear. Against our 4B: AG News -0.120 and Yelp +0.147 clear; the rest not clear. Reading: the 340M-class encoder matches our 4B on average with a different profile (worse at news topics, better at many options and ordered levels). Decide-1B is no better than the smaller one anywhere.

The GLiNER numbers use the model card's own calling style. Adding our instructions and criteria as prompts and descriptions changed little, with one real failure: yes/no with true/false descriptions attached collapses on Decide (0.877 to 0.447) and hurts the 1B (0.823 to 0.630). So these models are wording-sensitive.

### Mapika decider-2b, first 100 rows per task (about +/-0.08), run by us on CPU

| Task | decider-2b | MirethSTM1 1.5B | MirethSTM1 4B |
| --- | --- | --- | --- |
| AG News | 0.920 | 0.850 | 0.910 |
| Banking77 | 0.830 | 0.480 | 0.690 |
| SST-2 yes/no | 0.920 | 0.900 | 0.870 |
| SST-2 choice | 0.910 | 0.920 | 0.890 |
| Yelp | 0.710 | 0.340 | 0.440 |
| Mean | 0.858 | 0.698 | 0.760 |

This is NOT a fair zero-shot win. Mapika's public training code lists ag_news, banking77, sst2 and yelp as training tasks (VERIFIED in its GitHub training code). Our rows come from test and validation splits, so the model has not seen those exact rows, but it was trained on those four datasets. Whether the GLiNER models saw them is UNVERIFIED (no data list). The honest reading: a trained decision model can be much better on tasks like these, and we cannot yet say by how much on unseen tasks.

Yelp as a scale, mean absolute error of the top level (lower is better): GLiNER Decide 0.537, decider-2b 0.310, our 4B 0.740, our 1.5B 0.903.

### Calibration (ECE, 15 bins, mean over the five tasks, lower is better)

- GLiNER2.5-Decide raw 0.114, Decide-1B 0.102, multi-Decide 0.121 (no temperature fitted).
- decider-2b at its stored temperatures 0.071 (100 rows).
- MirethSTM1 1.5B: 0.181 at T=1, 0.109 at the shipped temperature. 4B: 0.241 at T=1, 0.130 at the shipped temperature.
- A temperature fitted out of fold took GLiNER Decide's AG News ECE from 0.195 to 0.070.

### Speed on the same CPU (p50 ms per call, one job at a time, 12 threads, 20 to 30 rows)

| Model | AG News | Banking77 | SST-2 yes/no | Yelp | 5 questions, one call | 20 questions, one call |
| --- | --- | --- | --- | --- | --- | --- |
| GLiNER2.5-multi-Decide | 116 | 511 | 99 | 149 | 160 | 443 |
| GLiNER2.5-Decide | 308 | 1202 | 284 | 409 | 432 | 1069 |
| GLiNER2.5-Decide-1B | 561 | 2284 | 466 | 762 | 873 | 1956 |
| decider-2b (slow reference kernels) | 1114 | 3836 | 950 | 5304 | 3785 | 3991 (packed) |
| MirethSTM1 + Qwen2.5-1.5B (commit 18c2455) | 1462 | 5698 | 1184 | 2050 | 2060 | 3593 |

On this CPU, GLiNER2.5-Decide is about 4 to 5 times faster than our engine with the 1.5B, and multi-Decide about 11 to 14 times faster. decider-2b ran without its fast kernels (UNVERIFIED how much they would help; the library warned they were missing). Whether an encoder beats our 98 ms on the 5070 is UNVERIFIED.

Other limits found: GLiNER takes a 512-token window and does not warn when over (VERIFIED: a 77-label call sends 533 to 587 tokens); 255 labels ran (7.3 s on CPU) but accuracy there was not measured.

## 3. Many questions in one call

AG News rows, one topic question plus four yes/no questions with known answers, plus filler yes/no questions for the 20-question calls. Mean accuracy over the five checked questions.

| Model | n | Alone (5 calls) | 5 in one call | 20 in one call | Same answer as alone |
| --- | --- | --- | --- | --- | --- |
| GLiNER2.5-Decide | 300 | 0.827 | 0.817 | 0.823 to 0.826 | 0.89 to 0.90 |
| GLiNER2.5-Decide-1B | 150 | 0.783 | 0.807 | 0.813 (ours last) | 0.84 to 0.86 |
| decider-2b, default (own row per question) | 100 | 0.914 | 0.914 | not run | 1.000 |
| decider-2b, packed in one row | 100 | 0.914 | 0.930 | 0.908 (ours last) | 0.93 to 0.96 |
| MirethSTM1 1.5B, working tree after today's fix (GPU) | 200 | 0.844 | not run | 0.848 to 0.850 | 0.99 |
| MirethSTM1 1.5B, committed 18c2455 (CPU, 20 rows) | 20 | 0.920 | 0.920 | 0.560 (ours last) | not computed |

Reading: GLiNER answers several questions in one pass and accuracy holds even at 20 questions, but about 10 percent of single answers change when other questions share the call (heads are not independent). decider-2b's default mode gives identical answers to asking alone; its faster packed mode changes 4 to 7 percent. The committed engine still shows the old drop (0.92 to 0.56); the post-fix working tree is flat. That is a repo state fact, not a property of the other models.

## 4. Can they plug into MirethSTM1?

- GLiNER models: no. They are encoders with a trained label head, with no chat template and no language-model head (VERIFIED in config). There is nothing for our scorer to score. They need a separate backend.
- Mapika decider-2b: no as a plain model-list entry. The committed engine refuses it: `ValueError: Mapika/decider-2b is not supported: it has sliding-window or other non-full attention layers` (VERIFIED by running). Cause: 18 of its 24 layers are linear attention, so our packed-tree mask cannot apply (VERIFIED in the cache layer types). Forced past the check it runs but gives confident wrong answers: 0.49 on 100 AG News rows against 0.92 when asked the model's own way (VERIFIED). The refusal is correct. A second cause is the prompt: it was trained on a plain letter-slot layout ("Context ... Question ... Options: (A) ... Answer: ("), not our chat and JSON prompt (VERIFIED in its prompt code); we did not separate the two causes.
- Mapika decider-2b does accept our wire format through its own `system_one()` call (VERIFIED), so a wrapper is thin.
- Control: Qwen3-0.6B is accepted by our loader and matches a one-label-at-a-time reference to 0.0001 (read from the earlier run's saved output).
- Windows note: CUDA_VISIBLE_DEVICES set to an empty string did not hide the GPU on Windows in our run (torch still reported CUDA available). Use "-1".

## 5. Options for MirethSTM1

(a) List such models as comparison rows in our benchmark only.
- Effort: small. A runner per model (the scripts already exist), results pasted into the benchmark table, each row labelled CPU or GPU and "trained on these tasks" where true.
- Risk: low. Main danger is an unfair headline (decider-2b was trained on four of our five tasks), handled by labelling. Gives the founder real numbers without touching the engine.

(b) Add a second backend so a trained decision model can be picked in the console and raced.
- Effort: medium to large. A common `decide(state, questions)` interface returning per-question probabilities in our wire format; one adapter for the GLiNER encoders (hook on the classification head to get full distributions, since the library returns only the winner), one for Mapika decider (letter-slot prompt). Needs a per-model "backend" field in the approved list, text rendering of state, the 512-token limit handled, and a calibration step per question type.
- Risk: medium. Wording sensitivity (the yes/no collapse), 512-token window, cross-question coupling, ordinal weakness on some variants, and the licensing questions on training data. Upside: this is the only route that lets the founder swap in the models that actually beat ours on several tasks, and on CPU it is 4 to 14 times faster.

(c) Load causal-LM decision models through the existing engine and approve them like any other model.
- Effort: small for plain Qwen3-style models (already works), but for the decision-trained models it is a research task: they would need a non-tree path.
- Risk: high for the trained models. Mapika decider is refused today for a sound reason, and forced use gives wrong answers. Only worth it for a decision model that is a full-attention causal LM trained on our prompt shape. None found in this sweep (UNVERIFIED beyond the sweep).

## 6. Recommendation

For the Sunday release: do (a) only, if anything, and keep it out of the engine. Add GLiNER2.5-Decide and decider-2b as labelled comparison rows (CPU numbers, trained-on flag, vendor claims marked). Do not add them to the approved list, and do not promise plug-in support: the loader's refusal of Mapika decider should stay.

After the release: build (b), starting with the GLiNER2.5-Decide encoder adapter, because it matches our 4B on average, is Apache-2.0, small (1 to 2 GB), and gives the full distribution via a head hook. Before committing, close these gaps (all UNVERIFIED today): GPU latency on the 5070 for the encoders and for decider-2b with its fast kernels; accuracy on tasks the decider models were not trained on; behavior past 512 tokens and at 255 options; the training-data terms behind the Fastino and Mapika weights; and the licenses of the base encoders. Skip (c), and skip the 1B, the 12B and 27B models and the non-commercial ones.
