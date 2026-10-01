# MirethSTM1 benchmark: full

Generated 2026-10-01 17:06 UTC by `python -m bench.report` from bench/out/full/ (per-sample JSONL). Models: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-1.7B, HuggingFaceTB/SmolLM3-3B, Qwen/Qwen3-0.6B.

Accuracy runs: seed 0, bf16 on one RTX 5070, torch 2.11.0+cu128, transformers 5.18.0. Samples are stratified by class; texts longer than 2000 characters are cut at a word boundary (only Yelp).

- Qwen2.5-1.5B-Instruct, Qwen3-4B-Instruct-2507, Qwen3-1.7B: the scoring arms on 1000 evaluation rows per dataset (SST-2: the whole 872-row validation split) and 500 calibration rows; normal generation on 300 evaluation rows.
- SmolLM3-3B, Qwen3-0.6B: the scoring arms on 500 evaluation and 300 calibration rows; normal generation on 100. With the same seed these rows are subsets of the larger runs' rows, so all five models share 500 evaluation rows per dataset.
- Qwen2.5-1.5B-Instruct, questions before the state: 200 rows of AG News, SST-2 and Yelp.
- Latency: 10 warmup runs, then 30 timed runs per field count (10 for SmolLM3-3B and Qwen3-0.6B) and 20 per console scenario.

<!-- summary:start -->
## Summary across models

Means over the five datasets (AG News, Banking77, SST-2 yes/no, SST-2 choice, Yelp), each dataset weighted equally, evaluation split. MirethSTM1 and the first-token arm ran the same rows (each model's full evaluation rows, listed above). Normal generation ran fewer rows (300, or 100 for SmolLM3-3B and Qwen3-0.6B); the column next to it scores MirethSTM1 on exactly those rows. Normal generation counts an invalid, hallucinated or missing answer as wrong and is strict about types: a quoted "true" is not a boolean, which alone puts Qwen3-4B-Instruct-2507 at 0.250 and Qwen3-1.7B at 0.033 on SST-2 yes/no. ECE-15 is the mean of the per-dataset values. The shipped T is the pooled T fitted on the model's calibration rows (`mirethstm.calibration.DEFAULT_TEMPERATURES`).

| Model | MirethSTM1 accuracy | First-token accuracy | Normal generation accuracy (valid answers) | MirethSTM1 on the normal-generation rows | MirethSTM1 ECE-15 at T = 1 | MirethSTM1 ECE-15 at shipped T | Shipped T | First token ECE-15 at T = 1 / at its pooled T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 0.713 | 0.670 | 0.555 (0.850) | 0.705 | 0.164 | 0.094 | 2.285 | 0.133 / 0.089 |
| Qwen/Qwen3-4B-Instruct-2507 | 0.752 | 0.692 | 0.611 (0.849) | 0.747 | 0.238 | 0.119 | 8.036 | 0.207 / 0.106 |
| Qwen/Qwen3-1.7B | 0.696 | 0.660 | 0.515 (0.758) | 0.696 | 0.268 | 0.091 | 6.750 | 0.219 / 0.096 |
| HuggingFaceTB/SmolLM3-3B | 0.672 | 0.657 | 0.572 (0.846) | 0.686 | 0.195 | 0.076 | 2.677 | 0.143 / 0.076 |
| Qwen/Qwen3-0.6B | 0.581 | 0.540 | 0.008 (0.008) | 0.576 | 0.291 | 0.127 | 3.830 | 0.250 / 0.118 |

Accuracy on the 500 evaluation rows per dataset that all five models ran, MirethSTM1 / first token:

| Model | AG News | Banking77 | SST-2 yes/no | SST-2 choice | Yelp | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 0.826 / 0.826 | 0.550 / 0.326 | 0.884 / 0.882 | 0.920 / 0.922 | 0.374 / 0.378 | 0.711 / 0.667 |
| Qwen/Qwen3-4B-Instruct-2507 | 0.868 / 0.868 | 0.684 / 0.388 | 0.882 / 0.882 | 0.894 / 0.892 | 0.416 / 0.410 | 0.749 / 0.688 |
| Qwen/Qwen3-1.7B | 0.820 / 0.818 | 0.494 / 0.326 | 0.850 / 0.848 | 0.906 / 0.906 | 0.416 / 0.412 | 0.697 / 0.662 |
| HuggingFaceTB/SmolLM3-3B | 0.800 / 0.802 | 0.266 / 0.168 | 0.856 / 0.866 | 0.898 / 0.900 | 0.538 / 0.548 | 0.672 / 0.657 |
| Qwen/Qwen3-0.6B | 0.724 / 0.728 | 0.478 / 0.286 | 0.530 / 0.526 | 0.822 / 0.822 | 0.350 / 0.338 | 0.581 / 0.540 |

MirethSTM1 p50 latency in ms (warm, batch 1, bf16, RTX 5070; full tables in Latency below). Peak memory: the largest `torch.cuda.max_memory_allocated` of any MirethSTM1 run of the model, in MiB.

| Model | 1 field | 5 fields | 10 fields | 20 fields | 28 fields, support | 28 fields, security review | 20 fields with scores, incident | 255-option router | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 37.2 | 36.5 | 37.1 | 50.6 | 98.1 | 99.3 | 83.3 | 220 | 3545 |
| Qwen/Qwen3-4B-Instruct-2507 | 58.7 | 72.0 | 83.5 | 121 | 274 | 279 | 223 | 637 | 8331 |
| Qwen/Qwen3-1.7B | 48.5 | 44.7 | 44.0 | 52.6 | 110 | 113 | 95.3 | 258 | 3883 |
| HuggingFaceTB/SmolLM3-3B | 48.4 | 52.5 | 64.9 | 83.8 | 184 | 186 | 154 | 409 | 6487 |
| Qwen/Qwen3-0.6B | 45.5 | 45.7 | 44.0 | 44.1 | 64.6 | 66.4 | 54.8 | 163 | 1733 |

<!-- summary:end -->

## Accuracy on the evaluation split

Raw scores (T = 1). Accuracy and macro-F1 do not depend on T (the argmax does not move), so they are given once. Normal generation has no probabilities: an invalid, hallucinated or missing answer counts wrong, and NLL, Brier and ECE do not apply. Salvaged accuracy (secondary): the first allowed value written for the question's key anywhere in the output, even when the JSON is invalid or cut off. Brackets: bootstrap 95% CI (1000 resamples of the evaluation rows). ECE is biased upward on small samples, and more so on resamples (they repeat rows), so its interval can sit above the point estimate.

### ag_news (fancyzhx/ag_news test, 4 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.840 [0.817, 0.861] | 0.840 [0.817, 0.861] | 0.951 [0.802, 1.123] | 0.284 [0.247, 0.326] | 0.125 [0.107, 0.149] | 0.125 | n/a | n/a | n/a | 36.8 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.838 [0.815, 0.860] | 0.838 [0.815, 0.859] | 0.954 [0.805, 1.127] | 0.285 [0.248, 0.327] | 0.127 [0.109, 0.150] | 0.126 | n/a | n/a | n/a | 84.0 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.787 [0.740, 0.830] | 0.794 [0.750, 0.837] | n/a | n/a | n/a | n/a | 0.973 | 0.787 | 0.023 | 386 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | 200 | 0.485 [0.415, 0.550] | 0.475 [0.397, 0.537] | 1.280 [1.116, 1.466] | 0.699 [0.623, 0.786] | 0.198 [0.166, 0.291] | 0.197 | n/a | n/a | n/a | 201 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.883 [0.862, 0.902] | 0.883 [0.863, 0.902] | 2.165 [1.767, 2.599] | 0.221 [0.184, 0.261] | 0.112 [0.094, 0.133] | 0.112 | n/a | n/a | n/a | 59.5 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.883 [0.862, 0.903] | 0.883 [0.863, 0.903] | 2.172 [1.775, 2.605] | 0.222 [0.184, 0.262] | 0.112 [0.093, 0.132] | 0.112 | n/a | n/a | n/a | 133 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.863 [0.827, 0.900] | 0.864 [0.825, 0.900] | n/a | n/a | n/a | n/a | 1.000 | 0.863 | 0.000 | 631 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 1000 | 0.821 [0.796, 0.844] | 0.820 [0.795, 0.841] | 2.791 [2.340, 3.260] | 0.340 [0.297, 0.388] | 0.170 [0.148, 0.195] | 0.169 | n/a | n/a | n/a | 45.5 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 1000 | 0.821 [0.796, 0.843] | 0.820 [0.795, 0.841] | 2.789 [2.337, 3.266] | 0.341 [0.298, 0.389] | 0.169 [0.147, 0.194] | 0.169 | n/a | n/a | n/a | 100 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.760 [0.713, 0.810] | 0.798 [0.755, 0.843] | n/a | n/a | n/a | n/a | 0.903 | 0.847 | 0.333 | 950 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.800 [0.764, 0.836] | 0.797 [0.760, 0.831] | 1.321 [1.029, 1.621] | 0.350 [0.289, 0.413] | 0.164 [0.133, 0.200] | 0.165 | n/a | n/a | n/a | 43.8 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.802 [0.766, 0.836] | 0.799 [0.763, 0.833] | 1.322 [1.031, 1.628] | 0.350 [0.289, 0.411] | 0.161 [0.132, 0.197] | 0.160 | n/a | n/a | n/a | 101 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.680 [0.590, 0.760] | 0.709 [0.616, 0.784] | n/a | n/a | n/a | n/a | 0.940 | 0.710 | 0.100 | 463 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.724 [0.686, 0.760] | 0.715 [0.677, 0.752] | 2.479 [2.036, 2.929] | 0.502 [0.434, 0.571] | 0.241 [0.208, 0.279] | 0.236 | n/a | n/a | n/a | 45.8 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.728 [0.690, 0.766] | 0.719 [0.681, 0.756] | 2.469 [2.035, 2.914] | 0.500 [0.432, 0.567] | 0.235 [0.206, 0.275] | 0.230 | n/a | n/a | n/a | 109 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.430 | 1.000 | 1220 |

### banking77 (mteb/banking77 test, 77 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.545 [0.513, 0.576] | 0.524 [0.491, 0.543] | 2.723 [2.481, 2.951] | 0.696 [0.648, 0.744] | 0.235 [0.209, 0.267] | 0.235 | n/a | n/a | n/a | 80.8 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.323 [0.293, 0.354] | 0.267 [0.246, 0.281] | 3.029 [2.825, 3.232] | 0.805 [0.773, 0.836] | 0.095 [0.081, 0.125] | 0.099 | n/a | n/a | n/a | 109 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.157 [0.117, 0.200] | 0.189 [0.128, 0.211] | n/a | n/a | n/a | n/a | 0.367 | 0.157 | 0.003 | 551 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.670 [0.642, 0.697] | 0.657 [0.624, 0.674] | 6.581 [5.922, 7.240] | 0.637 [0.583, 0.692] | 0.309 [0.282, 0.337] | 0.309 | n/a | n/a | n/a | 200 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.380 [0.351, 0.412] | 0.319 [0.299, 0.333] | 6.512 [5.927, 7.119] | 0.786 [0.750, 0.824] | 0.154 [0.131, 0.180] | 0.152 | n/a | n/a | n/a | 264 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.660 [0.603, 0.720] | 0.647 [0.568, 0.672] | n/a | n/a | n/a | n/a | 0.987 | 0.660 | 0.000 | 925 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 1000 | 0.505 [0.475, 0.535] | 0.506 [0.469, 0.524] | 7.657 [7.016, 8.256] | 0.893 [0.837, 0.949] | 0.429 [0.401, 0.460] | 0.429 | n/a | n/a | n/a | 86.1 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 1000 | 0.327 [0.297, 0.354] | 0.280 [0.257, 0.293] | 6.679 [6.133, 7.249] | 0.863 [0.827, 0.898] | 0.179 [0.159, 0.208] | 0.181 | n/a | n/a | n/a | 128 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.537 [0.480, 0.597] | 0.531 [0.448, 0.551] | n/a | n/a | n/a | n/a | 0.997 | 0.537 | 0.000 | 689 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.266 [0.228, 0.306] | 0.226 [0.183, 0.247] | 4.684 [4.323, 5.071] | 1.035 [0.978, 1.094] | 0.383 [0.348, 0.421] | 0.383 | n/a | n/a | n/a | 147 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.168 [0.134, 0.202] | 0.124 [0.095, 0.139] | 4.653 [4.348, 4.985] | 0.945 [0.915, 0.978] | 0.131 [0.107, 0.161] | 0.129 | n/a | n/a | n/a | 235 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.340 [0.250, 0.430] | 0.245 [0.179, 0.311] | n/a | n/a | n/a | n/a | 0.910 | 0.380 | 0.090 | 1049 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.478 [0.434, 0.526] | 0.463 [0.406, 0.486] | 4.825 [4.264, 5.367] | 0.865 [0.788, 0.939] | 0.388 [0.347, 0.431] | 0.388 | n/a | n/a | n/a | 51.6 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.286 [0.248, 0.328] | 0.235 [0.201, 0.255] | 4.621 [4.125, 5.131] | 0.863 [0.818, 0.912] | 0.165 [0.143, 0.205] | 0.165 | n/a | n/a | n/a | 116 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.210 | 1.000 | 1564 |

### sst2 (stanfordnlp/sst2 validation, 2 labels, n = 872)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 872 | 0.884 [0.862, 0.904] | 0.884 [0.862, 0.903] | 0.331 [0.281, 0.388] | 0.188 [0.160, 0.220] | 0.044 [0.034, 0.068] | 0.044 | n/a | n/a | n/a | 38.4 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 872 | 0.882 [0.860, 0.901] | 0.881 [0.860, 0.900] | 0.330 [0.281, 0.387] | 0.188 [0.160, 0.219] | 0.041 [0.032, 0.066] | 0.042 | n/a | n/a | n/a | 85.4 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.653 [0.597, 0.703] | 0.608 [0.548, 0.665] | n/a | n/a | n/a | n/a | 1.000 | 0.653 | 0.000 | 300 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | 200 | 0.820 [0.770, 0.865] | 0.815 [0.761, 0.863] | 0.402 [0.319, 0.500] | 0.251 [0.198, 0.313] | 0.058 [0.043, 0.117] | 0.066 | n/a | n/a | n/a | 130 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 872 | 0.899 [0.877, 0.917] | 0.899 [0.877, 0.917] | 1.661 [1.344, 2.022] | 0.201 [0.164, 0.244] | 0.101 [0.083, 0.122] | 0.101 | n/a | n/a | n/a | 57.3 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 872 | 0.899 [0.877, 0.917] | 0.899 [0.877, 0.917] | 1.658 [1.339, 2.028] | 0.201 [0.164, 0.243] | 0.101 [0.083, 0.122] | 0.101 | n/a | n/a | n/a | 128 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.250 [0.203, 0.300] | 0.318 [0.282, 0.350] | n/a | n/a | n/a | n/a | 0.287 | 0.250 | 0.000 | 555 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 872 | 0.846 [0.823, 0.873] | 0.844 [0.820, 0.871] | 0.815 [0.652, 0.979] | 0.263 [0.218, 0.304] | 0.121 [0.099, 0.145] | 0.121 | n/a | n/a | n/a | 45.1 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 872 | 0.844 [0.820, 0.870] | 0.842 [0.818, 0.868] | 0.806 [0.650, 0.969] | 0.262 [0.218, 0.302] | 0.124 [0.102, 0.148] | 0.124 | n/a | n/a | n/a | 99.6 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.033 [0.017, 0.057] | 0.063 [0.031, 0.102] | n/a | n/a | n/a | n/a | 0.033 | 0.033 | 0.053 | 778 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.856 [0.824, 0.888] | 0.854 [0.822, 0.886] | 0.369 [0.284, 0.453] | 0.208 [0.162, 0.255] | 0.082 [0.059, 0.114] | 0.082 | n/a | n/a | n/a | 43.3 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.866 [0.834, 0.896] | 0.865 [0.834, 0.896] | 0.364 [0.280, 0.448] | 0.207 [0.160, 0.254] | 0.083 [0.058, 0.113] | 0.082 | n/a | n/a | n/a | 100 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.480 [0.380, 0.580] | 0.625 [0.522, 0.715] | n/a | n/a | n/a | n/a | 0.530 | 0.660 | 0.200 | 473 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.530 [0.484, 0.572] | 0.397 [0.360, 0.434] | 1.366 [1.230, 1.508] | 0.765 [0.695, 0.842] | 0.401 [0.362, 0.446] | 0.399 | n/a | n/a | n/a | 45.8 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.526 [0.478, 0.568] | 0.389 [0.352, 0.425] | 1.354 [1.217, 1.497] | 0.764 [0.696, 0.838] | 0.403 [0.366, 0.448] | 0.403 | n/a | n/a | n/a | 100 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.000 | 1.000 | 1020 |

### sst2_choice (stanfordnlp/sst2 validation, 2 labels, n = 872)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 872 | 0.924 [0.906, 0.940] | 0.924 [0.906, 0.940] | 0.314 [0.243, 0.398] | 0.134 [0.106, 0.166] | 0.063 [0.049, 0.081] | 0.058 | n/a | n/a | n/a | 37.7 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 872 | 0.925 [0.907, 0.942] | 0.925 [0.907, 0.941] | 0.315 [0.244, 0.400] | 0.134 [0.107, 0.167] | 0.060 [0.046, 0.077] | 0.058 | n/a | n/a | n/a | 85.7 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.903 [0.867, 0.937] | 0.903 [0.867, 0.936] | n/a | n/a | n/a | n/a | 1.000 | 0.903 | 0.000 | 348 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 872 | 0.903 [0.883, 0.921] | 0.902 [0.883, 0.921] | 1.379 [1.078, 1.691] | 0.191 [0.155, 0.230] | 0.095 [0.077, 0.116] | 0.095 | n/a | n/a | n/a | 57.2 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 872 | 0.900 [0.880, 0.919] | 0.900 [0.880, 0.919] | 1.380 [1.079, 1.692] | 0.191 [0.155, 0.229] | 0.096 [0.079, 0.116] | 0.096 | n/a | n/a | n/a | 130 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.887 [0.850, 0.920] | 0.888 [0.853, 0.922] | n/a | n/a | n/a | n/a | 0.997 | 0.887 | 0.000 | 563 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 872 | 0.904 [0.884, 0.923] | 0.904 [0.884, 0.923] | 0.783 [0.596, 0.970] | 0.181 [0.145, 0.218] | 0.083 [0.069, 0.106] | 0.084 | n/a | n/a | n/a | 45.0 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 872 | 0.903 [0.883, 0.922] | 0.902 [0.882, 0.922] | 0.776 [0.594, 0.962] | 0.181 [0.144, 0.218] | 0.088 [0.072, 0.109] | 0.086 | n/a | n/a | n/a | 99.2 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.847 [0.803, 0.887] | 0.868 [0.827, 0.903] | n/a | n/a | n/a | n/a | 0.950 | 0.887 | 0.047 | 811 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.898 [0.870, 0.924] | 0.898 [0.870, 0.924] | 0.495 [0.359, 0.638] | 0.191 [0.143, 0.243] | 0.093 [0.070, 0.121] | 0.094 | n/a | n/a | n/a | 43.2 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.900 [0.872, 0.924] | 0.900 [0.872, 0.924] | 0.491 [0.355, 0.632] | 0.191 [0.143, 0.243] | 0.095 [0.070, 0.121] | 0.092 | n/a | n/a | n/a | 100 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.840 [0.770, 0.910] | 0.878 [0.817, 0.933] | n/a | n/a | n/a | n/a | 0.910 | 0.910 | 0.120 | 424 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.822 [0.790, 0.856] | 0.820 [0.788, 0.854] | 0.581 [0.460, 0.701] | 0.287 [0.235, 0.337] | 0.112 [0.088, 0.145] | 0.108 | n/a | n/a | n/a | 45.7 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.822 [0.788, 0.856] | 0.820 [0.787, 0.854] | 0.584 [0.465, 0.703] | 0.290 [0.238, 0.339] | 0.122 [0.096, 0.154] | 0.112 | n/a | n/a | n/a | 98.9 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.040 [0.010, 0.080] | 0.074 [0.018, 0.138] | n/a | n/a | n/a | n/a | 0.040 | 0.340 | 0.960 | 1084 |

### yelp (Yelp/yelp_review_full test, 5 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.372 [0.344, 0.401] | 0.331 [0.306, 0.355] | 2.112 [1.999, 2.233] | 0.916 [0.877, 0.957] | 0.352 [0.323, 0.380] | 0.354 | n/a | n/a | n/a | 37.9 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.383 [0.354, 0.410] | 0.342 [0.315, 0.367] | 2.110 [1.998, 2.233] | 0.916 [0.877, 0.959] | 0.343 [0.314, 0.374] | 0.345 | n/a | n/a | n/a | 87.0 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.273 [0.223, 0.323] | 0.252 [0.208, 0.297] | n/a | n/a | n/a | n/a | 0.910 | 0.273 | 0.000 | 356 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | 200 | 0.435 [0.375, 0.505] | 0.357 [0.309, 0.408] | 1.559 [1.342, 1.784] | 0.775 [0.682, 0.861] | 0.314 [0.252, 0.372] | 0.314 | n/a | n/a | n/a | 256 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.403 [0.371, 0.431] | 0.375 [0.347, 0.401] | 9.960 [9.360, 10.598] | 1.161 [1.106, 1.223] | 0.572 [0.543, 0.603] | 0.572 | n/a | n/a | n/a | 76.9 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.400 [0.369, 0.428] | 0.371 [0.344, 0.397] | 9.963 [9.364, 10.593] | 1.163 [1.108, 1.224] | 0.574 [0.546, 0.605] | 0.573 | n/a | n/a | n/a | 150 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.393 [0.340, 0.447] | 0.368 [0.321, 0.412] | n/a | n/a | n/a | n/a | 0.977 | 0.393 | 0.000 | 584 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 1000 | 0.402 [0.370, 0.431] | 0.373 [0.340, 0.401] | 5.464 [5.091, 5.836] | 1.102 [1.048, 1.163] | 0.535 [0.505, 0.568] | 0.534 | n/a | n/a | n/a | 45.9 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 1000 | 0.403 [0.371, 0.434] | 0.374 [0.341, 0.403] | 5.439 [5.062, 5.819] | 1.101 [1.044, 1.162] | 0.534 [0.503, 0.567] | 0.532 | n/a | n/a | n/a | 104 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.400 [0.343, 0.453] | 0.392 [0.339, 0.445] | n/a | n/a | n/a | n/a | 0.907 | 0.440 | 0.077 | 815 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.538 [0.496, 0.584] | 0.494 [0.453, 0.537] | 1.466 [1.302, 1.633] | 0.674 [0.611, 0.732] | 0.252 [0.215, 0.294] | 0.252 | n/a | n/a | n/a | 60.6 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.548 [0.504, 0.592] | 0.506 [0.465, 0.549] | 1.474 [1.309, 1.645] | 0.674 [0.610, 0.733] | 0.247 [0.215, 0.295] | 0.238 | n/a | n/a | n/a | 120 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.520 [0.420, 0.610] | 0.510 [0.404, 0.593] | n/a | n/a | n/a | n/a | 0.940 | 0.550 | 0.050 | 498 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.350 [0.310, 0.388] | 0.264 [0.234, 0.293] | 2.245 [2.077, 2.410] | 0.909 [0.858, 0.960] | 0.312 [0.273, 0.354] | 0.312 | n/a | n/a | n/a | 45.4 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.338 [0.296, 0.378] | 0.255 [0.225, 0.284] | 2.254 [2.085, 2.422] | 0.911 [0.860, 0.963] | 0.326 [0.285, 0.363] | 0.326 | n/a | n/a | n/a | 100 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.110 | 1.000 | 1141 |

## Calibration

T is one scalar fitted by NLL on the calibration split (train rows, none of whose texts occur in the evaluation split): per dataset, and pooled over all datasets per model and arm. ECE and NLL are measured on the evaluation split only. n/a: the fit did not converge inside [0.05, 20] (too few calibration rows) or the split was not run. ECE brackets can sit above the point estimate (see Accuracy).

| Model | Arm | Dataset | T dataset | T pooled | ECE-15 at T = 1 | NLL at T = 1 | ECE-15 at T dataset | NLL at T dataset | ECE-15 at T pooled | NLL at T pooled | Diagram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | ag_news | 3.039 | 2.285 | 0.125 [0.107, 0.149] | 0.951 | 0.044 [0.031, 0.069] | 0.498 | 0.056 [0.049, 0.086] | 0.530 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.ag_news.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | banking77 | 2.015 | 2.285 | 0.235 [0.209, 0.267] | 2.723 | 0.078 [0.055, 0.108] | 2.010 | 0.125 [0.102, 0.157] | 2.033 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.banking77.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | sst2 | 1.647 | 2.285 | 0.044 [0.034, 0.068] | 0.331 | 0.034 [0.024, 0.056] | 0.315 | 0.090 [0.071, 0.109] | 0.344 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | sst2_choice | 1.995 | 2.285 | 0.063 [0.049, 0.081] | 0.314 | 0.019 [0.013, 0.039] | 0.222 | 0.017 [0.014, 0.038] | 0.223 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2_choice.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | yelp | 3.493 | 2.285 | 0.352 [0.323, 0.380] | 2.112 | 0.084 [0.075, 0.119] | 1.376 | 0.179 [0.155, 0.209] | 1.439 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.yelp.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | ag_news | 3.033 | 2.341 | 0.127 [0.109, 0.150] | 0.954 | 0.046 [0.033, 0.071] | 0.499 | 0.059 [0.050, 0.086] | 0.527 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.ag_news.first_token.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | banking77 | 2.007 | 2.341 | 0.095 [0.081, 0.125] | 3.029 | 0.085 [0.063, 0.112] | 2.503 | 0.104 [0.083, 0.134] | 2.524 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.banking77.first_token.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | sst2 | 1.645 | 2.341 | 0.041 [0.032, 0.066] | 0.330 | 0.032 [0.024, 0.057] | 0.315 | 0.092 [0.074, 0.111] | 0.346 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2.first_token.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | sst2_choice | 2.002 | 2.341 | 0.060 [0.046, 0.077] | 0.315 | 0.019 [0.013, 0.039] | 0.222 | 0.025 [0.019, 0.043] | 0.224 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2_choice.first_token.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | yelp | 3.494 | 2.341 | 0.343 [0.314, 0.374] | 2.110 | 0.093 [0.076, 0.122] | 1.376 | 0.168 [0.147, 0.200] | 1.431 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.yelp.first_token.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | ag_news | n/a | n/a | 0.198 [0.166, 0.291] | 1.280 | n/a | n/a | n/a | n/a | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.ag_news.questions_first.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | sst2 | n/a | n/a | 0.058 [0.043, 0.117] | 0.402 | n/a | n/a | n/a | n/a | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2.questions_first.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | yelp | n/a | n/a | 0.314 [0.252, 0.372] | 1.559 | n/a | n/a | n/a | n/a | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.yelp.questions_first.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | ag_news | 8.009 | 8.036 | 0.112 [0.094, 0.133] | 2.165 | 0.039 [0.027, 0.060] | 0.415 | 0.039 [0.027, 0.059] | 0.415 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.ag_news.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | banking77 | 6.430 | 8.036 | 0.309 [0.282, 0.337] | 6.581 | 0.056 [0.043, 0.087] | 1.526 | 0.146 [0.126, 0.176] | 1.606 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.banking77.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | sst2 | 8.934 | 8.036 | 0.101 [0.083, 0.122] | 1.661 | 0.034 [0.023, 0.056] | 0.281 | 0.045 [0.031, 0.065] | 0.284 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | sst2_choice | 8.409 | 8.036 | 0.095 [0.077, 0.116] | 1.379 | 0.040 [0.027, 0.062] | 0.269 | 0.049 [0.034, 0.069] | 0.268 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2_choice.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | yelp | 16.728 | 8.036 | 0.572 [0.543, 0.603] | 9.960 | 0.079 [0.060, 0.114] | 1.363 | 0.317 [0.290, 0.349] | 1.607 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.yelp.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | ag_news | 8.002 | 8.552 | 0.112 [0.093, 0.132] | 2.172 | 0.037 [0.026, 0.059] | 0.416 | 0.030 [0.021, 0.050] | 0.414 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.ag_news.first_token.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | banking77 | 6.643 | 8.552 | 0.154 [0.131, 0.180] | 6.512 | 0.059 [0.048, 0.087] | 2.219 | 0.127 [0.107, 0.155] | 2.285 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.banking77.first_token.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | sst2 | 8.924 | 8.552 | 0.101 [0.083, 0.122] | 1.658 | 0.033 [0.022, 0.055] | 0.281 | 0.035 [0.021, 0.058] | 0.281 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2.first_token.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | sst2_choice | 8.383 | 8.552 | 0.096 [0.079, 0.116] | 1.380 | 0.038 [0.027, 0.060] | 0.269 | 0.035 [0.025, 0.056] | 0.269 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2_choice.first_token.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | yelp | 16.704 | 8.552 | 0.574 [0.546, 0.605] | 9.963 | 0.080 [0.061, 0.114] | 1.362 | 0.302 [0.274, 0.332] | 1.560 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.yelp.first_token.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | ag_news | 8.731 | 6.750 | 0.170 [0.148, 0.195] | 2.791 | 0.044 [0.036, 0.072] | 0.527 | 0.081 [0.063, 0.105] | 0.566 | [png](benchmark/Qwen--Qwen3-1.7B.ag_news.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | banking77 | 6.311 | 6.750 | 0.429 [0.401, 0.460] | 7.657 | 0.049 [0.040, 0.081] | 2.096 | 0.058 [0.045, 0.089] | 2.107 | [png](benchmark/Qwen--Qwen3-1.7B.banking77.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | sst2 | 4.679 | 6.750 | 0.121 [0.099, 0.145] | 0.815 | 0.029 [0.021, 0.054] | 0.364 | 0.087 [0.066, 0.109] | 0.389 | [png](benchmark/Qwen--Qwen3-1.7B.sst2.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | sst2_choice | 4.288 | 6.750 | 0.083 [0.069, 0.106] | 0.783 | 0.031 [0.019, 0.052] | 0.278 | 0.057 [0.042, 0.077] | 0.291 | [png](benchmark/Qwen--Qwen3-1.7B.sst2_choice.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | yelp | 9.219 | 6.750 | 0.535 [0.505, 0.568] | 5.464 | 0.085 [0.068, 0.121] | 1.362 | 0.170 [0.142, 0.205] | 1.403 | [png](benchmark/Qwen--Qwen3-1.7B.yelp.mireth.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | ag_news | 8.706 | 6.760 | 0.169 [0.147, 0.194] | 2.789 | 0.042 [0.032, 0.069] | 0.527 | 0.080 [0.060, 0.103] | 0.565 | [png](benchmark/Qwen--Qwen3-1.7B.ag_news.first_token.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | banking77 | 6.180 | 6.760 | 0.179 [0.159, 0.208] | 6.679 | 0.070 [0.056, 0.100] | 2.512 | 0.085 [0.067, 0.113] | 2.527 | [png](benchmark/Qwen--Qwen3-1.7B.banking77.first_token.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | sst2 | 4.652 | 6.760 | 0.124 [0.102, 0.148] | 0.806 | 0.040 [0.029, 0.064] | 0.363 | 0.089 [0.070, 0.111] | 0.390 | [png](benchmark/Qwen--Qwen3-1.7B.sst2.first_token.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | sst2_choice | 4.228 | 6.760 | 0.088 [0.072, 0.109] | 0.776 | 0.035 [0.023, 0.056] | 0.278 | 0.057 [0.043, 0.078] | 0.291 | [png](benchmark/Qwen--Qwen3-1.7B.sst2_choice.first_token.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | yelp | 9.263 | 6.760 | 0.534 [0.503, 0.567] | 5.439 | 0.077 [0.061, 0.113] | 1.361 | 0.171 [0.142, 0.204] | 1.401 | [png](benchmark/Qwen--Qwen3-1.7B.yelp.first_token.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | ag_news | 3.041 | 2.677 | 0.164 [0.133, 0.200] | 1.321 | 0.044 [0.036, 0.085] | 0.617 | 0.067 [0.044, 0.103] | 0.639 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.ag_news.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | banking77 | 2.734 | 2.677 | 0.383 [0.348, 0.421] | 4.684 | 0.088 [0.061, 0.123] | 3.222 | 0.095 [0.067, 0.127] | 3.220 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.banking77.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | sst2 | 1.847 | 2.677 | 0.082 [0.059, 0.114] | 0.369 | 0.043 [0.032, 0.073] | 0.312 | 0.081 [0.068, 0.112] | 0.339 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | sst2_choice | 2.314 | 2.677 | 0.093 [0.070, 0.121] | 0.495 | 0.034 [0.023, 0.063] | 0.290 | 0.043 [0.034, 0.071] | 0.287 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2_choice.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | yelp | 2.713 | 2.677 | 0.252 [0.215, 0.294] | 1.466 | 0.096 [0.075, 0.144] | 1.047 | 0.096 [0.074, 0.141] | 1.047 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.yelp.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | ag_news | 3.034 | 2.704 | 0.161 [0.132, 0.197] | 1.322 | 0.048 [0.035, 0.084] | 0.619 | 0.066 [0.046, 0.100] | 0.638 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.ag_news.first_token.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | banking77 | 2.860 | 2.704 | 0.131 [0.107, 0.161] | 4.653 | 0.094 [0.063, 0.123] | 3.616 | 0.085 [0.055, 0.115] | 3.616 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.banking77.first_token.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | sst2 | 1.811 | 2.704 | 0.083 [0.058, 0.113] | 0.364 | 0.052 [0.037, 0.079] | 0.310 | 0.072 [0.066, 0.108] | 0.340 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2.first_token.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | sst2_choice | 2.289 | 2.704 | 0.095 [0.070, 0.121] | 0.491 | 0.031 [0.020, 0.061] | 0.292 | 0.045 [0.033, 0.073] | 0.290 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2_choice.first_token.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | yelp | 2.728 | 2.704 | 0.247 [0.215, 0.295] | 1.474 | 0.105 [0.076, 0.152] | 1.050 | 0.110 [0.079, 0.154] | 1.050 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.yelp.first_token.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | ag_news | 5.390 | 3.830 | 0.241 [0.208, 0.279] | 2.479 | 0.087 [0.071, 0.132] | 0.832 | 0.097 [0.080, 0.140] | 0.890 | [png](benchmark/Qwen--Qwen3-0.6B.ag_news.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | banking77 | 3.380 | 3.830 | 0.388 [0.347, 0.431] | 4.825 | 0.107 [0.086, 0.157] | 2.404 | 0.145 [0.109, 0.190] | 2.435 | [png](benchmark/Qwen--Qwen3-0.6B.banking77.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | sst2 | 8.101 | 3.830 | 0.401 [0.362, 0.446] | 1.366 | 0.185 [0.158, 0.223] | 0.651 | 0.250 [0.212, 0.291] | 0.673 | [png](benchmark/Qwen--Qwen3-0.6B.sst2.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | sst2_choice | 2.797 | 3.830 | 0.112 [0.088, 0.145] | 0.581 | 0.036 [0.033, 0.077] | 0.407 | 0.072 [0.061, 0.109] | 0.427 | [png](benchmark/Qwen--Qwen3-0.6B.sst2_choice.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | yelp | 3.956 | 3.830 | 0.312 [0.273, 0.354] | 2.245 | 0.073 [0.052, 0.118] | 1.505 | 0.073 [0.053, 0.114] | 1.506 | [png](benchmark/Qwen--Qwen3-0.6B.yelp.mireth.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | ag_news | 5.357 | 3.968 | 0.235 [0.206, 0.275] | 2.469 | 0.081 [0.067, 0.126] | 0.831 | 0.096 [0.079, 0.142] | 0.878 | [png](benchmark/Qwen--Qwen3-0.6B.ag_news.first_token.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | banking77 | 3.409 | 3.968 | 0.165 [0.143, 0.205] | 4.621 | 0.072 [0.054, 0.115] | 2.830 | 0.109 [0.080, 0.148] | 2.861 | [png](benchmark/Qwen--Qwen3-0.6B.banking77.first_token.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | sst2 | 8.108 | 3.968 | 0.403 [0.366, 0.448] | 1.354 | 0.201 [0.171, 0.239] | 0.650 | 0.233 [0.206, 0.277] | 0.668 | [png](benchmark/Qwen--Qwen3-0.6B.sst2.first_token.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | sst2_choice | 2.805 | 3.968 | 0.122 [0.096, 0.154] | 0.584 | 0.030 [0.030, 0.074] | 0.409 | 0.074 [0.058, 0.110] | 0.431 | [png](benchmark/Qwen--Qwen3-0.6B.sst2_choice.first_token.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | yelp | 3.994 | 3.968 | 0.326 [0.285, 0.363] | 2.254 | 0.073 [0.051, 0.119] | 1.506 | 0.075 [0.054, 0.120] | 1.506 | [png](benchmark/Qwen--Qwen3-0.6B.yelp.first_token.png) |

## Yelp: score levels

Levels 0 to 4 (one to five stars). Argmax is the most likely level; the expected level is sum of k p_k. Normal generation is scored on its valid answers only.

| Model | Arm | n | MAE argmax | RMSE argmax | Within one level | MAE expected | RMSE expected | MAE expected at T | RMSE expected at T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.947 | 1.346 | 0.774 | 0.884 | 1.140 | 0.884 | 1.056 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.921 | 1.322 | 0.786 | 0.885 | 1.140 | 0.884 | 1.057 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 273 valid of 300 | 1.073 | 1.441 | 0.729 | n/a | n/a | n/a | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | 200 | 0.755 | 1.107 | 0.845 | 0.706 | 0.953 | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.760 | 1.057 | 0.851 | 0.767 | 1.055 | 0.806 | 0.961 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.764 | 1.059 | 0.849 | 0.767 | 1.054 | 0.805 | 0.960 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 293 valid of 300 | 0.730 | 1.002 | 0.870 | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 1000 | 0.764 | 1.074 | 0.856 | 0.759 | 1.040 | 0.909 | 1.075 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 1000 | 0.763 | 1.071 | 0.854 | 0.759 | 1.036 | 0.910 | 1.075 |
| Qwen/Qwen3-1.7B | Normal generation | 272 valid of 300 | 0.625 | 0.883 | 0.941 | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.558 | 0.884 | 0.918 | 0.527 | 0.766 | 0.543 | 0.686 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.544 | 0.872 | 0.922 | 0.527 | 0.765 | 0.544 | 0.686 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 94 valid of 100 | 0.532 | 0.838 | 0.915 | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 1.264 | 1.779 | 0.640 | 1.146 | 1.457 | 1.065 | 1.249 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 1.292 | 1.801 | 0.632 | 1.152 | 1.461 | 1.067 | 1.251 |

## SST-2: yes-bias

The same sentences as a yes/no question ("Is the sentiment positive?") and as a two-option choice. A predicted-positive rate well above the gold rate in the yes/no form only is a yes-bias.

| Model | Arm | Form | n | Accuracy | Predicted positive | Gold positive |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | yes/no | 872 | 0.884 | 0.423 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | yes/no | 872 | 0.882 | 0.421 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | yes/no | 300 | 0.653 | 0.840 | 0.500 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1, questions before state | yes/no | 200 | 0.820 | 0.330 | 0.500 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | choice | 872 | 0.924 | 0.489 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | choice | 872 | 0.925 | 0.487 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | choice | 300 | 0.903 | 0.497 | 0.500 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | yes/no | 872 | 0.899 | 0.459 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | yes/no | 872 | 0.899 | 0.459 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | yes/no | 300 | 0.250 | 0.000 | 0.500 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | choice | 872 | 0.903 | 0.524 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | choice | 872 | 0.900 | 0.524 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | choice | 300 | 0.887 | 0.517 | 0.500 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | yes/no | 872 | 0.846 | 0.612 | 0.509 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | yes/no | 872 | 0.844 | 0.615 | 0.509 |
| Qwen/Qwen3-1.7B | Normal generation | yes/no | 300 | 0.033 | 0.003 | 0.500 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | choice | 872 | 0.904 | 0.528 | 0.509 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | choice | 872 | 0.903 | 0.531 | 0.509 |
| Qwen/Qwen3-1.7B | Normal generation | choice | 300 | 0.847 | 0.513 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | yes/no | 500 | 0.856 | 0.392 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | yes/no | 500 | 0.866 | 0.410 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | yes/no | 100 | 0.480 | 0.230 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | choice | 500 | 0.898 | 0.446 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | choice | 500 | 0.900 | 0.444 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | choice | 100 | 0.840 | 0.490 | 0.500 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | yes/no | 500 | 0.530 | 0.970 | 0.500 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | yes/no | 500 | 0.526 | 0.974 | 0.500 |
| Qwen/Qwen3-0.6B | Normal generation | yes/no | 100 | 0.000 | 0.000 | 0.500 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | choice | 500 | 0.822 | 0.602 | 0.500 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | choice | 500 | 0.822 | 0.598 | 0.500 |
| Qwen/Qwen3-0.6B | Normal generation | choice | 100 | 0.040 | 0.000 | 0.500 |

## First-token ties

The first-token arm scores only each label's first token, so options whose first tokens are the same token get the same score. A tie goes to the earlier option in the request (as the original demo's argmax does), and the tied options split the probability. Labels in ties: mean per row of options sharing their first token with another option.

| Model | Dataset | n | Rows with a tie | Labels in ties | Predictions decided by the tie rule | Accuracy of those |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | ag_news | 1000 | 38 | 0.1 | 2 | 0.000 |
| Qwen/Qwen2.5-1.5B-Instruct | banking77 | 1000 | 1000 | 55.4 | 637 | 0.149 |
| Qwen/Qwen2.5-1.5B-Instruct | sst2 | 872 | 5 | 0.0 | 5 | 0.600 |
| Qwen/Qwen2.5-1.5B-Instruct | sst2_choice | 872 | 1 | 0.0 | 1 | 1.000 |
| Qwen/Qwen2.5-1.5B-Instruct | yelp | 1000 | 151 | 0.3 | 42 | 0.333 |
| Qwen/Qwen2.5-1.5B-Instruct | jevbench_easy | 48 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | jevbench_standard | 72 | 13 | 0.4 | 9 | 0.222 |
| Qwen/Qwen2.5-1.5B-Instruct | jevbench_hard | 111 | 31 | 0.8 | 20 | 0.150 |
| Qwen/Qwen3-4B-Instruct-2507 | ag_news | 1000 | 21 | 0.0 | 2 | 0.500 |
| Qwen/Qwen3-4B-Instruct-2507 | banking77 | 1000 | 1000 | 52.0 | 631 | 0.200 |
| Qwen/Qwen3-4B-Instruct-2507 | sst2 | 872 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | sst2_choice | 872 | 1 | 0.0 | 1 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | yelp | 1000 | 29 | 0.1 | 7 | 0.429 |
| Qwen/Qwen3-4B-Instruct-2507 | jevbench_easy | 48 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | jevbench_standard | 72 | 13 | 0.4 | 10 | 0.200 |
| Qwen/Qwen3-4B-Instruct-2507 | jevbench_hard | 111 | 27 | 0.7 | 15 | 0.200 |
| Qwen/Qwen3-1.7B | ag_news | 1000 | 16 | 0.0 | 1 | 0.000 |
| Qwen/Qwen3-1.7B | banking77 | 1000 | 1000 | 53.3 | 670 | 0.169 |
| Qwen/Qwen3-1.7B | sst2 | 872 | 6 | 0.0 | 6 | 0.000 |
| Qwen/Qwen3-1.7B | sst2_choice | 872 | 1 | 0.0 | 1 | 0.000 |
| Qwen/Qwen3-1.7B | yelp | 1000 | 82 | 0.2 | 11 | 0.273 |
| Qwen/Qwen3-1.7B | jevbench_easy | 48 | 1 | 0.0 | 0 | n/a |
| Qwen/Qwen3-1.7B | jevbench_standard | 72 | 13 | 0.4 | 13 | 0.154 |
| Qwen/Qwen3-1.7B | jevbench_hard | 111 | 31 | 0.8 | 21 | 0.143 |
| HuggingFaceTB/SmolLM3-3B | ag_news | 500 | 16 | 0.1 | 0 | n/a |
| HuggingFaceTB/SmolLM3-3B | banking77 | 500 | 500 | 57.0 | 243 | 0.111 |
| HuggingFaceTB/SmolLM3-3B | sst2 | 500 | 7 | 0.0 | 7 | 0.857 |
| HuggingFaceTB/SmolLM3-3B | sst2_choice | 500 | 1 | 0.0 | 1 | 1.000 |
| HuggingFaceTB/SmolLM3-3B | yelp | 500 | 60 | 0.2 | 19 | 0.421 |
| HuggingFaceTB/SmolLM3-3B | jevbench_easy | 48 | 2 | 0.1 | 0 | n/a |
| HuggingFaceTB/SmolLM3-3B | jevbench_standard | 72 | 14 | 0.4 | 10 | 0.200 |
| HuggingFaceTB/SmolLM3-3B | jevbench_hard | 111 | 30 | 0.7 | 18 | 0.222 |
| Qwen/Qwen3-0.6B | ag_news | 500 | 13 | 0.1 | 2 | 0.500 |
| Qwen/Qwen3-0.6B | banking77 | 500 | 500 | 53.8 | 326 | 0.138 |
| Qwen/Qwen3-0.6B | sst2 | 500 | 2 | 0.0 | 2 | 0.000 |
| Qwen/Qwen3-0.6B | sst2_choice | 500 | 2 | 0.0 | 2 | 0.500 |
| Qwen/Qwen3-0.6B | yelp | 500 | 87 | 0.4 | 19 | 0.211 |
| Qwen/Qwen3-0.6B | jevbench_easy | 48 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-0.6B | jevbench_standard | 72 | 15 | 0.5 | 11 | 0.182 |
| Qwen/Qwen3-0.6B | jevbench_hard | 111 | 29 | 0.7 | 19 | 0.105 |

## Question order (SPEC 3.1)

The state before the questions (the shipped prompt) against the questions before the state (the order that allowed a question-part cache, dropped for accuracy), same rows, raw scores.

| Model | Dataset | n | Accuracy, state first | Accuracy, questions first | ECE-15, state first | ECE-15, questions first | Same answer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | ag_news | 200 | 0.790 [0.735, 0.845] | 0.485 [0.420, 0.555] | 0.181 | 0.198 | 0.580 |
| Qwen/Qwen2.5-1.5B-Instruct | sst2 | 200 | 0.915 [0.875, 0.950] | 0.820 [0.765, 0.870] | 0.049 | 0.058 | 0.885 |
| Qwen/Qwen2.5-1.5B-Instruct | yelp | 200 | 0.385 [0.320, 0.450] | 0.435 [0.365, 0.500] | 0.369 | 0.314 | 0.675 |

## JevBench public items

The 231 public items of JevBench (github.com/fstandhartinger/jevbench, MIT), fetched at run time, scored here by us. Not a JevBench score: its board adds sealed items, a speed and a cost axis and runs entrants on its own hardware. Chance: mean of 1/options over the items run; chance-corrected accuracy is 100 (accuracy - chance) / (1 - chance), floored at 0, as JevBench's Intelligence axis per tier. ECE-10 matches JevBench's binning. The pooled T comes from our own calibration splits, never from JevBench items.

| Model | Arm | Tier | n | Accuracy | Chance | Chance-corrected | ECE-10 at T = 1 | ECE-10 at T pooled | ECE-15 at T = 1 | NLL | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | easy | 48 | 0.958 [0.896, 1.000] | 0.284 | 94.2 | 0.041 | 0.116 (T = 2.29) | 0.046 | 0.100 | 0.064 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | standard | 72 | 0.611 [0.500, 0.722] | 0.311 | 43.5 | 0.217 | 0.051 (T = 2.29) | 0.217 | 1.114 | 0.558 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | hard | 111 | 0.351 [0.261, 0.441] | 0.336 | 2.3 | 0.405 | 0.259 (T = 2.29) | 0.443 | 2.090 | 0.985 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | easy | 48 | 0.958 [0.896, 1.000] | 0.284 | 94.2 | 0.043 | 0.127 (T = 2.34) | 0.047 | 0.104 | 0.065 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | standard | 72 | 0.597 [0.486, 0.708] | 0.311 | 41.5 | 0.186 | 0.087 (T = 2.34) | 0.195 | 1.213 | 0.586 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | hard | 111 | 0.333 [0.243, 0.423] | 0.336 | 0.0 | 0.380 | 0.254 (T = 2.34) | 0.401 | 2.009 | 0.956 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | easy | 48 | 0.479 [0.333, 0.625] | 0.284 | 27.2 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | standard | 72 | 0.431 [0.319, 0.542] | 0.311 | 17.3 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | hard | 111 | 0.234 [0.162, 0.315] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.000 | 0.051 (T = 8.04) | 0.000 | 0.000 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | standard | 72 | 0.750 [0.653, 0.847] | 0.311 | 63.7 | 0.243 | 0.140 (T = 8.04) | 0.243 | 4.455 | 0.484 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | hard | 111 | 0.477 [0.387, 0.568] | 0.336 | 21.3 | 0.511 | 0.338 (T = 8.04) | 0.511 | 9.526 | 1.029 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.000 | 0.063 (T = 8.55) | 0.000 | 0.000 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | standard | 72 | 0.750 [0.639, 0.847] | 0.311 | 63.7 | 0.176 | 0.097 (T = 8.55) | 0.176 | 4.275 | 0.424 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | hard | 111 | 0.450 [0.360, 0.541] | 0.336 | 17.2 | 0.461 | 0.337 (T = 8.55) | 0.461 | 8.890 | 0.978 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | easy | 48 | 0.771 [0.646, 0.876] | 0.284 | 68.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | standard | 72 | 0.667 [0.556, 0.778] | 0.311 | 51.6 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | hard | 111 | 0.351 [0.270, 0.450] | 0.336 | 2.3 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | easy | 48 | 0.958 [0.896, 1.000] | 0.284 | 94.2 | 0.031 | 0.076 (T = 6.75) | 0.031 | 0.066 | 0.047 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | standard | 72 | 0.625 [0.514, 0.736] | 0.311 | 45.6 | 0.290 | 0.103 (T = 6.75) | 0.290 | 2.248 | 0.589 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | hard | 111 | 0.315 [0.225, 0.405] | 0.336 | 0.0 | 0.620 | 0.348 (T = 6.75) | 0.629 | 6.999 | 1.273 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | easy | 48 | 0.979 [0.938, 1.000] | 0.284 | 97.1 | 0.022 | 0.056 (T = 6.76) | 0.022 | 0.033 | 0.024 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | standard | 72 | 0.583 [0.458, 0.681] | 0.311 | 39.5 | 0.271 | 0.129 (T = 6.76) | 0.270 | 2.390 | 0.597 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | hard | 111 | 0.315 [0.225, 0.405] | 0.336 | 0.0 | 0.526 | 0.304 (T = 6.76) | 0.530 | 6.326 | 1.144 |
| Qwen/Qwen3-1.7B | Normal generation | easy | 48 | 0.750 [0.625, 0.875] | 0.284 | 65.1 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | Normal generation | standard | 72 | 0.458 [0.333, 0.569] | 0.311 | 21.4 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | Normal generation | hard | 111 | 0.153 [0.090, 0.225] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.002 | 0.066 (T = 2.68) | 0.002 | 0.002 | 0.000 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | standard | 72 | 0.653 [0.542, 0.764] | 0.311 | 49.6 | 0.218 | 0.071 (T = 2.68) | 0.227 | 1.540 | 0.559 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | hard | 111 | 0.387 [0.297, 0.477] | 0.336 | 7.7 | 0.408 | 0.234 (T = 2.68) | 0.408 | 2.109 | 0.924 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.002 | 0.071 (T = 2.70) | 0.002 | 0.002 | 0.000 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | standard | 72 | 0.625 [0.514, 0.736] | 0.311 | 45.6 | 0.204 | 0.077 (T = 2.70) | 0.186 | 1.626 | 0.552 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | hard | 111 | 0.387 [0.297, 0.477] | 0.336 | 7.7 | 0.360 | 0.207 (T = 2.70) | 0.360 | 2.083 | 0.894 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | easy | 48 | 0.938 [0.875, 1.000] | 0.284 | 91.3 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | Normal generation | standard | 72 | 0.667 [0.556, 0.778] | 0.311 | 51.6 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | Normal generation | hard | 111 | 0.207 [0.135, 0.288] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | easy | 48 | 0.854 [0.750, 0.938] | 0.284 | 79.6 | 0.139 | 0.129 (T = 3.83) | 0.139 | 0.552 | 0.260 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | standard | 72 | 0.486 [0.375, 0.597] | 0.311 | 25.4 | 0.413 | 0.181 (T = 3.83) | 0.430 | 2.760 | 0.864 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | hard | 111 | 0.351 [0.270, 0.441] | 0.336 | 2.3 | 0.526 | 0.301 (T = 3.83) | 0.534 | 3.983 | 1.121 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | easy | 48 | 0.854 [0.750, 0.938] | 0.284 | 79.6 | 0.134 | 0.134 (T = 3.97) | 0.134 | 0.557 | 0.260 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | standard | 72 | 0.486 [0.375, 0.597] | 0.311 | 25.4 | 0.350 | 0.141 (T = 3.97) | 0.369 | 2.778 | 0.822 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | hard | 111 | 0.333 [0.243, 0.423] | 0.336 | 0.0 | 0.483 | 0.278 (T = 3.97) | 0.488 | 3.784 | 1.071 |
| Qwen/Qwen3-0.6B | Normal generation | easy | 48 | 0.000 [0.000, 0.000] | 0.284 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | Normal generation | standard | 72 | 0.000 [0.000, 0.000] | 0.311 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | Normal generation | hard | 111 | 0.000 [0.000, 0.000] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |

## Latency

Batch 1, one request at a time, model loaded, after the warmup runs; each timed span is one `Engine.decide` or `baseline.generate` call with torch.cuda.synchronize() at both ends on CUDA. p50 and p95 with linear interpolation; brackets: bootstrap 95% CI. With few runs p95 rests on the slowest one or two runs. Field-count settings use a different AG News article per run (topic choice plus rule-checked yes/no questions); scenarios repeat their own text. Checked answers: share of fields whose answer matches the gold topic or the rule. Bad fields: hallucinated or missing answers per generation run.

### Qwen/Qwen2.5-1.5B-Instruct

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi at start: ['NVIDIA GeForce RTX 5070, 591.86, 40, 2655 MHz, 13801 MHz, 24.10 W']; at end: ['NVIDIA GeForce RTX 5070, 591.86, 45, 2925 MHz, 13801 MHz, 72.20 W'].

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 30 | 37.2 [36.3, 39.9] | 63.8 [47.5, 77.3] | 370 [338, 388] | 754 [455, 961] | 9.9x | 226 | 9 | 0.73 / 0.70 | 0.1 | 2985 |
| 5 fields | 5 | 30 | 36.5 [35.9, 44.8] | 66.6 [55.4, 73.0] | 1470 [1430, 1557] | 1801 [1654, 1982] | 40.3x | 333 | 32 | 0.71 / 0.61 | 0.0 | 2995 |
| 10 fields | 10 | 30 | 37.1 [36.8, 39.9] | 56.0 [52.4, 57.3] | 2911 [2795, 3037] | 3400 [3217, 3494] | 78.5x | 468 | 63 | 0.62 / 0.72 | 0.0 | 3005 |
| 20 fields | 20 | 30 | 50.6 [50.4, 51.2] | 55.3 [53.1, 55.8] | 6274 [6056, 6549] | 6876 [6688, 6898] | 124.0x | 777 | 134 | 0.50 / 0.66 | 0.6 | 3034 |
| Support ticket triage (28 fields) | 28 | 20 | 98.1 [97.8, 98.3] | 100 [98.7, 101] | 9704 [9530, 9945] | 10156 [10009, 10543] | 99.0x | 1741 | 206 | n/a / n/a | 0.0 | 3168 |
| Code change security review (28 fields) | 28 | 20 | 99.3 [99.1, 99.4] | 101 [99.5, 101] | 10249 [10078, 10397] | 10571 [10416, 10841] | 103.2x | 1822 | 216 | n/a / n/a | 0.0 | 3193 |
| Incident triage with scores (20 fields) | 20 | 20 | 83.3 [83.0, 84.9] | 85.9 [85.0, 87.8] | 6864 [6677, 7076] | 7402 [7201, 7439] | 82.4x | 1488 | 149 | n/a / n/a | 0.0 | 3122 |
| Request router, 255 queues (4 fields) | 4 | 20 | 220 [220, 221] | 224 [222, 227] | 1649 [1567, 1730] | 1899 [1749, 2050] | 7.5x | 4388 | 30 | n/a / n/a | 2.0 | 3586 |

### Qwen/Qwen3-4B-Instruct-2507

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi at start: ['NVIDIA GeForce RTX 5070, 591.86, 43, 2677 MHz, 13801 MHz, 34.38 W']; at end: ['NVIDIA GeForce RTX 5070, 591.86, 52, 2917 MHz, 13801 MHz, 96.19 W'].

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 30 | 58.7 [58.2, 62.7] | 84.2 [68.0, 88.8] | 571 [529, 633] | 734 [677, 913] | 9.7x | 226 | 8 | 0.83 / 0.83 | 0.0 | 7831 |
| 5 fields | 5 | 30 | 72.0 [68.5, 75.2] | 83.4 [81.5, 85.1] | 2349 [2270, 2400] | 2699 [2538, 2730] | 32.6x | 333 | 32 | 0.65 / 0.61 | 0.0 | 7857 |
| 10 fields | 10 | 30 | 83.5 [83.4, 84.6] | 89.7 [85.1, 95.7] | 4549 [4390, 4629] | 5128 [4808, 5469] | 54.5x | 468 | 63 | 0.74 / 0.70 | 0.0 | 7888 |
| 20 fields | 20 | 30 | 121 [120, 122] | 124 [122, 125] | 9844 [9570, 9975] | 10632 [10300, 11042] | 81.6x | 777 | 133 | 0.71 / 0.68 | 0.0 | 7988 |
| Support ticket triage (28 fields) | 28 | 20 | 274 [274, 274] | 276 [275, 276] | 17163 [16998, 17416] | 17881 [17505, 17988] | 62.7x | 1741 | 236 | n/a / n/a | 0.0 | 8365 |
| Code change security review (28 fields) | 28 | 20 | 279 [279, 279] | 282 [280, 282] | 18134 [17812, 18259] | 18803 [18401, 18984] | 64.9x | 1822 | 248 | n/a / n/a | 1.0 | 8377 |
| Incident triage with scores (20 fields) | 20 | 20 | 223 [223, 223] | 224 [223, 225] | 11001 [10820, 11103] | 11767 [11191, 11892] | 49.4x | 1488 | 149 | n/a / n/a | 0.0 | 8261 |
| Request router, 255 queues (4 fields) | 4 | 20 | 637 [636, 637] | 639 [637, 639] | 2862 [2772, 2941] | 3078 [2957, 3154] | 4.5x | 4388 | 29 | n/a / n/a | 0.0 | 9408 |

### Qwen/Qwen3-1.7B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi at start: ['NVIDIA GeForce RTX 5070, 591.86, 39, 2715 MHz, 13801 MHz, 18.67 W']; at end: ['NVIDIA GeForce RTX 5070, 591.86, 47, 2925 MHz, 13801 MHz, 68.23 W'].

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 30 | 48.5 [46.3, 54.5] | 98.5 [63.3, 105] | 891 [840, 1012] | 1315 [1073, 1350] | 18.4x | 230 | 16 | 0.77 / 0.73 | 0.0 | 3355 |
| 5 fields | 5 | 30 | 44.7 [44.2, 46.7] | 68.9 [58.6, 83.0] | 1873 [1804, 1956] | 2282 [2004, 2338] | 41.9x | 337 | 32 | 0.78 / 0.74 | 0.0 | 3371 |
| 10 fields | 10 | 30 | 44.0 [43.6, 46.9] | 75.2 [54.1, 84.0] | 3662 [3527, 3759] | 4117 [3985, 4159] | 83.3x | 472 | 63 | 0.75 / 0.78 | 0.0 | 3389 |
| 20 fields | 20 | 30 | 52.6 [52.3, 53.1] | 72.4 [66.6, 73.9] | 7517 [7348, 7675] | 8123 [7860, 8360] | 142.9x | 781 | 134 | 0.57 / 0.71 | 0.6 | 3444 |
| Support ticket triage (28 fields) | 28 | 20 | 110 [110, 111] | 113 [111, 114] | 11827 [11650, 11984] | 12464 [12108, 13036] | 107.3x | 1745 | 206 | n/a / n/a | 0.0 | 3663 |
| Code change security review (28 fields) | 28 | 20 | 113 [112, 114] | 116 [114, 116] | 12367 [12012, 12562] | 13427 [12662, 13938] | 109.8x | 1826 | 219 | n/a / n/a | 0.0 | 3669 |
| Incident triage with scores (20 fields) | 20 | 20 | 95.3 [95.1, 95.5] | 97.5 [96.1, 97.6] | 8566 [8389, 8763] | 9303 [8814, 9335] | 89.9x | 1492 | 149 | n/a / n/a | 0.0 | 3593 |
| Request router, 255 queues (4 fields) | 4 | 20 | 258 [257, 258] | 262 [259, 262] | 1911 [1865, 1972] | 2147 [1976, 2535] | 7.4x | 4392 | 28 | n/a / n/a | 0.0 | 4236 |

### HuggingFaceTB/SmolLM3-3B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi at start: ['NVIDIA GeForce RTX 5070, 591.86, 42, 2662 MHz, 13801 MHz, 30.39 W']; at end: ['NVIDIA GeForce RTX 5070, 591.86, 51, 2917 MHz, 13801 MHz, 89.29 W'].

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 10 | 48.4 [43.1, 57.9] | 64.0 [52.0, 64.1] | 444 [405, 604] | 647 [540, 663] | 9.2x | 273 | 8 | 0.80 / 0.70 | 0.0 | 6041 |
| 5 fields | 5 | 10 | 52.5 [51.7, 62.3] | 65.4 [52.9, 66.0] | 1741 [1689, 1905] | 2031 [1810, 2089] | 33.2x | 378 | 32 | 0.60 / 0.68 | 0.0 | 6054 |
| 10 fields | 10 | 10 | 64.9 [64.5, 67.0] | 74.0 [65.9, 79.5] | 3287 [3205, 3792] | 3962 [3558, 4008] | 50.7x | 510 | 62 | 0.73 / 0.83 | 0.0 | 6071 |
| 20 fields | 20 | 10 | 83.8 [83.5, 90.4] | 106 [84.2, 117] | 6798 [6496, 7069] | 7091 [7031, 7107] | 81.1x | 784 | 122 | 0.67 / 0.77 | 0.0 | 6114 |
| Support ticket triage (28 fields) | 28 | 20 | 184 [184, 185] | 187 [186, 187] | 12887 [12644, 13334] | 13675 [13486, 13704] | 70.0x | 1696 | 216 | n/a / n/a | 0.0 | 6309 |
| Code change security review (28 fields) | 28 | 20 | 186 [186, 187] | 189 [187, 190] | 13474 [13320, 13789] | 14311 [13921, 14361] | 72.3x | 1783 | 238 | n/a / n/a | 1.0 | 6318 |
| Incident triage with scores (20 fields) | 20 | 20 | 154 [154, 154] | 157 [155, 159] | 9329 [9142, 9617] | 9995 [9647, 10186] | 60.5x | 1462 | 162 | n/a / n/a | 0.0 | 6243 |
| Request router, 255 queues (4 fields) | 4 | 20 | 409 [409, 409] | 413 [410, 413] | 2214 [2103, 2336] | 2525 [2371, 2541] | 5.4x | 4420 | 29 | n/a / n/a | 0.0 | 6859 |

### Qwen/Qwen3-0.6B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi at start: ['NVIDIA GeForce RTX 5070, 591.86, 52, 2722 MHz, 13801 MHz, 23.24 W']; at end: ['NVIDIA GeForce RTX 5070, 591.86, 44, 2925 MHz, 13801 MHz, 66.43 W'].

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 10 | 45.5 [44.9, 57.0] | 91.5 [48.3, 110] | 1305 [1184, 1391] | 1553 [1350, 1645] | 28.7x | 241 | 22 | 0.70 / 0.00 | 1.0 | 1209 |
| 5 fields | 5 | 10 | 45.7 [44.6, 68.8] | 75.4 [59.0, 80.2] | 2140 [1985, 2435] | 2518 [2217, 2525] | 46.8x | 348 | 40 | 0.42 / 0.44 | 0.0 | 1222 |
| 10 fields | 10 | 10 | 44.0 [43.6, 53.5] | 83.0 [44.2, 99.4] | 4075 [3704, 4163] | 4284 [4133, 4358] | 92.5x | 483 | 81 | 0.50 / 0.40 | 2.0 | 1243 |
| 20 fields | 20 | 10 | 44.1 [43.2, 52.4] | 78.9 [45.0, 94.9] | 7791 [7628, 8111] | 8674 [7979, 9080] | 176.6x | 792 | 156 | 0.47 / 0.49 | 1.9 | 1295 |
| Support ticket triage (28 fields) | 28 | 20 | 64.6 [64.5, 64.8] | 67.6 [65.0, 68.2] | 15014 [14898, 15951] | 17217 [16124, 17971] | 232.4x | 1745 | 275 | n/a / n/a | 17.0 | 1511 |
| Code change security review (28 fields) | 28 | 20 | 66.4 [66.0, 68.6] | 75.2 [69.9, 76.1] | 12331 [11477, 13527] | 14234 [14063, 14571] | 185.6x | 1826 | 219 | n/a / n/a | 20.0 | 1517 |
| Incident triage with scores (20 fields) | 20 | 20 | 54.8 [54.6, 55.4] | 72.0 [55.6, 72.9] | 9381 [9092, 9752] | 10724 [9887, 10928] | 171.1x | 1492 | 178 | n/a / n/a | 20.0 | 1443 |
| Request router, 255 queues (4 fields) | 4 | 20 | 163 [163, 163] | 164 [163, 164] | 2136 [2044, 2284] | 2733 [2333, 2991] | 13.1x | 4392 | 37 | n/a / n/a | 2.0 | 2078 |

<!-- precision:start -->
## Precision: bf16, fp16 and fp32 (Qwen2.5-1.5B-Instruct)

The MirethSTM1 arm in three dtypes on the same 500 evaluation rows per dataset, from bench/out/full (bf16), bench/out/full-fp16 and bench/out/full-fp32. Each dtype fitted its own pooled T on the same 500 calibration rows per dataset: bf16 2.285, fp16 2.289, fp32 2.291. Agreement and score differences are against fp32 on the same rows, over every label's summed log-prob. Median ms is the per-row time of the accuracy run (one question per call).

| Dataset | n | Accuracy bf16 / fp16 / fp32 | ECE-15 at T = 1 | ECE-15 at own pooled T | Same answer as fp32: bf16 / fp16 | Mean (max) abs score difference from fp32: bf16 / fp16 | Median ms bf16 / fp16 / fp32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AG News | 500 | 0.826 / 0.824 / 0.824 | 0.143 / 0.143 / 0.146 | 0.082 / 0.081 / 0.080 | 0.994 / 1.000 | 0.193 (1.26) / 0.029 (0.26) | 36.8 / 36.7 / 57.2 |
| Banking77 | 500 | 0.550 / 0.542 / 0.542 | 0.240 / 0.249 / 0.249 | 0.132 / 0.124 / 0.115 | 0.966 / 1.000 | 0.179 (1.26) / 0.024 (0.23) | 80.9 / 72.6 / 240 |
| SST-2 yes/no | 500 | 0.884 / 0.886 / 0.886 | 0.048 / 0.052 / 0.053 | 0.089 / 0.090 / 0.090 | 0.994 / 1.000 | 0.065 (0.44) / 0.008 (0.06) | 38.5 / 36.5 / 51.2 |
| SST-2 choice | 500 | 0.920 / 0.922 / 0.922 | 0.066 / 0.069 / 0.069 | 0.020 / 0.027 / 0.029 | 0.998 / 1.000 | 0.189 (0.92) / 0.022 (0.09) | 37.6 / 36.7 / 50.9 |
| Yelp | 500 | 0.374 / 0.374 / 0.372 | 0.365 / 0.364 / 0.366 | 0.198 / 0.199 / 0.195 | 0.978 / 0.996 | 0.063 (0.54) / 0.009 (0.13) | 38.2 / 37.1 / 78.2 |

Against fp32, bf16 moves accuracy by at most 0.008 and ECE-15 at the pooled T by at most 0.017, and the pooled T by 0.005, all well inside the bootstrap intervals in the Accuracy and Calibration sections. fp16 is as fast as bf16 and 7 to 9 times closer to fp32 in score; fp32 is 1.3x to 3.0x slower than bf16 per row.

<!-- precision:end -->
