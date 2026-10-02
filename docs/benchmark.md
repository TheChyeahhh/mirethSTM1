# MirethSTM1 benchmark: v2

Generated 2026-10-02 15:32 UTC by `python -m bench.report` from bench/out/v2/ (per-sample JSONL). Models: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-1.7B, HuggingFaceTB/SmolLM3-3B, Qwen/Qwen3-0.6B.

Accuracy runs: seed 0; NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, transformers 5.18.0. Samples are stratified by class, and a smaller sample is a subset of a larger one (same seed). Texts longer than 2000 characters are cut at a word boundary (only Yelp).

- Qwen/Qwen2.5-1.5B-Instruct: bf16, 2026-10-02; scoring arms on 1000 (sst2: 872, sst2_choice: 872) evaluation rows per dataset and 500 calibration rows; normal generation on 300 evaluation rows (files copied in from an earlier run: meta.json records no run of that arm); latency over 30 timed runs per field count and 20 timed runs per scenario.
- Qwen/Qwen3-4B-Instruct-2507: bf16, 2026-10-02; scoring arms on 1000 (sst2: 872, sst2_choice: 872) evaluation rows per dataset and 500 calibration rows; normal generation on 300 evaluation rows (files copied in from an earlier run: meta.json records no run of that arm); latency over 30 timed runs per field count and 20 timed runs per scenario.
- Qwen/Qwen3-1.7B: bf16, 2026-10-02; scoring arms on 500 evaluation rows per dataset and 300 calibration rows; normal generation on 300 evaluation rows (files copied in from an earlier run: meta.json records no run of that arm); latency over 10 timed runs per field count and 20 timed runs per scenario.
- HuggingFaceTB/SmolLM3-3B: bf16, 2026-10-02; scoring arms on 500 evaluation rows per dataset and 300 calibration rows; normal generation on 100 evaluation rows (files copied in from an earlier run: meta.json records no run of that arm); latency over 10 timed runs per field count and 20 timed runs per scenario.
- Qwen/Qwen3-0.6B: bf16, 2026-10-02; scoring arms on 500 evaluation rows per dataset and 300 calibration rows; normal generation on 100 evaluation rows (files copied in from an earlier run: meta.json records no run of that arm); latency over 10 timed runs per field count and 20 timed runs per scenario.

## Summary across models

Means over the datasets (ag_news, banking77, sst2, sst2_choice, yelp), each weighted equally, evaluation split. Rows: the largest per-dataset row count of the scoring arms and of normal generation; models run on different row counts are compared on their shared rows in the second table. MirethSTM1 on the generation rows scores MirethSTM1 on exactly the rows normal generation ran. Normal generation counts an invalid, hallucinated or missing answer as wrong and is strict about types (a quoted "true" is not a boolean). ECE-15 is the mean of the per-dataset values; at the pooled T it is computed at T rounded to three decimals, the value that ships in `mirethstm.calibration.DEFAULT_TEMPERATURES`.

| Model | Rows: scoring / generation | MirethSTM1 accuracy | First-token accuracy | Normal generation accuracy (valid answers) | MirethSTM1 on the generation rows | MirethSTM1 ECE-15 at T = 1 | MirethSTM1 ECE-15 at pooled T | Pooled T | First token ECE-15 at T = 1 / at its pooled T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 1000 / 300 | 0.728 | 0.694 | 0.555 (0.850) | 0.722 | 0.148 | 0.109 | 2.112 | 0.119 / 0.108 |
| Qwen/Qwen3-4B-Instruct-2507 | 1000 / 300 | 0.760 | 0.702 | 0.611 (0.849) | 0.747 | 0.230 | 0.117 | 7.930 | 0.200 / 0.112 |
| Qwen/Qwen3-1.7B | 500 / 300 | 0.726 | 0.690 | 0.515 (0.758) | 0.727 | 0.238 | 0.097 | 7.007 | 0.195 / 0.097 |
| HuggingFaceTB/SmolLM3-3B | 500 / 100 | 0.688 | 0.671 | 0.572 (0.846) | 0.702 | 0.190 | 0.061 | 2.702 | 0.147 / 0.060 |
| Qwen/Qwen3-0.6B | 500 / 100 | 0.575 | 0.532 | 0.008 (0.008) | 0.586 | 0.339 | 0.137 | 4.639 | 0.296 / 0.135 |

Accuracy on the evaluation rows that every model ran (ag_news: 500, banking77: 500, sst2: 500, sst2_choice: 500, yelp: 500), MirethSTM1 / first token:

| Model | ag_news | banking77 | sst2 | sst2_choice | yelp | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 0.840 / 0.834 | 0.498 / 0.326 | 0.902 / 0.908 | 0.920 / 0.920 | 0.484 / 0.482 | 0.729 / 0.694 |
| Qwen/Qwen3-4B-Instruct-2507 | 0.874 / 0.874 | 0.672 / 0.388 | 0.872 / 0.874 | 0.900 / 0.898 | 0.438 / 0.436 | 0.751 / 0.694 |
| Qwen/Qwen3-1.7B | 0.810 / 0.806 | 0.500 / 0.326 | 0.876 / 0.880 | 0.900 / 0.898 | 0.546 / 0.542 | 0.726 / 0.690 |
| HuggingFaceTB/SmolLM3-3B | 0.838 / 0.840 | 0.300 / 0.210 | 0.818 / 0.820 | 0.898 / 0.898 | 0.584 / 0.588 | 0.688 / 0.671 |
| Qwen/Qwen3-0.6B | 0.746 / 0.740 | 0.464 / 0.270 | 0.542 / 0.540 | 0.848 / 0.844 | 0.276 / 0.268 | 0.575 / 0.532 |

MirethSTM1 p50 latency in ms (warm, batch 1; full tables in Latency below). Peak memory: the largest `torch.cuda.max_memory_allocated` of any MirethSTM1 run of the model.

| Model | 1 field | 5 fields | 10 fields | 20 fields | Support ticket triage (28 fields) | Code change security review (28 fields) | Incident triage with scores (20 fields) | Request router, 255 queues (4 fields) | Peak memory MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 37.2 | 35.7 | 34.9 | 48.1 | 96.4 | 97.7 | 81.4 | 209 | 3550 |
| Qwen/Qwen3-4B-Instruct-2507 | 54.6 | 67.4 | 76.0 | 114 | 271 | 279 | 211 | 615 | 8336 |
| Qwen/Qwen3-1.7B | 44.8 | 43.5 | 42.8 | 52.7 | 111 | 132 | 95.2 | 248 | 3889 |
| HuggingFaceTB/SmolLM3-3B | 41.5 | 51.6 | 64.5 | 93.2 | 213 | 215 | 167 | 408 | 6493 |
| Qwen/Qwen3-0.6B | 43.8 | 42.9 | 42.9 | 42.8 | 63.7 | 76.2 | 53.4 | 156 | 1739 |

Normal generation p50 latency in ms on the same settings: the same loaded model writing the answers as one JSON object with Hugging Face `generate` (greedy). Tokens per second: generated tokens over wall time, summed over every timed run (the prompt pass included). Every speedup in this report is against this decoding speed.

| Model | 1 field | 5 fields | 10 fields | 20 fields | Support ticket triage (28 fields) | Code change security review (28 fields) | Incident triage with scores (20 fields) | Request router, 255 queues (4 fields) | Generated tokens per second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 310 | 1269 | 2548 | 5436 | 8214 | 8796 | 6122 | 1461 | 24.0 |
| Qwen/Qwen3-4B-Instruct-2507 | 486 | 2056 | 3915 | 8651 | 16863 | 16990 | 11175 | 2698 | 14.4 |
| Qwen/Qwen3-1.7B | 712 | 1528 | 3016 | 6449 | 10247 | 11150 | 7613 | 1659 | 19.4 |
| HuggingFaceTB/SmolLM3-3B | 372 | 1551 | 3069 | 6135 | 11677 | 12914 | 8709 | 2430 | 18.3 |
| Qwen/Qwen3-0.6B | 1049 | 1988 | 3979 | 7601 | 13797 | 10894 | 8693 | 2018 | 19.9 |

## Accuracy on the evaluation split

Raw scores (T = 1). Accuracy and macro-F1 do not depend on T (the argmax does not move), so they are given once. Normal generation has no probabilities: an invalid, hallucinated or missing answer counts wrong, and NLL, Brier and ECE do not apply. Salvaged accuracy (secondary): the first allowed value written for the question's key anywhere in the output, even when the JSON is invalid or cut off. Brackets: bootstrap 95% CI (1000 resamples of the evaluation rows). ECE is biased upward on small samples, and more so on resamples (they repeat rows), so its interval can sit above the point estimate. NLL clips the gold label's probability at 1e-12 (27.6 per row), which lowers the T = 1 values of the Qwen3 models.

### ag_news (fancyzhx/ag_news test, 4 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.846 [0.823, 0.868] | 0.846 [0.823, 0.867] | 0.832 [0.694, 0.991] | 0.268 [0.231, 0.309] | 0.116 [0.097, 0.139] | 0.111 | n/a | n/a | n/a | 35.9 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.845 [0.822, 0.867] | 0.845 [0.823, 0.866] | 0.831 [0.694, 0.991] | 0.268 [0.231, 0.309] | 0.112 [0.095, 0.137] | 0.111 | n/a | n/a | n/a | 80.2 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.787 [0.740, 0.830] | 0.794 [0.750, 0.837] | n/a | n/a | n/a | n/a | 0.973 | 0.787 | 0.023 | 386 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.890 [0.870, 0.908] | 0.890 [0.870, 0.908] | 2.438 [2.000, 2.918] | 0.220 [0.183, 0.260] | 0.110 [0.092, 0.130] | 0.110 | n/a | n/a | n/a | 54.3 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.890 [0.870, 0.908] | 0.890 [0.870, 0.908] | 2.442 [2.012, 2.920] | 0.220 [0.183, 0.260] | 0.110 [0.092, 0.131] | 0.110 | n/a | n/a | n/a | 119 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.863 [0.827, 0.900] | 0.864 [0.825, 0.900] | n/a | n/a | n/a | n/a | 1.000 | 0.863 | 0.000 | 631 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.810 [0.778, 0.844] | 0.810 [0.777, 0.843] | 3.394 [2.648, 4.128] | 0.365 [0.299, 0.429] | 0.182 [0.150, 0.217] | 0.180 | n/a | n/a | n/a | 43.7 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.806 [0.772, 0.840] | 0.805 [0.773, 0.841] | 3.400 [2.673, 4.132] | 0.371 [0.303, 0.436] | 0.185 [0.153, 0.219] | 0.185 | n/a | n/a | n/a | 96.7 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.760 [0.713, 0.810] | 0.798 [0.755, 0.843] | n/a | n/a | n/a | n/a | 0.903 | 0.847 | 0.333 | 950 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.838 [0.806, 0.870] | 0.836 [0.802, 0.866] | 1.319 [0.966, 1.674] | 0.288 [0.233, 0.345] | 0.129 [0.105, 0.163] | 0.126 | n/a | n/a | n/a | 41.9 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.840 [0.808, 0.872] | 0.838 [0.805, 0.869] | 1.242 [0.907, 1.581] | 0.282 [0.226, 0.337] | 0.124 [0.099, 0.158] | 0.122 | n/a | n/a | n/a | 99.2 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.680 [0.590, 0.760] | 0.709 [0.616, 0.784] | n/a | n/a | n/a | n/a | 0.940 | 0.710 | 0.100 | 463 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.746 [0.712, 0.784] | 0.737 [0.701, 0.774] | 2.684 [2.226, 3.170] | 0.494 [0.421, 0.563] | 0.242 [0.209, 0.281] | 0.242 | n/a | n/a | n/a | 44.4 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.740 [0.702, 0.778] | 0.731 [0.695, 0.767] | 2.702 [2.248, 3.187] | 0.494 [0.423, 0.562] | 0.245 [0.210, 0.282] | 0.243 | n/a | n/a | n/a | 96.9 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.430 | 1.000 | 1220 |

### banking77 (mteb/banking77 test, 77 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.494 [0.461, 0.526] | 0.478 [0.445, 0.498] | 2.805 [2.571, 3.023] | 0.740 [0.697, 0.784] | 0.224 [0.197, 0.255] | 0.224 | n/a | n/a | n/a | 78.9 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.313 [0.284, 0.342] | 0.255 [0.234, 0.269] | 3.024 [2.838, 3.207] | 0.830 [0.804, 0.858] | 0.077 [0.066, 0.108] | 0.080 | n/a | n/a | n/a | 97.8 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.157 [0.117, 0.200] | 0.189 [0.128, 0.211] | n/a | n/a | n/a | n/a | 0.367 | 0.157 | 0.003 | 551 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.670 [0.640, 0.699] | 0.658 [0.623, 0.675] | 6.562 [5.894, 7.246] | 0.634 [0.579, 0.690] | 0.310 [0.282, 0.337] | 0.310 | n/a | n/a | n/a | 198 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.380 [0.350, 0.412] | 0.322 [0.301, 0.335] | 6.539 [5.929, 7.142] | 0.781 [0.743, 0.820] | 0.153 [0.130, 0.181] | 0.152 | n/a | n/a | n/a | 192 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.660 [0.603, 0.720] | 0.647 [0.568, 0.672] | n/a | n/a | n/a | n/a | 0.987 | 0.660 | 0.000 | 925 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.500 [0.454, 0.542] | 0.488 [0.430, 0.509] | 8.298 [7.380, 9.267] | 0.926 [0.848, 1.011] | 0.448 [0.409, 0.494] | 0.447 | n/a | n/a | n/a | 84.1 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.326 [0.284, 0.364] | 0.273 [0.238, 0.289] | 7.220 [6.435, 8.117] | 0.884 [0.833, 0.939] | 0.201 [0.175, 0.245] | 0.195 | n/a | n/a | n/a | 108 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.537 [0.480, 0.597] | 0.531 [0.448, 0.551] | n/a | n/a | n/a | n/a | 0.997 | 0.537 | 0.000 | 689 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.300 [0.258, 0.340] | 0.292 [0.241, 0.311] | 4.690 [4.315, 5.061] | 1.003 [0.947, 1.062] | 0.359 [0.323, 0.398] | 0.359 | n/a | n/a | n/a | 145 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.210 [0.174, 0.246] | 0.176 [0.142, 0.193] | 4.320 [4.030, 4.641] | 0.934 [0.896, 0.971] | 0.157 [0.126, 0.192] | 0.157 | n/a | n/a | n/a | 173 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.340 [0.250, 0.430] | 0.245 [0.179, 0.311] | n/a | n/a | n/a | n/a | 0.910 | 0.380 | 0.090 | 1049 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.464 [0.420, 0.508] | 0.464 [0.408, 0.485] | 5.571 [4.914, 6.206] | 0.913 [0.833, 0.989] | 0.414 [0.369, 0.455] | 0.414 | n/a | n/a | n/a | 48.6 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.270 [0.232, 0.308] | 0.231 [0.198, 0.249] | 5.265 [4.717, 5.829] | 0.904 [0.856, 0.951] | 0.190 [0.159, 0.227] | 0.190 | n/a | n/a | n/a | 100 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.210 | 1.000 | 1564 |

### sst2 (stanfordnlp/sst2 validation, 2 labels, n = 872)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 872 | 0.904 [0.882, 0.923] | 0.904 [0.882, 0.923] | 0.286 [0.259, 0.315] | 0.163 [0.143, 0.186] | 0.070 [0.057, 0.089] | 0.070 | n/a | n/a | n/a | 36.0 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 872 | 0.909 [0.889, 0.928] | 0.909 [0.889, 0.927] | 0.287 [0.261, 0.314] | 0.163 [0.143, 0.184] | 0.082 [0.065, 0.100] | 0.082 | n/a | n/a | n/a | 79.9 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.653 [0.597, 0.703] | 0.608 [0.548, 0.665] | n/a | n/a | n/a | n/a | 1.000 | 0.653 | 0.000 | 300 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 872 | 0.893 [0.873, 0.913] | 0.893 [0.872, 0.913] | 1.273 [1.026, 1.552] | 0.207 [0.166, 0.248] | 0.103 [0.084, 0.124] | 0.103 | n/a | n/a | n/a | 54.3 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 872 | 0.896 [0.875, 0.915] | 0.895 [0.874, 0.915] | 1.257 [1.008, 1.533] | 0.203 [0.163, 0.244] | 0.101 [0.083, 0.123] | 0.100 | n/a | n/a | n/a | 119 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.250 [0.203, 0.300] | 0.318 [0.282, 0.350] | n/a | n/a | n/a | n/a | 0.287 | 0.250 | 0.000 | 555 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.876 [0.848, 0.904] | 0.876 [0.848, 0.904] | 0.467 [0.343, 0.586] | 0.205 [0.157, 0.253] | 0.070 [0.057, 0.107] | 0.074 | n/a | n/a | n/a | 44.0 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.880 [0.852, 0.908] | 0.880 [0.852, 0.908] | 0.467 [0.338, 0.588] | 0.203 [0.155, 0.249] | 0.083 [0.062, 0.114] | 0.076 | n/a | n/a | n/a | 96.5 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.033 [0.017, 0.057] | 0.063 [0.031, 0.102] | n/a | n/a | n/a | n/a | 0.033 | 0.033 | 0.053 | 778 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.818 [0.784, 0.852] | 0.813 [0.779, 0.846] | 0.662 [0.528, 0.804] | 0.312 [0.254, 0.373] | 0.144 [0.116, 0.179] | 0.145 | n/a | n/a | n/a | 44.2 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.820 [0.786, 0.854] | 0.816 [0.782, 0.849] | 0.656 [0.524, 0.795] | 0.315 [0.257, 0.376] | 0.142 [0.114, 0.179] | 0.143 | n/a | n/a | n/a | 102 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.480 [0.380, 0.580] | 0.625 [0.522, 0.715] | n/a | n/a | n/a | n/a | 0.530 | 0.660 | 0.200 | 473 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.542 [0.496, 0.584] | 0.420 [0.383, 0.457] | 1.656 [1.469, 1.848] | 0.770 [0.699, 0.848] | 0.395 [0.358, 0.439] | 0.395 | n/a | n/a | n/a | 43.8 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.540 [0.494, 0.582] | 0.417 [0.381, 0.453] | 1.666 [1.478, 1.859] | 0.769 [0.699, 0.849] | 0.398 [0.363, 0.445] | 0.398 | n/a | n/a | n/a | 96.9 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.000 | 1.000 | 1020 |

### sst2_choice (stanfordnlp/sst2 validation, 2 labels, n = 872)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 872 | 0.923 [0.905, 0.939] | 0.923 [0.905, 0.939] | 0.387 [0.296, 0.489] | 0.141 [0.111, 0.173] | 0.069 [0.054, 0.086] | 0.066 | n/a | n/a | n/a | 35.9 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 872 | 0.924 [0.906, 0.940] | 0.924 [0.906, 0.940] | 0.382 [0.290, 0.485] | 0.140 [0.111, 0.172] | 0.068 [0.054, 0.086] | 0.067 | n/a | n/a | n/a | 79.7 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.903 [0.867, 0.937] | 0.903 [0.867, 0.936] | n/a | n/a | n/a | n/a | 1.000 | 0.903 | 0.000 | 348 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 872 | 0.908 [0.888, 0.927] | 0.908 [0.887, 0.927] | 1.546 [1.218, 1.921] | 0.182 [0.146, 0.222] | 0.092 [0.074, 0.113] | 0.091 | n/a | n/a | n/a | 54.3 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 872 | 0.905 [0.884, 0.923] | 0.905 [0.884, 0.923] | 1.551 [1.218, 1.925] | 0.184 [0.148, 0.224] | 0.092 [0.074, 0.113] | 0.093 | n/a | n/a | n/a | 119 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.887 [0.850, 0.920] | 0.888 [0.853, 0.922] | n/a | n/a | n/a | n/a | 0.997 | 0.887 | 0.000 | 563 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.900 [0.874, 0.924] | 0.900 [0.874, 0.924] | 1.079 [0.748, 1.390] | 0.193 [0.143, 0.242] | 0.096 [0.072, 0.122] | 0.095 | n/a | n/a | n/a | 44.0 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.898 [0.872, 0.922] | 0.898 [0.872, 0.922] | 1.058 [0.741, 1.367] | 0.194 [0.145, 0.243] | 0.097 [0.073, 0.123] | 0.097 | n/a | n/a | n/a | 96.4 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.847 [0.803, 0.887] | 0.868 [0.827, 0.903] | n/a | n/a | n/a | n/a | 0.950 | 0.887 | 0.047 | 811 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.898 [0.872, 0.926] | 0.898 [0.872, 0.926] | 0.395 [0.289, 0.507] | 0.185 [0.136, 0.233] | 0.087 [0.063, 0.115] | 0.087 | n/a | n/a | n/a | 43.6 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.898 [0.872, 0.926] | 0.898 [0.872, 0.926] | 0.368 [0.271, 0.471] | 0.182 [0.134, 0.229] | 0.081 [0.058, 0.109] | 0.081 | n/a | n/a | n/a | 99.1 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.840 [0.770, 0.910] | 0.878 [0.817, 0.933] | n/a | n/a | n/a | n/a | 0.910 | 0.910 | 0.120 | 424 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.848 [0.818, 0.878] | 0.847 [0.817, 0.878] | 0.878 [0.679, 1.077] | 0.288 [0.230, 0.341] | 0.141 [0.112, 0.172] | 0.137 | n/a | n/a | n/a | 43.8 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.844 [0.814, 0.874] | 0.843 [0.811, 0.874] | 0.880 [0.681, 1.077] | 0.288 [0.230, 0.342] | 0.137 [0.111, 0.170] | 0.133 | n/a | n/a | n/a | 96.9 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.040 [0.010, 0.080] | 0.074 [0.018, 0.138] | n/a | n/a | n/a | n/a | 0.040 | 0.340 | 0.960 | 1084 |

### yelp (Yelp/yelp_review_full test, 5 labels, n = 1000)

| Model | Arm | n | Accuracy | Macro-F1 | NLL | Brier | ECE (15 bins) | ECE (10 bins) | Valid answers | Salvaged accuracy | Hit token cap | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.472 [0.443, 0.501] | 0.403 [0.376, 0.429] | 1.616 [1.512, 1.725] | 0.754 [0.715, 0.793] | 0.262 [0.235, 0.295] | 0.262 | n/a | n/a | n/a | 35.3 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.476 [0.446, 0.505] | 0.406 [0.380, 0.431] | 1.616 [1.515, 1.725] | 0.754 [0.715, 0.793] | 0.258 [0.233, 0.289] | 0.257 | n/a | n/a | n/a | 79.6 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 300 | 0.273 [0.223, 0.323] | 0.252 [0.208, 0.297] | n/a | n/a | n/a | n/a | 0.910 | 0.273 | 0.000 | 356 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.439 [0.407, 0.470] | 0.405 [0.377, 0.431] | 9.317 [8.739, 9.928] | 1.097 [1.037, 1.158] | 0.537 [0.508, 0.570] | 0.537 | n/a | n/a | n/a | 68.8 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.438 [0.406, 0.468] | 0.404 [0.375, 0.430] | 9.340 [8.745, 9.951] | 1.099 [1.041, 1.159] | 0.542 [0.512, 0.574] | 0.536 | n/a | n/a | n/a | 122 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 300 | 0.393 [0.340, 0.447] | 0.368 [0.321, 0.412] | n/a | n/a | n/a | n/a | 0.977 | 0.393 | 0.000 | 584 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.546 [0.504, 0.590] | 0.507 [0.465, 0.545] | 3.896 [3.402, 4.437] | 0.825 [0.745, 0.904] | 0.394 [0.354, 0.438] | 0.392 | n/a | n/a | n/a | 44.0 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.542 [0.498, 0.584] | 0.503 [0.460, 0.542] | 3.895 [3.405, 4.433] | 0.833 [0.755, 0.910] | 0.407 [0.366, 0.449] | 0.403 | n/a | n/a | n/a | 97.3 |
| Qwen/Qwen3-1.7B | Normal generation | 300 | 0.400 [0.343, 0.453] | 0.392 [0.339, 0.445] | n/a | n/a | n/a | n/a | 0.907 | 0.440 | 0.077 | 815 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.584 [0.546, 0.626] | 0.558 [0.518, 0.599] | 1.424 [1.256, 1.601] | 0.635 [0.572, 0.695] | 0.234 [0.202, 0.282] | 0.234 | n/a | n/a | n/a | 50.8 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.588 [0.550, 0.630] | 0.565 [0.525, 0.606] | 1.422 [1.255, 1.593] | 0.631 [0.566, 0.690] | 0.232 [0.197, 0.275] | 0.227 | n/a | n/a | n/a | 99.9 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 100 | 0.520 [0.420, 0.610] | 0.510 [0.404, 0.593] | n/a | n/a | n/a | n/a | 0.940 | 0.550 | 0.050 | 498 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 0.276 [0.238, 0.316] | 0.192 [0.160, 0.225] | 2.861 [2.660, 3.054] | 1.118 [1.055, 1.174] | 0.502 [0.463, 0.538] | 0.502 | n/a | n/a | n/a | 44.2 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 0.268 [0.230, 0.308] | 0.183 [0.153, 0.216] | 2.901 [2.693, 3.094] | 1.128 [1.067, 1.187] | 0.510 [0.470, 0.546] | 0.510 | n/a | n/a | n/a | 97.6 |
| Qwen/Qwen3-0.6B | Normal generation | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/a | n/a | n/a | n/a | 0.000 | 0.110 | 1.000 | 1141 |

## Calibration

T is one scalar fitted by NLL on the calibration split (train rows, none of whose texts occur in the evaluation split): per dataset, and pooled over all datasets per model and arm. ECE and NLL are measured on the evaluation split only. n/a: the fit did not converge inside [0.05, 20] (too few calibration rows) or the split was not run. ECE brackets can sit above the point estimate (see Accuracy). Diagrams: reliability before (T = 1) and after the pooled T, the one that ships, for the MirethSTM1 arm only (a report built without --figures keeps every arm's diagram and bin table in bench/out/<run>/report/).

| Model | Arm | Dataset | T dataset | T pooled | ECE-15 at T = 1 | NLL at T = 1 | ECE-15 at T dataset | NLL at T dataset | ECE-15 at T pooled | NLL at T pooled | Diagram |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | ag_news | 2.729 | 2.112 | 0.116 [0.097, 0.139] | 0.832 | 0.056 [0.050, 0.084] | 0.488 | 0.062 [0.053, 0.089] | 0.511 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.ag_news.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | banking77 | 1.933 | 2.112 | 0.224 [0.197, 0.255] | 2.805 | 0.095 [0.073, 0.128] | 2.224 | 0.125 [0.101, 0.158] | 2.239 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.banking77.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | sst2 | 0.820 | 2.112 | 0.070 [0.057, 0.089] | 0.286 | 0.039 [0.030, 0.059] | 0.269 | 0.199 [0.179, 0.216] | 0.404 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | sst2_choice | 2.490 | 2.112 | 0.069 [0.054, 0.086] | 0.387 | 0.022 [0.014, 0.039] | 0.224 | 0.036 [0.025, 0.056] | 0.232 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.sst2_choice.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | yelp | 2.636 | 2.112 | 0.262 [0.235, 0.295] | 1.616 | 0.094 [0.070, 0.126] | 1.220 | 0.122 [0.102, 0.155] | 1.238 | [png](benchmark/Qwen--Qwen2.5-1.5B-Instruct.yelp.mireth.png) |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | ag_news | 2.723 | 2.103 | 0.112 [0.095, 0.137] | 0.831 | 0.055 [0.043, 0.080] | 0.488 | 0.067 [0.053, 0.090] | 0.512 | not published |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | banking77 | 1.846 | 2.103 | 0.077 [0.066, 0.108] | 3.024 | 0.083 [0.065, 0.114] | 2.660 | 0.115 [0.091, 0.142] | 2.678 | not published |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | sst2 | 0.744 | 2.103 | 0.082 [0.065, 0.100] | 0.287 | 0.037 [0.031, 0.059] | 0.262 | 0.209 [0.189, 0.226] | 0.406 | not published |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | sst2_choice | 2.497 | 2.103 | 0.068 [0.054, 0.086] | 0.382 | 0.021 [0.014, 0.040] | 0.222 | 0.036 [0.026, 0.056] | 0.230 | not published |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | yelp | 2.643 | 2.103 | 0.258 [0.233, 0.289] | 1.616 | 0.097 [0.077, 0.129] | 1.221 | 0.115 [0.100, 0.151] | 1.240 | not published |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | ag_news | 8.762 | 7.930 | 0.110 [0.092, 0.130] | 2.438 | 0.025 [0.020, 0.047] | 0.428 | 0.047 [0.035, 0.066] | 0.438 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.ag_news.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | banking77 | 6.382 | 7.930 | 0.310 [0.282, 0.337] | 6.562 | 0.048 [0.040, 0.082] | 1.552 | 0.143 [0.122, 0.171] | 1.625 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.banking77.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | sst2 | 6.908 | 7.930 | 0.103 [0.084, 0.124] | 1.273 | 0.033 [0.021, 0.053] | 0.295 | 0.068 [0.054, 0.089] | 0.301 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | sst2_choice | 8.880 | 7.930 | 0.092 [0.074, 0.113] | 1.546 | 0.053 [0.039, 0.074] | 0.272 | 0.030 [0.022, 0.051] | 0.274 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.sst2_choice.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | yelp | 15.383 | 7.930 | 0.537 [0.508, 0.570] | 9.317 | 0.124 [0.099, 0.155] | 1.325 | 0.298 [0.272, 0.329] | 1.524 | [png](benchmark/Qwen--Qwen3-4B-Instruct-2507.yelp.mireth.png) |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | ag_news | 8.733 | 8.428 | 0.110 [0.092, 0.131] | 2.442 | 0.023 [0.019, 0.046] | 0.429 | 0.033 [0.021, 0.052] | 0.432 | not published |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | banking77 | 6.622 | 8.428 | 0.153 [0.130, 0.181] | 6.539 | 0.053 [0.046, 0.083] | 2.236 | 0.120 [0.101, 0.148] | 2.291 | not published |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | sst2 | 6.867 | 8.428 | 0.101 [0.083, 0.123] | 1.257 | 0.036 [0.024, 0.055] | 0.293 | 0.082 [0.068, 0.103] | 0.304 | not published |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | sst2_choice | 8.835 | 8.428 | 0.092 [0.074, 0.113] | 1.551 | 0.040 [0.027, 0.062] | 0.273 | 0.046 [0.032, 0.069] | 0.273 | not published |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | yelp | 15.366 | 8.428 | 0.542 [0.512, 0.574] | 9.340 | 0.120 [0.096, 0.150] | 1.325 | 0.281 [0.255, 0.313] | 1.484 | not published |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | ag_news | 9.557 | 7.007 | 0.182 [0.150, 0.217] | 3.394 | 0.052 [0.035, 0.090] | 0.603 | 0.106 [0.079, 0.140] | 0.677 | [png](benchmark/Qwen--Qwen3-1.7B.ag_news.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | banking77 | 6.772 | 7.007 | 0.448 [0.409, 0.494] | 8.298 | 0.068 [0.054, 0.119] | 2.139 | 0.071 [0.062, 0.122] | 2.142 | [png](benchmark/Qwen--Qwen3-1.7B.banking77.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | sst2 | 3.293 | 7.007 | 0.070 [0.057, 0.107] | 0.467 | 0.079 [0.058, 0.110] | 0.356 | 0.202 [0.179, 0.232] | 0.459 | [png](benchmark/Qwen--Qwen3-1.7B.sst2.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | sst2_choice | 6.499 | 7.007 | 0.096 [0.072, 0.122] | 1.079 | 0.027 [0.019, 0.057] | 0.290 | 0.033 [0.025, 0.064] | 0.292 | [png](benchmark/Qwen--Qwen3-1.7B.sst2_choice.mireth.png) |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | yelp | 6.763 | 7.007 | 0.394 [0.354, 0.438] | 3.896 | 0.076 [0.063, 0.129] | 1.157 | 0.071 [0.062, 0.129] | 1.156 | [png](benchmark/Qwen--Qwen3-1.7B.yelp.mireth.png) |
| Qwen/Qwen3-1.7B | First token (original demo's method) | ag_news | 9.562 | 7.024 | 0.185 [0.153, 0.219] | 3.400 | 0.046 [0.032, 0.084] | 0.605 | 0.103 [0.074, 0.137] | 0.678 | not published |
| Qwen/Qwen3-1.7B | First token (original demo's method) | banking77 | 6.717 | 7.024 | 0.201 [0.175, 0.245] | 7.220 | 0.084 [0.064, 0.123] | 2.523 | 0.083 [0.063, 0.123] | 2.529 | not published |
| Qwen/Qwen3-1.7B | First token (original demo's method) | sst2 | 3.213 | 7.024 | 0.083 [0.062, 0.114] | 0.467 | 0.078 [0.057, 0.109] | 0.351 | 0.205 [0.180, 0.234] | 0.458 | not published |
| Qwen/Qwen3-1.7B | First token (original demo's method) | sst2_choice | 6.556 | 7.024 | 0.097 [0.073, 0.123] | 1.058 | 0.028 [0.020, 0.057] | 0.288 | 0.030 [0.024, 0.061] | 0.290 | not published |
| Qwen/Qwen3-1.7B | First token (original demo's method) | yelp | 6.766 | 7.024 | 0.407 [0.366, 0.449] | 3.895 | 0.064 [0.063, 0.124] | 1.159 | 0.067 [0.060, 0.123] | 1.158 | not published |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | ag_news | 2.956 | 2.702 | 0.129 [0.105, 0.163] | 1.319 | 0.033 [0.030, 0.071] | 0.620 | 0.039 [0.031, 0.079] | 0.636 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.ag_news.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | banking77 | 2.694 | 2.702 | 0.359 [0.323, 0.398] | 4.690 | 0.107 [0.074, 0.145] | 3.216 | 0.108 [0.074, 0.146] | 3.216 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.banking77.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | sst2 | 3.084 | 2.702 | 0.144 [0.116, 0.179] | 0.662 | 0.066 [0.045, 0.099] | 0.418 | 0.049 [0.032, 0.084] | 0.415 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | sst2_choice | 1.840 | 2.702 | 0.087 [0.063, 0.115] | 0.395 | 0.034 [0.023, 0.062] | 0.279 | 0.054 [0.039, 0.083] | 0.286 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.sst2_choice.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | yelp | 2.738 | 2.702 | 0.234 [0.202, 0.282] | 1.424 | 0.059 [0.048, 0.112] | 0.990 | 0.056 [0.047, 0.109] | 0.990 | [png](benchmark/HuggingFaceTB--SmolLM3-3B.yelp.mireth.png) |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | ag_news | 2.810 | 2.545 | 0.124 [0.099, 0.158] | 1.242 | 0.027 [0.027, 0.067] | 0.614 | 0.034 [0.026, 0.071] | 0.631 | not published |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | banking77 | 2.441 | 2.545 | 0.157 [0.126, 0.192] | 4.320 | 0.091 [0.059, 0.123] | 3.441 | 0.097 [0.066, 0.131] | 3.443 | not published |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | sst2 | 3.022 | 2.545 | 0.142 [0.114, 0.179] | 0.656 | 0.057 [0.037, 0.091] | 0.427 | 0.047 [0.030, 0.080] | 0.424 | not published |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | sst2_choice | 1.652 | 2.545 | 0.081 [0.058, 0.109] | 0.368 | 0.036 [0.020, 0.064] | 0.282 | 0.051 [0.039, 0.080] | 0.291 | not published |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | yelp | 2.721 | 2.545 | 0.232 [0.197, 0.275] | 1.422 | 0.065 [0.050, 0.117] | 0.990 | 0.069 [0.054, 0.117] | 0.991 | not published |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | ag_news | 5.900 | 4.639 | 0.242 [0.209, 0.281] | 2.684 | 0.059 [0.053, 0.104] | 0.753 | 0.098 [0.076, 0.140] | 0.785 | [png](benchmark/Qwen--Qwen3-0.6B.ag_news.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | banking77 | 3.909 | 4.639 | 0.414 [0.369, 0.455] | 5.571 | 0.077 [0.064, 0.129] | 2.444 | 0.153 [0.118, 0.197] | 2.509 | [png](benchmark/Qwen--Qwen3-0.6B.banking77.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | sst2 | 11.539 | 4.639 | 0.395 [0.358, 0.439] | 1.656 | 0.204 [0.167, 0.242] | 0.644 | 0.261 [0.225, 0.297] | 0.664 | [png](benchmark/Qwen--Qwen3-0.6B.sst2.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | sst2_choice | 5.027 | 4.639 | 0.141 [0.112, 0.172] | 0.878 | 0.062 [0.043, 0.094] | 0.373 | 0.041 [0.032, 0.079] | 0.370 | [png](benchmark/Qwen--Qwen3-0.6B.sst2_choice.mireth.png) |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | yelp | 5.945 | 4.639 | 0.502 [0.463, 0.538] | 2.861 | 0.117 [0.081, 0.153] | 1.524 | 0.130 [0.101, 0.168] | 1.534 | [png](benchmark/Qwen--Qwen3-0.6B.yelp.mireth.png) |
| Qwen/Qwen3-0.6B | First token (original demo's method) | ag_news | 5.902 | 4.886 | 0.245 [0.210, 0.282] | 2.702 | 0.059 [0.050, 0.102] | 0.755 | 0.097 [0.071, 0.136] | 0.776 | not published |
| Qwen/Qwen3-0.6B | First token (original demo's method) | banking77 | 3.973 | 4.886 | 0.190 [0.159, 0.227] | 5.265 | 0.075 [0.055, 0.116] | 2.892 | 0.121 [0.089, 0.160] | 2.948 | not published |
| Qwen/Qwen3-0.6B | First token (original demo's method) | sst2 | 11.531 | 4.886 | 0.398 [0.363, 0.445] | 1.666 | 0.195 [0.157, 0.234] | 0.644 | 0.266 [0.230, 0.301] | 0.659 | not published |
| Qwen/Qwen3-0.6B | First token (original demo's method) | sst2_choice | 4.973 | 4.886 | 0.137 [0.111, 0.170] | 0.880 | 0.047 [0.031, 0.080] | 0.374 | 0.048 [0.031, 0.081] | 0.373 | not published |
| Qwen/Qwen3-0.6B | First token (original demo's method) | yelp | 6.140 | 4.886 | 0.510 [0.470, 0.546] | 2.901 | 0.119 [0.084, 0.156] | 1.524 | 0.144 [0.116, 0.182] | 1.532 | not published |

## Yelp: score levels

Levels 0 to 4 (one to five stars). Argmax is the most likely level; the expected level is sum of k p_k. Normal generation is scored on its valid answers only.

| Model | Arm | n | MAE argmax | RMSE argmax | Within one level | MAE expected | RMSE expected | MAE expected at T pooled | RMSE expected at T pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | 1000 | 0.691 | 1.038 | 0.863 | 0.640 | 0.853 | 0.691 | 0.864 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | 1000 | 0.677 | 1.017 | 0.870 | 0.641 | 0.854 | 0.693 | 0.866 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | 273 valid of 300 | 1.073 | 1.441 | 0.729 | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | 1000 | 0.716 | 1.029 | 0.860 | 0.723 | 1.022 | 0.710 | 0.897 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | 1000 | 0.713 | 1.023 | 0.864 | 0.723 | 1.021 | 0.711 | 0.886 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | 293 valid of 300 | 0.730 | 1.002 | 0.870 | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | 500 | 0.564 | 0.906 | 0.906 | 0.566 | 0.881 | 0.661 | 0.820 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | 500 | 0.580 | 0.932 | 0.898 | 0.574 | 0.886 | 0.664 | 0.823 |
| Qwen/Qwen3-1.7B | Normal generation | 272 valid of 300 | 0.625 | 0.883 | 0.941 | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | 500 | 0.498 | 0.823 | 0.926 | 0.503 | 0.741 | 0.536 | 0.673 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | 500 | 0.498 | 0.833 | 0.926 | 0.501 | 0.739 | 0.529 | 0.671 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | 94 valid of 100 | 0.532 | 0.838 | 0.915 | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | 500 | 1.608 | 2.123 | 0.536 | 1.445 | 1.832 | 1.111 | 1.319 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | 500 | 1.636 | 2.148 | 0.530 | 1.473 | 1.861 | 1.112 | 1.319 |

## SST-2: yes-bias

The same sentences as a yes/no question ("Is the sentiment positive?") and as a two-option choice. A predicted-positive rate well above the gold rate in the yes/no form only is a yes-bias.

| Model | Arm | Form | n | Accuracy | Predicted positive | Gold positive |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | yes/no | 872 | 0.904 | 0.463 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | yes/no | 872 | 0.909 | 0.474 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | yes/no | 300 | 0.653 | 0.840 | 0.500 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | choice | 872 | 0.923 | 0.492 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | choice | 872 | 0.924 | 0.493 | 0.509 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | choice | 300 | 0.903 | 0.497 | 0.500 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | yes/no | 872 | 0.893 | 0.453 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | yes/no | 872 | 0.896 | 0.453 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | yes/no | 300 | 0.250 | 0.000 | 0.500 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | choice | 872 | 0.908 | 0.525 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | choice | 872 | 0.905 | 0.526 | 0.509 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | choice | 300 | 0.887 | 0.517 | 0.500 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | yes/no | 500 | 0.876 | 0.496 | 0.500 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | yes/no | 500 | 0.880 | 0.488 | 0.500 |
| Qwen/Qwen3-1.7B | Normal generation | yes/no | 300 | 0.033 | 0.003 | 0.500 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | choice | 500 | 0.900 | 0.536 | 0.500 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | choice | 500 | 0.898 | 0.538 | 0.500 |
| Qwen/Qwen3-1.7B | Normal generation | choice | 300 | 0.847 | 0.513 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | yes/no | 500 | 0.818 | 0.342 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | yes/no | 500 | 0.820 | 0.344 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | yes/no | 100 | 0.480 | 0.230 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | choice | 500 | 0.898 | 0.442 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | choice | 500 | 0.898 | 0.442 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | choice | 100 | 0.840 | 0.490 | 0.500 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | yes/no | 500 | 0.542 | 0.958 | 0.500 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | yes/no | 500 | 0.540 | 0.960 | 0.500 |
| Qwen/Qwen3-0.6B | Normal generation | yes/no | 100 | 0.000 | 0.000 | 0.500 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | choice | 500 | 0.848 | 0.572 | 0.500 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | choice | 500 | 0.844 | 0.572 | 0.500 |
| Qwen/Qwen3-0.6B | Normal generation | choice | 100 | 0.040 | 0.000 | 0.500 |

## Probability on the allowed answers

A label's score is the log-probability of its exact answer text, so exp(score) summed over a question's labels is the share of the model's probability that falls on an allowed answer at all. Where that share is small the model would rather write something else at that place (a quoted "true", a quoted number, other text), and the answer is read from the tail of its distribution: the ranking can still be right, but the scores sit far below 0, where bf16 rounding is coarser and exact ties become possible. MirethSTM1 arm, evaluation split: the median share over the rows and, in brackets, the share of rows where the labels hold under 1%.

| Model | ag_news | banking77 | sst2 | sst2_choice | yelp |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | 0.998 [0.000] | 0.842 [0.000] | 0.240 [0.000] | 0.998 [0.000] | 0.965 [0.001] |
| Qwen/Qwen3-4B-Instruct-2507 | 1.000 [0.000] | 1.000 [0.001] | 0.047 [0.460] | 1.000 [0.005] | 1.000 [0.016] |
| Qwen/Qwen3-1.7B | 1.000 [0.000] | 1.000 [0.002] | 0.852 [0.000] | 1.000 [0.000] | 0.000 [0.990] |
| HuggingFaceTB/SmolLM3-3B | 0.855 [0.000] | 0.155 [0.008] | 0.848 [0.000] | 0.906 [0.000] | 0.390 [0.012] |
| Qwen/Qwen3-0.6B | 0.999 [0.004] | 0.989 [0.000] | 0.001 [1.000] | 0.986 [0.000] | 0.397 [0.032] |

## First-token ties

The first-token arm scores only each label's first token, so options whose first tokens are the same token get the same score. A tie goes to the earlier option in the request, as the argmax of the original demo's PyTorch path does (its Mac build generates a few tokens when first tokens collide; this arm does not), and the tied options split the probability. Labels in ties: mean per row of options whose score equals another option's, which counts shared first tokens and also chance ties on the bf16 grid.

| Model | Dataset | n | Rows with a tie | Labels in ties | Predictions decided by the tie rule | Accuracy of those |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | ag_news | 1000 | 37 | 0.1 | 7 | 0.429 |
| Qwen/Qwen2.5-1.5B-Instruct | banking77 | 1000 | 1000 | 55.2 | 636 | 0.171 |
| Qwen/Qwen2.5-1.5B-Instruct | sst2 | 872 | 9 | 0.0 | 9 | 0.889 |
| Qwen/Qwen2.5-1.5B-Instruct | sst2_choice | 872 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | yelp | 1000 | 152 | 0.3 | 21 | 0.429 |
| Qwen/Qwen2.5-1.5B-Instruct | public items, easy | 48 | 2 | 0.1 | 1 | 0.000 |
| Qwen/Qwen2.5-1.5B-Instruct | public items, standard | 72 | 15 | 0.4 | 12 | 0.250 |
| Qwen/Qwen2.5-1.5B-Instruct | public items, hard | 111 | 32 | 0.8 | 21 | 0.143 |
| Qwen/Qwen3-4B-Instruct-2507 | ag_news | 1000 | 27 | 0.1 | 1 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | banking77 | 1000 | 1000 | 51.9 | 635 | 0.197 |
| Qwen/Qwen3-4B-Instruct-2507 | sst2 | 872 | 1 | 0.0 | 1 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | sst2_choice | 872 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | yelp | 1000 | 75 | 0.2 | 4 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | public items, easy | 48 | 2 | 0.1 | 0 | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | public items, standard | 72 | 14 | 0.4 | 10 | 0.200 |
| Qwen/Qwen3-4B-Instruct-2507 | public items, hard | 111 | 28 | 0.7 | 16 | 0.188 |
| Qwen/Qwen3-1.7B | ag_news | 500 | 18 | 0.1 | 1 | 0.000 |
| Qwen/Qwen3-1.7B | banking77 | 500 | 500 | 53.0 | 331 | 0.166 |
| Qwen/Qwen3-1.7B | sst2 | 500 | 3 | 0.0 | 3 | 0.667 |
| Qwen/Qwen3-1.7B | sst2_choice | 500 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-1.7B | yelp | 500 | 24 | 0.1 | 2 | 0.500 |
| Qwen/Qwen3-1.7B | public items, easy | 48 | 2 | 0.1 | 0 | n/a |
| Qwen/Qwen3-1.7B | public items, standard | 72 | 15 | 0.4 | 11 | 0.182 |
| Qwen/Qwen3-1.7B | public items, hard | 111 | 28 | 0.7 | 18 | 0.111 |
| HuggingFaceTB/SmolLM3-3B | ag_news | 500 | 8 | 0.0 | 2 | 0.500 |
| HuggingFaceTB/SmolLM3-3B | banking77 | 500 | 500 | 56.1 | 231 | 0.147 |
| HuggingFaceTB/SmolLM3-3B | sst2 | 500 | 1 | 0.0 | 1 | 1.000 |
| HuggingFaceTB/SmolLM3-3B | sst2_choice | 500 | 0 | 0.0 | 0 | n/a |
| HuggingFaceTB/SmolLM3-3B | yelp | 500 | 47 | 0.2 | 10 | 0.400 |
| HuggingFaceTB/SmolLM3-3B | public items, easy | 48 | 1 | 0.0 | 0 | n/a |
| HuggingFaceTB/SmolLM3-3B | public items, standard | 72 | 19 | 0.6 | 13 | 0.231 |
| HuggingFaceTB/SmolLM3-3B | public items, hard | 111 | 34 | 0.8 | 21 | 0.190 |
| Qwen/Qwen3-0.6B | ag_news | 500 | 6 | 0.0 | 2 | 0.500 |
| Qwen/Qwen3-0.6B | banking77 | 500 | 500 | 53.6 | 315 | 0.127 |
| Qwen/Qwen3-0.6B | sst2 | 500 | 1 | 0.0 | 1 | 0.000 |
| Qwen/Qwen3-0.6B | sst2_choice | 500 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-0.6B | yelp | 500 | 71 | 0.3 | 13 | 0.385 |
| Qwen/Qwen3-0.6B | public items, easy | 48 | 0 | 0.0 | 0 | n/a |
| Qwen/Qwen3-0.6B | public items, standard | 72 | 12 | 0.4 | 11 | 0.091 |
| Qwen/Qwen3-0.6B | public items, hard | 111 | 30 | 0.8 | 20 | 0.150 |

## Many questions in one call (SPEC 3.1)

Five checked questions (the topic, and four yes/no questions that follow from the gold topic) asked alone, one call each, and inside one 20-question call next to 15 filler yes/no questions, at three placements. Every question has a branch of its own, so a question scores the same wherever it sits; what is left is float rounding (the largest score difference, in summed log-prob, over every label). Brackets: share of articles answered yes (the true share is about a quarter). A model whose fp32 weights fit the card has the same run in fp32 under its table. The earlier engine put all questions into one shared prompt; on Qwen/Qwen2.5-1.5B-Instruct the same test then gave 0.864 alone, 0.809 first, 0.575 last and 0.500 spread.

### Qwen/Qwen2.5-1.5B-Instruct, bf16 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.810 | 0.975 [0.26] | 0.825 [0.38] | 0.880 [0.25] | 0.720 [0.51] | 0.842 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.805 | 0.975 [0.26] | 0.815 [0.39] | 0.885 [0.24] | 0.730 [0.50] | 0.842 | 0.992 | 0.831 |
| One call of 20, checked questions last (16 to 20) | 0.810 | 0.975 [0.26] | 0.820 [0.38] | 0.885 [0.24] | 0.720 [0.51] | 0.842 | 0.994 | 0.834 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.805 | 0.975 [0.26] | 0.815 [0.39] | 0.890 [0.25] | 0.720 [0.51] | 0.841 | 0.991 | 0.749 |

### Qwen/Qwen2.5-1.5B-Instruct, fp32 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.805 | 0.980 [0.26] | 0.820 [0.38] | 0.880 [0.25] | 0.740 [0.49] | 0.845 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.805 | 0.980 [0.26] | 0.820 [0.38] | 0.880 [0.25] | 0.740 [0.49] | 0.845 | 1.000 | 0.00016 |
| One call of 20, checked questions last (16 to 20) | 0.805 | 0.980 [0.26] | 0.820 [0.38] | 0.880 [0.25] | 0.740 [0.49] | 0.845 | 1.000 | 0.000147 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.805 | 0.980 [0.26] | 0.820 [0.38] | 0.880 [0.25] | 0.740 [0.49] | 0.845 | 1.000 | 0.000152 |

### Qwen/Qwen3-4B-Instruct-2507, bf16 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.840 | 0.975 [0.23] | 0.860 [0.23] | 0.860 [0.13] | 0.860 [0.17] | 0.879 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.840 | 0.975 [0.23] | 0.850 [0.23] | 0.855 [0.14] | 0.860 [0.17] | 0.876 | 0.997 | 2.87 |
| One call of 20, checked questions last (16 to 20) | 0.840 | 0.975 [0.23] | 0.860 [0.22] | 0.855 [0.14] | 0.860 [0.17] | 0.878 | 0.997 | 2.75 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.840 | 0.975 [0.23] | 0.850 [0.23] | 0.855 [0.14] | 0.860 [0.17] | 0.876 | 0.997 | 2.75 |

### Qwen/Qwen3-1.7B, bf16 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.775 | 0.980 [0.26] | 0.825 [0.34] | 0.870 [0.24] | 0.870 [0.27] | 0.864 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.770 | 0.980 [0.26] | 0.825 [0.34] | 0.880 [0.25] | 0.875 [0.27] | 0.866 | 0.994 | 3.25 |
| One call of 20, checked questions last (16 to 20) | 0.770 | 0.980 [0.26] | 0.825 [0.34] | 0.880 [0.25] | 0.870 [0.27] | 0.865 | 0.994 | 4.38 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.770 | 0.980 [0.26] | 0.825 [0.34] | 0.880 [0.25] | 0.870 [0.27] | 0.865 | 0.993 | 3.38 |

### Qwen/Qwen3-1.7B, fp32 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.780 | 0.980 [0.26] | 0.835 [0.35] | 0.875 [0.24] | 0.870 [0.27] | 0.868 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.780 | 0.980 [0.26] | 0.835 [0.35] | 0.875 [0.24] | 0.870 [0.27] | 0.868 | 1.000 | 0.000529 |
| One call of 20, checked questions last (16 to 20) | 0.780 | 0.980 [0.26] | 0.835 [0.35] | 0.875 [0.24] | 0.870 [0.27] | 0.868 | 1.000 | 0.00102 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.780 | 0.980 [0.26] | 0.835 [0.35] | 0.875 [0.24] | 0.870 [0.27] | 0.868 | 1.000 | 0.00069 |

### HuggingFaceTB/SmolLM3-3B, bf16 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.805 | 0.970 [0.22] | 0.865 [0.32] | 0.840 [0.14] | 0.810 [0.36] | 0.858 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.800 | 0.970 [0.22] | 0.880 [0.33] | 0.840 [0.14] | 0.815 [0.35] | 0.861 | 0.995 | 0.651 |
| One call of 20, checked questions last (16 to 20) | 0.800 | 0.970 [0.22] | 0.880 [0.33] | 0.840 [0.14] | 0.810 [0.36] | 0.860 | 0.996 | 0.497 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.800 | 0.970 [0.22] | 0.880 [0.33] | 0.835 [0.14] | 0.810 [0.36] | 0.859 | 0.995 | 0.698 |

### Qwen/Qwen3-0.6B, bf16 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.720 | 0.960 [0.24] | 0.805 [0.41] | 0.865 [0.26] | 0.425 [0.81] | 0.755 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.715 | 0.965 [0.24] | 0.805 [0.41] | 0.870 [0.25] | 0.420 [0.82] | 0.755 | 0.996 | 1.5 |
| One call of 20, checked questions last (16 to 20) | 0.720 | 0.960 [0.24] | 0.805 [0.41] | 0.865 [0.26] | 0.415 [0.82] | 0.753 | 0.996 | 4.91 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.715 | 0.965 [0.24] | 0.800 [0.41] | 0.870 [0.25] | 0.415 [0.82] | 0.753 | 0.994 | 1.49 |

### Qwen/Qwen3-0.6B, fp32 (AG News test, n = 200)

| Mode | topic | is_sports | is_business | is_scitech | is_world | Mean | Same answer as alone | Largest score difference from alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alone, one call per question | 0.715 | 0.965 [0.24] | 0.795 [0.41] | 0.855 [0.27] | 0.390 [0.85] | 0.744 | n/a | n/a |
| One call of 20, checked questions first (1 to 5) | 0.715 | 0.965 [0.24] | 0.795 [0.41] | 0.855 [0.27] | 0.390 [0.85] | 0.744 | 1.000 | 0.000294 |
| One call of 20, checked questions last (16 to 20) | 0.715 | 0.965 [0.24] | 0.795 [0.41] | 0.855 [0.27] | 0.390 [0.85] | 0.744 | 1.000 | 0.000238 |
| One call of 20, checked questions spread (4, 8, 12, 16, 20) | 0.715 | 0.965 [0.24] | 0.795 [0.41] | 0.855 [0.27] | 0.390 [0.85] | 0.744 | 1.000 | 0.000314 |

## JevBench public items

The 231 public items of JevBench (github.com/fstandhartinger/jevbench, MIT), fetched at run time, scored here by us. Not a JevBench score: its board adds sealed items, a speed and a cost axis and runs entrants on its own hardware. Chance: mean of 1/options over the items run; chance-corrected accuracy is 100 (accuracy - chance) / (1 - chance), floored at 0, as JevBench's Intelligence axis per tier. ECE-10 matches JevBench's binning. The pooled T comes from our own calibration splits, never from JevBench items.

| Model | Arm | Tier | n | Accuracy | Chance | Chance-corrected | ECE-10 at T = 1 | ECE-10 at T pooled | ECE-15 at T = 1 | NLL | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | easy | 48 | 0.917 [0.833, 0.979] | 0.284 | 88.4 | 0.060 | 0.115 (T = 2.112) | 0.059 | 0.243 | 0.117 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | standard | 72 | 0.569 [0.444, 0.694] | 0.311 | 37.5 | 0.240 | 0.089 (T = 2.112) | 0.280 | 1.224 | 0.578 |
| Qwen/Qwen2.5-1.5B-Instruct | MirethSTM1 (full label) | hard | 111 | 0.342 [0.261, 0.432] | 0.336 | 0.9 | 0.401 | 0.247 (T = 2.112) | 0.390 | 2.077 | 0.985 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | easy | 48 | 0.917 [0.833, 0.979] | 0.284 | 88.4 | 0.061 | 0.141 (T = 2.103) | 0.055 | 0.255 | 0.119 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | standard | 72 | 0.542 [0.417, 0.667] | 0.311 | 33.5 | 0.238 | 0.088 (T = 2.103) | 0.252 | 1.319 | 0.603 |
| Qwen/Qwen2.5-1.5B-Instruct | First token (original demo's method) | hard | 111 | 0.333 [0.243, 0.423] | 0.336 | 0.0 | 0.346 | 0.251 (T = 2.103) | 0.346 | 2.017 | 0.960 |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | easy | 48 | 0.479 [0.333, 0.625] | 0.284 | 27.2 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | standard | 72 | 0.431 [0.319, 0.542] | 0.311 | 17.3 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen2.5-1.5B-Instruct | Normal generation | hard | 111 | 0.234 [0.162, 0.315] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.000 | 0.069 (T = 7.930) | 0.000 | 0.000 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | standard | 72 | 0.708 [0.611, 0.819] | 0.311 | 57.7 | 0.286 | 0.161 (T = 7.930) | 0.286 | 4.740 | 0.575 |
| Qwen/Qwen3-4B-Instruct-2507 | MirethSTM1 (full label) | hard | 111 | 0.441 [0.351, 0.541] | 0.336 | 15.9 | 0.516 | 0.316 (T = 7.930) | 0.516 | 7.803 | 1.042 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.000 | 0.082 (T = 8.428) | 0.000 | 0.000 | 0.000 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | standard | 72 | 0.708 [0.597, 0.819] | 0.311 | 57.7 | 0.219 | 0.100 (T = 8.428) | 0.221 | 4.352 | 0.508 |
| Qwen/Qwen3-4B-Instruct-2507 | First token (original demo's method) | hard | 111 | 0.432 [0.342, 0.532] | 0.336 | 14.5 | 0.454 | 0.266 (T = 8.428) | 0.461 | 7.225 | 0.986 |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | easy | 48 | 0.771 [0.646, 0.876] | 0.284 | 68.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | standard | 72 | 0.667 [0.556, 0.778] | 0.311 | 51.6 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-4B-Instruct-2507 | Normal generation | hard | 111 | 0.351 [0.270, 0.450] | 0.336 | 2.3 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | easy | 48 | 0.958 [0.896, 1.000] | 0.284 | 94.2 | 0.045 | 0.085 (T = 7.007) | 0.045 | 0.148 | 0.074 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | standard | 72 | 0.569 [0.458, 0.694] | 0.311 | 37.5 | 0.327 | 0.133 (T = 7.007) | 0.327 | 3.458 | 0.673 |
| Qwen/Qwen3-1.7B | MirethSTM1 (full label) | hard | 111 | 0.288 [0.207, 0.378] | 0.336 | 0.0 | 0.648 | 0.365 (T = 7.007) | 0.650 | 7.565 | 1.331 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | easy | 48 | 0.958 [0.896, 1.000] | 0.284 | 94.2 | 0.046 | 0.085 (T = 7.024) | 0.046 | 0.165 | 0.078 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | standard | 72 | 0.569 [0.458, 0.694] | 0.311 | 37.5 | 0.294 | 0.155 (T = 7.024) | 0.304 | 3.549 | 0.686 |
| Qwen/Qwen3-1.7B | First token (original demo's method) | hard | 111 | 0.297 [0.216, 0.378] | 0.336 | 0.0 | 0.567 | 0.314 (T = 7.024) | 0.575 | 6.998 | 1.221 |
| Qwen/Qwen3-1.7B | Normal generation | easy | 48 | 0.750 [0.625, 0.875] | 0.284 | 65.1 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | Normal generation | standard | 72 | 0.458 [0.333, 0.569] | 0.311 | 21.4 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-1.7B | Normal generation | hard | 111 | 0.153 [0.090, 0.225] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.004 | 0.103 (T = 2.702) | 0.004 | 0.004 | 0.000 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | standard | 72 | 0.694 [0.583, 0.806] | 0.311 | 55.6 | 0.187 | 0.150 (T = 2.702) | 0.203 | 1.161 | 0.480 |
| HuggingFaceTB/SmolLM3-3B | MirethSTM1 (full label) | hard | 111 | 0.396 [0.306, 0.486] | 0.336 | 9.1 | 0.359 | 0.187 (T = 2.702) | 0.365 | 2.151 | 0.851 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | easy | 48 | 1.000 [1.000, 1.000] | 0.284 | 100.0 | 0.006 | 0.119 (T = 2.545) | 0.006 | 0.006 | 0.000 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | standard | 72 | 0.667 [0.556, 0.778] | 0.311 | 51.6 | 0.119 | 0.086 (T = 2.545) | 0.167 | 1.178 | 0.479 |
| HuggingFaceTB/SmolLM3-3B | First token (original demo's method) | hard | 111 | 0.405 [0.315, 0.495] | 0.336 | 10.4 | 0.297 | 0.146 (T = 2.545) | 0.303 | 2.019 | 0.831 |
| HuggingFaceTB/SmolLM3-3B | Normal generation | easy | 48 | 0.938 [0.875, 1.000] | 0.284 | 91.3 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | Normal generation | standard | 72 | 0.667 [0.556, 0.778] | 0.311 | 51.6 | n/a | n/a | n/a | n/a | n/a |
| HuggingFaceTB/SmolLM3-3B | Normal generation | hard | 111 | 0.207 [0.135, 0.288] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | easy | 48 | 0.875 [0.791, 0.958] | 0.284 | 82.5 | 0.112 | 0.178 (T = 4.639) | 0.112 | 0.373 | 0.200 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | standard | 72 | 0.444 [0.333, 0.556] | 0.311 | 19.4 | 0.437 | 0.183 (T = 4.639) | 0.437 | 3.262 | 0.889 |
| Qwen/Qwen3-0.6B | MirethSTM1 (full label) | hard | 111 | 0.360 [0.279, 0.459] | 0.336 | 3.6 | 0.538 | 0.301 (T = 4.639) | 0.538 | 4.696 | 1.141 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | easy | 48 | 0.875 [0.791, 0.958] | 0.284 | 82.5 | 0.104 | 0.182 (T = 4.886) | 0.112 | 0.386 | 0.202 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | standard | 72 | 0.472 [0.361, 0.583] | 0.311 | 23.4 | 0.367 | 0.151 (T = 4.886) | 0.367 | 3.342 | 0.844 |
| Qwen/Qwen3-0.6B | First token (original demo's method) | hard | 111 | 0.351 [0.270, 0.450] | 0.336 | 2.3 | 0.480 | 0.250 (T = 4.886) | 0.475 | 4.430 | 1.086 |
| Qwen/Qwen3-0.6B | Normal generation | easy | 48 | 0.000 [0.000, 0.000] | 0.284 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | Normal generation | standard | 72 | 0.000 [0.000, 0.000] | 0.311 | 0.0 | n/a | n/a | n/a | n/a | n/a |
| Qwen/Qwen3-0.6B | Normal generation | hard | 111 | 0.000 [0.000, 0.000] | 0.336 | 0.0 | n/a | n/a | n/a | n/a | n/a |

## Latency

Batch 1, one request at a time, model loaded, after the warmup runs; each timed span is one `Engine.decide` or `baseline.generate` call with torch.cuda.synchronize() at both ends on CUDA. Normal generation is the same loaded model writing every answer as one JSON object with Hugging Face `generate` (greedy), so each speedup is against that library's decoding speed on this machine (the rate is in the Summary), not against an optimized inference server. p50 and p95 with linear interpolation; brackets: bootstrap 95% CI. With few runs p95 rests on the slowest one or two runs. Field-count settings use a different AG News article per run (topic choice plus rule-checked yes/no questions); scenarios repeat their own text. Checked answers: share of fields whose answer matches the gold topic or the rule. Bad fields: hallucinated or missing answers per generation run. Peak memory: the largest `torch.cuda.max_memory_allocated` of a timed run of that arm.

### Qwen/Qwen2.5-1.5B-Instruct

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: NVIDIA GeForce RTX 5070, 591.86, 42, 2677 MHz, 13801 MHz, 15.65 W; at end: NVIDIA GeForce RTX 5070, 591.86, 46, 2917 MHz, 13801 MHz, 98.04 W.

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MiB (MirethSTM1 / generation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 30 | 37.2 [35.9, 37.5] | 74.8 [37.6, 77.5] | 310 [309, 350] | 910 [422, 1463] | 8.3x | 166 | 9 | 0.77 / 0.70 | 0.1 | 2971 / 2985 |
| 5 fields | 5 | 30 | 35.7 [35.2, 36.3] | 43.0 [36.9, 67.5] | 1269 [1223, 1291] | 1736 [1362, 2352] | 35.6x | 285 | 32 | 0.71 / 0.61 | 0.0 | 2980 / 2995 |
| 10 fields | 10 | 30 | 34.9 [34.3, 35.1] | 56.6 [39.3, 66.0] | 2548 [2435, 2678] | 3284 [2828, 3773] | 73.0x | 432 | 63 | 0.72 / 0.72 | 0.0 | 2997 / 3005 |
| 20 fields | 20 | 30 | 48.1 [46.6, 49.6] | 50.9 [50.1, 52.2] | 5436 [5243, 5500] | 6161 [5652, 6507] | 113.1x | 741 | 134 | 0.67 / 0.66 | 0.6 | 3033 / 3034 |
| Support ticket triage (28 fields) | 28 | 20 | 96.4 [96.1, 96.7] | 97.3 [96.8, 97.3] | 8214 [8109, 8771] | 9444 [9148, 10142] | 85.2x | 1595 | 206 | n/a / n/a | 0.0 | 3130 / 3168 |
| Code change security review (28 fields) | 28 | 20 | 97.7 [97.6, 97.7] | 98.9 [97.8, 99.6] | 8796 [8587, 8944] | 9755 [9052, 9985] | 90.0x | 1667 | 216 | n/a / n/a | 0.0 | 3195 / 3172 |
| Incident triage with scores (20 fields) | 20 | 20 | 81.4 [81.3, 81.5] | 82.3 [81.7, 84.1] | 6122 [5921, 6517] | 6989 [6615, 7017] | 75.2x | 1328 | 149 | n/a / n/a | 0.0 | 3103 / 3122 |
| Request router, 255 queues (4 fields) | 4 | 20 | 209 [209, 209] | 210 [210, 211] | 1461 [1405, 1558] | 1698 [1601, 1967] | 7.0x | 3325 | 30 | n/a / n/a | 2.0 | 3550 / 3586 |

### Qwen/Qwen3-4B-Instruct-2507

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: NVIDIA GeForce RTX 5070, 591.86, 43, 2745 MHz, 13801 MHz, 32.02 W; at end: NVIDIA GeForce RTX 5070, 591.86, 54, 2902 MHz, 13801 MHz, 117.15 W.

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MiB (MirethSTM1 / generation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 30 | 54.6 [54.0, 55.4] | 62.3 [56.0, 72.8] | 486 [480, 532] | 602 [594, 698] | 8.9x | 166 | 8 | 0.83 / 0.83 | 0.0 | 7755 / 7831 |
| 5 fields | 5 | 30 | 67.4 [57.3, 67.8] | 70.7 [68.0, 75.4] | 2056 [1975, 2152] | 2438 [2331, 2479] | 30.5x | 285 | 32 | 0.65 / 0.61 | 0.0 | 7762 / 7857 |
| 10 fields | 10 | 30 | 76.0 [75.4, 82.5] | 83.7 [82.8, 89.1] | 3915 [3848, 4030] | 4303 [4197, 4342] | 51.5x | 432 | 63 | 0.74 / 0.70 | 0.0 | 7778 / 7888 |
| 20 fields | 20 | 30 | 114 [112, 119] | 121 [120, 124] | 8651 [8421, 8910] | 10145 [9309, 10432] | 75.8x | 741 | 133 | 0.67 / 0.68 | 0.0 | 7815 / 7988 |
| Support ticket triage (28 fields) | 28 | 20 | 271 [271, 272] | 272 [272, 272] | 16863 [16015, 17587] | 18962 [17951, 19102] | 62.2x | 1595 | 236 | n/a / n/a | 0.0 | 7913 / 8365 |
| Code change security review (28 fields) | 28 | 20 | 279 [279, 280] | 280 [280, 280] | 16990 [16696, 17374] | 17733 [17541, 18101] | 60.9x | 1667 | 248 | n/a / n/a | 1.0 | 7976 / 8377 |
| Incident triage with scores (20 fields) | 20 | 20 | 211 [210, 211] | 212 [211, 212] | 11175 [10452, 11863] | 12195 [11928, 12876] | 53.1x | 1328 | 149 | n/a / n/a | 0.0 | 7885 / 8261 |
| Request router, 255 queues (4 fields) | 4 | 20 | 615 [614, 615] | 616 [615, 616] | 2698 [2613, 2783] | 3055 [2790, 3444] | 4.4x | 3325 | 29 | n/a / n/a | 0.0 | 8336 / 9408 |

### Qwen/Qwen3-1.7B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: NVIDIA GeForce RTX 5070, 591.86, 41, 2670 MHz, 13801 MHz, 22.05 W; at end: NVIDIA GeForce RTX 5070, 591.86, 48, 2917 MHz, 13801 MHz, 71.02 W.

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MiB (MirethSTM1 / generation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 10 | 44.8 [44.3, 45.3] | 45.8 [45.1, 46.0] | 712 [705, 794] | 943 [725, 1064] | 15.9x | 181 | 16 | 0.70 / 0.60 | 0.1 | 3304 / 3355 |
| 5 fields | 5 | 10 | 43.5 [43.2, 44.0] | 44.3 [43.8, 44.5] | 1528 [1503, 1684] | 1853 [1610, 1899] | 35.2x | 316 | 32 | 0.68 / 0.68 | 0.0 | 3315 / 3371 |
| 10 fields | 10 | 10 | 42.8 [42.3, 52.3] | 68.2 [46.1, 77.6] | 3016 [2927, 3173] | 3388 [3023, 3515] | 70.5x | 483 | 63 | 0.74 / 0.77 | 0.0 | 3333 / 3389 |
| 20 fields | 20 | 10 | 52.7 [52.3, 56.3] | 66.0 [53.0, 71.1] | 6449 [6261, 6821] | 7419 [6539, 7848] | 122.5x | 832 | 133 | 0.67 / 0.72 | 0.0 | 3371 / 3444 |
| Support ticket triage (28 fields) | 28 | 20 | 111 [110, 111] | 112 [111, 112] | 10247 [9998, 10681] | 12106 [11096, 13270] | 92.7x | 1707 | 206 | n/a / n/a | 0.0 | 3468 / 3663 |
| Code change security review (28 fields) | 28 | 20 | 132 [131, 132] | 134 [132, 134] | 11150 [10996, 11584] | 12025 [11671, 12072] | 84.7x | 1779 | 219 | n/a / n/a | 0.0 | 3533 / 3669 |
| Incident triage with scores (20 fields) | 20 | 20 | 95.2 [94.8, 95.8] | 96.8 [95.8, 97.1] | 7613 [7333, 7836] | 8624 [7881, 8735] | 80.0x | 1408 | 149 | n/a / n/a | 0.0 | 3441 / 3593 |
| Request router, 255 queues (4 fields) | 4 | 20 | 248 [248, 249] | 249 [249, 250] | 1659 [1646, 1789] | 2028 [1956, 2405] | 6.7x | 3341 | 28 | n/a / n/a | 0.0 | 3889 / 4236 |

### HuggingFaceTB/SmolLM3-3B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: NVIDIA GeForce RTX 5070, 591.86, 42, 2677 MHz, 13801 MHz, 24.50 W; at end: NVIDIA GeForce RTX 5070, 591.86, 51, 2917 MHz, 13801 MHz, 98.70 W.

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MiB (MirethSTM1 / generation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 10 | 41.5 [41.0, 42.3] | 45.9 [42.0, 48.6] | 372 [372, 378] | 427 [374, 463] | 9.0x | 213 | 8 | 0.80 / 0.70 | 0.0 | 6006 / 6041 |
| 5 fields | 5 | 10 | 51.6 [48.3, 57.4] | 63.2 [52.2, 63.7] | 1551 [1485, 1755] | 1864 [1683, 1953] | 30.1x | 346 | 32 | 0.70 / 0.68 | 0.0 | 6015 / 6054 |
| 10 fields | 10 | 10 | 64.5 [64.2, 64.8] | 72.7 [64.6, 79.1] | 3069 [2856, 3382] | 3854 [3168, 4063] | 47.6x | 513 | 62 | 0.82 / 0.83 | 0.0 | 6027 / 6071 |
| 20 fields | 20 | 10 | 93.2 [89.9, 115] | 116 [94.1, 116] | 6135 [5978, 6346] | 7373 [6308, 8009] | 65.9x | 857 | 122 | 0.79 / 0.78 | 0.0 | 6051 / 6114 |
| Support ticket triage (28 fields) | 28 | 20 | 213 [213, 214] | 214 [214, 214] | 11677 [11231, 12207] | 13303 [12445, 13536] | 54.7x | 1727 | 216 | n/a / n/a | 0.0 | 6134 / 6309 |
| Code change security review (28 fields) | 28 | 20 | 215 [215, 216] | 216 [216, 217] | 12914 [12222, 13459] | 14078 [13525, 14703] | 60.0x | 1807 | 238 | n/a / n/a | 1.0 | 6189 / 6318 |
| Incident triage with scores (20 fields) | 20 | 20 | 167 [167, 168] | 168 [168, 168] | 8709 [8031, 9179] | 9991 [9345, 10801] | 52.1x | 1423 | 162 | n/a / n/a | 0.0 | 6110 / 6243 |
| Request router, 255 queues (4 fields) | 4 | 20 | 408 [408, 409] | 409 [409, 410] | 2430 [2265, 2666] | 3155 [2781, 3362] | 5.9x | 3369 | 35 | n/a / n/a | 0.0 | 6493 / 6859 |

### Qwen/Qwen3-0.6B

Device cuda:0, torch.bfloat16, NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, CUDA 12.8, transformers 5.18.0.
nvidia-smi (name, driver, temperature C, core clock, memory clock, power) at start: NVIDIA GeForce RTX 5070, 591.86, 40, 2707 MHz, 13801 MHz, 15.57 W; at end: NVIDIA GeForce RTX 5070, 591.86, 44, 2925 MHz, 13801 MHz, 65.23 W.

| Setting | Fields | Runs | MirethSTM1 p50 ms | MirethSTM1 p95 ms | Normal generation p50 ms | Normal generation p95 ms | Speedup at p50 | MirethSTM1 input tokens | Generated tokens | Checked answers right (MirethSTM1 / generation) | Generation: bad fields per run | Peak memory MiB (MirethSTM1 / generation) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 field | 1 | 10 | 43.8 [43.1, 44.1] | 56.2 [44.0, 66.0] | 1049 [1030, 1135] | 1202 [1088, 1218] | 24.0x | 181 | 22 | 0.70 / 0.00 | 1.0 | 1155 / 1209 |
| 5 fields | 5 | 10 | 42.9 [42.5, 43.6] | 43.7 [43.1, 43.8] | 1988 [1707, 2116] | 2258 [2019, 2337] | 46.4x | 316 | 40 | 0.60 / 0.44 | 0.0 | 1170 / 1222 |
| 10 fields | 10 | 10 | 42.9 [42.5, 48.1] | 59.1 [43.1, 64.0] | 3979 [3755, 4060] | 4302 [4055, 4439] | 92.8x | 483 | 81 | 0.59 / 0.40 | 2.0 | 1187 / 1243 |
| 20 fields | 20 | 10 | 42.8 [42.0, 44.1] | 49.2 [43.0, 53.4] | 7601 [7437, 7866] | 8150 [7715, 8281] | 177.8x | 832 | 156 | 0.61 / 0.49 | 1.9 | 1224 / 1295 |
| Support ticket triage (28 fields) | 28 | 20 | 63.7 [63.6, 63.9] | 64.4 [64.0, 64.4] | 13797 [13272, 14114] | 14784 [14178, 15481] | 216.6x | 1707 | 275 | n/a / n/a | 17.0 | 1321 / 1511 |
| Code change security review (28 fields) | 28 | 20 | 76.2 [76.2, 76.8] | 77.7 [76.8, 78.6] | 10894 [10676, 11667] | 12540 [11726, 13022] | 142.9x | 1779 | 219 | n/a / n/a | 20.0 | 1385 / 1517 |
| Incident triage with scores (20 fields) | 20 | 20 | 53.4 [53.4, 53.8] | 54.4 [53.9, 54.6] | 8693 [8506, 9040] | 9493 [9240, 10267] | 162.7x | 1408 | 178 | n/a / n/a | 20.0 | 1293 / 1443 |
| Request router, 255 queues (4 fields) | 4 | 20 | 156 [156, 156] | 157 [156, 157] | 2018 [1989, 2066] | 3019 [2185, 3115] | 13.0x | 3341 | 37 | n/a / n/a | 2.0 | 1739 / 2078 |

## Precision: the same rows in other dtypes

The MirethSTM1 arm on the evaluation rows that this run and bench/out/v2-<dtype>/ both hold. Agreement and score differences are against fp32 on the same rows, over every label's summed log-prob. Median ms is the per-row time of the accuracy run (one question per call).

### Qwen/Qwen2.5-1.5B-Instruct

Pooled T, each dtype fitted on the same 500 calibration rows per dataset: bf16 2.112, fp16 2.114, fp32 2.116.

| Dataset | n | Accuracy bf16 / fp16 / fp32 | ECE-15 at T = 1 | ECE-15 at own pooled T | Same answer as fp32: bf16 / fp16 | Mean (max) abs score difference from fp32: bf16 / fp16 | Median ms bf16 / fp16 / fp32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ag_news | 500 | 0.840 / 0.832 / 0.834 | 0.131 / 0.127 / 0.126 | 0.097 / 0.100 / 0.100 | 0.990 / 0.998 | 0.122 (0.81) / 0.017 (0.11) | 35.9 / 36.0 / 50.4 |
| banking77 | 500 | 0.498 / 0.494 / 0.488 | 0.226 / 0.230 / 0.236 | 0.150 / 0.147 / 0.150 | 0.966 / 0.994 | 0.171 (1.77) / 0.021 (0.30) | 78.8 / 69.7 / 236 |
| sst2 | 500 | 0.902 / 0.902 / 0.900 | 0.072 / 0.076 / 0.076 | 0.200 / 0.199 / 0.197 | 0.994 / 0.998 | 0.091 (0.44) / 0.011 (0.04) | 36.0 / 35.9 / 34.3 |
| sst2_choice | 500 | 0.920 / 0.920 / 0.920 | 0.075 / 0.072 / 0.073 | 0.043 / 0.040 / 0.040 | 1.000 / 1.000 | 0.069 (0.84) / 0.009 (0.09) | 35.9 / 36.0 / 34.4 |
| yelp | 500 | 0.484 / 0.482 / 0.482 | 0.273 / 0.275 / 0.275 | 0.141 / 0.133 / 0.130 | 0.990 / 1.000 | 0.075 (0.49) / 0.010 (0.09) | 35.3 / 35.9 / 74.6 |
