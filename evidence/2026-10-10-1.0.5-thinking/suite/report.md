| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 84s | drafted 71 tok/s vs undrafted 59 tok/s |
| concurrent | 12/12 | 100.0% | 62s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 18s | identical |
| multiturn | 8/8 | 100.0% | 14s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 3s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 12s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 4s | 2/2 byte-identical |
| max-tokens | 0/3 | 0.0% | 99s | 0/3 exact |
| tools | 57/60 | 95.0% | 80s | 'Set a timer for 5 minutes.': got [('set_timer', {'label': '5 minutes', 'seconds': 300})] |
| json | 30/30 | 100.0% | 30s | all valid |
| ifeval | 131/150 | 87.3% | 1149s | loose 131/150, strict 121/150, 150 covered prompts |
| gsm8k | 248/250 | 99.2% | 320s | misses at [12, 119] |
| mgsm | 226/240 | 94.2% | 479s | de 28/30, es 29/30, fr 27/30, ja 28/30, zh 29/30, ru 29/30, sw 29/30, bn 27/30 |
| mmlu | 308/342 | 90.1% | 1218s | 57 subjects x 6; weakest business_ethics 3/6, virology 3/6, college_chemistry 4/6 |
| repetition | 8/8 | 100.0% | 61s | repeated 4-grams 0.19% mean, max 0.92%; 0 replies hit max_tokens |
| long-context | 54/55 | 98.2% | 1159s | needles 14/15 to 200k (missed 200k@0.5), facts 10/10@25k, 10/10@51k, 10/10@102k, 10/10@200k, slowest prefill 564 tok/s |
| vision | 12/12 | 100.0% | 7s | all probes answered |
| snapshot | 12/12 | 100.0% | 27s | replies written to evidence/2026-10-10-1.0.5-thinking/suite/replies.json |
