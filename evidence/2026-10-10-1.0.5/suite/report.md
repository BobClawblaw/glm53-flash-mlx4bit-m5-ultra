| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 52s | drafted 91 tok/s vs undrafted 73 tok/s |
| concurrent | 12/12 | 100.0% | 42s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 14s | identical |
| multiturn | 8/8 | 100.0% | 10s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 1s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 9s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 1s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 57/60 | 95.0% | 23s | 'Set a timer for 5 minutes.': got [('set_timer', {'label': '5 minutes', 'seconds': 300})] |
| json | 30/30 | 100.0% | 11s | all valid |
| ifeval | 96/150 | 64.0% | 733s | loose 96/150, strict 89/150, 150 covered prompts |
| gsm8k | 246/250 | 98.4% | 334s | misses at [21, 119, 154, 157] |
| mgsm | 217/240 | 90.4% | 474s | de 28/30, es 28/30, fr 27/30, ja 28/30, zh 26/30, ru 30/30, sw 26/30, bn 24/30 |
| mmlu | 292/342 | 85.4% | 401s | 57 subjects x 6; weakest global_facts 2/6, business_ethics 3/6, college_chemistry 3/6 |
| repetition | 8/8 | 100.0% | 64s | repeated 4-grams 0.18% mean, max 0.64%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 891s | needles 15/15 to 200k, facts 10/10@25k, 10/10@51k, 10/10@102k, 10/10@200k, slowest prefill 1633 tok/s |
| vision | 12/12 | 100.0% | 3s | all probes answered |
| snapshot | 12/12 | 100.0% | 23s | replies written to evidence/2026-10-10-1.0.5/suite/replies.json |
