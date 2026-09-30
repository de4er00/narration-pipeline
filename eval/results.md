Eval run 2026-09-30: 9 topics (history, science, space). 6:00 videos cut into 60 frames. Writing model `openai/gpt-5.6-luna-pro`, facts from `perplexity/sonar`. Total cost $0.97.

| Per script, mean over 6 topics | Baseline: one request, fixed slots | Pipeline |
|---|---:|---:|
| Frames | 60 | 61 |
| Verbatim-duplicate frames | 0.0 | 0.0 |
| Four-word phrases used 3+ times | 0.0 | 0.0 |
| Neighbouring frames that paraphrase each other | 0.0 | 0.0 |
| Numbers in the narration | 0.0 | 0.0 |
| Numbers not found in the collected facts | 0.0 | 0.0 |
| Disputed or speculative facts stated as fact | 0.0 | 0.0 |
| Runtime error vs target (absolute) | 7.8% | 4.3% |
| Frames ending mid-sentence | 0.0 | 1.5 |
| Worst picture/voice drift on a fixed grid, s | 30.0 | 18.3 |
| Problems flagged by the checks | 3.2 | 0.0 |
| Tokens in / out | 31,807 / 31,301 | 90,888 / 80,339 |
| Cost, USD | 0.044 | 0.119 |
| Wall time, s | 166 | 406 |

<details><summary>Per topic, baseline / pipeline</summary>

| Topic | Duplicate frames | Repeated phrases | Numbers not in facts | Runtime error | Problems | Cost, USD |
|---|---|---|---|---|---|---|
| How did people start a fire before matches? | 0 / 0 | 0 / 0 | 0 / 0 | +4% / -5% | 3 / 0 | 0.045 / 0.179 |
| How did the first people cross the sea to Australia? | 0 / 0 | 0 / 0 | 0 / 0 | +9% / +4% | 4 / 0 | 0.043 / 0.124 |
| How did people survive an Ice Age winter without a house? | 0 / 0 | 0 / 0 | 0 / 0 | +2% / -6% | 2 / 0 | 0.040 / 0.065 |
| What would happen if you fell into Jupiter? | 0 / 0 | 0 / 0 | 0 / 0 | +12% / +2% | 3 / 0 | 0.042 / 0.070 |
| Why do we only ever see one side of the Moon? | 0 / 0 | 0 / 0 | 0 / 0 | +7% / -7% | 3 / 0 | 0.046 / 0.134 |
| What if you had a teaspoon of a neutron star? | 0 / 0 | 0 / 0 | 0 / 0 | +12% / +3% | 4 / 0 | 0.047 / 0.140 |
| How does an octopus change colour in a fraction of a second? | failed: OpenRouter refused: HTTP 401 {"error":{"message":"API key expired.","code":401," | | | | | |
| Why can't you tickle yourself? | failed: OpenRouter refused: HTTP 401 {"error":{"message":"API key expired.","code":401," | | | | | |
| What colours can bees see that we can't? | failed: OpenRouter refused: HTTP 401 {"error":{"message":"API key expired.","code":401," | | | | | |

</details>

Raw outputs and metrics: `eval/runs/20260930T164030Z/`.
