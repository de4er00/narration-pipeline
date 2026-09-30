Eval run 2026-09-30, 3 rerun 2026-09-30: 9 topics (history, science, space), 9 finished. 6:00 videos, 60 frames. Writing model `openai/gpt-5.6-luna-pro`, facts from `perplexity/sonar`. Total cost $1.40.

| Per script, mean over 9 topics | Baseline: one request, 60 fixed slots | Pipeline |
|---|---:|---:|
| Frames, call to action included | 60 | 61 |
| Verbatim-duplicate frames | 0.0 | 0.0 |
| Four-word phrases used 3+ times | 0.0 | 0.0 |
| Neighbouring frames that paraphrase each other | 0.0 | 0.0 |
| Figures in the narration, digits or words | 0.1 | 2.4 |
| Figures not found in the collected facts | 0.1 | 0.0 |
| Disputed or speculative facts stated as fact | 0.0 | 0.0 |
| Runtime error vs target (absolute) | 9.7% | 4.3% |
| Frames ending at a clause boundary (comma, colon, dash) | 0.0 | 1.9 |
| Frames ending mid-clause | 0.0 | 0.0 |
| Worst picture/voice drift on a fixed grid, s | 36.1 | 17.5 |
| Problems flagged by the checks | 3.7 | 0.1 |
| Tokens in / out | 30,823 / 30,396 | 86,493 / 76,508 |
| Cost, USD | 0.043 | 0.113 |
| Wall time, s | 159 | 388 |

Both variants are scored on their whole narration, call to action included: the pipeline speaks it as an extra frame after its 60 cut frames, the baseline inside its 60 slots. The pipeline's tokens and cost include the facts query. Figures are numbers in digits or words other than 0-10, 100 and 1,000.

<details><summary>Per topic, baseline / pipeline</summary>

| Topic | Duplicate frames | Repeated phrases | Figures not in facts | Mid-clause breaks | Runtime error | Problems | Cost, USD |
|---|---|---|---|---|---|---|---|
| How did people start a fire before matches? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +4% / -5% | 3 / 0 | 0.045 / 0.179 |
| How did the first people cross the sea to Australia? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +9% / +4% | 4 / 0 | 0.043 / 0.124 |
| How did people survive an Ice Age winter without a house? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +2% / -6% | 2 / 0 | 0.040 / 0.065 |
| What would happen if you fell into Jupiter? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +12% / +2% | 3 / 0 | 0.042 / 0.070 |
| Why do we only ever see one side of the Moon? | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 | +7% / -7% | 4 / 0 | 0.046 / 0.134 |
| What if you had a teaspoon of a neutron star? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +12% / +3% | 4 / 0 | 0.047 / 0.140 |
| How does an octopus change colour in a fraction of a second? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +15% / +3% | 4 / 0 | 0.039 / 0.116 |
| Why can't you tickle yourself? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +9% / +9% | 5 / 1 | 0.041 / 0.062 |
| What colours can bees see that we can't? | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | +16% / -1% | 4 / 0 | 0.040 / 0.129 |

</details>

Raw outputs and metrics: `eval/runs/20260930T164030Z/`.
