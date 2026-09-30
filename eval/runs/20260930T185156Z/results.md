Eval run 2026-09-30: 9 topics (history, science, space), 9 finished. Baseline only: one request per topic for 80 fixed slots (8:00), model `openai/gpt-5.6-luna-pro`. Total cost $0.45.

| Per script, mean over 9 topics | Baseline: one request, 80 fixed slots |
|---|---:|
| Frames, call to action included | 80 |
| Verbatim-duplicate frames | 0.0 |
| Four-word phrases used 3+ times | 0.0 |
| Neighbouring frames that paraphrase each other | 0.0 |
| Figures in the narration, digits or words | 0.1 |
| Runtime error vs target (absolute) | 9.1% |
| Frames ending at a clause boundary (comma, colon, dash) | 0.0 |
| Frames ending mid-clause | 0.0 |
| Worst picture/voice drift on a fixed grid, s | 44.9 |
| Problems flagged by the checks | 3.7 |
| Tokens in / out | 34,710 / 35,709 |
| Cost, USD | 0.050 |
| Wall time, s | 163 |

The call to action sits inside the last slots. Figures are numbers in digits or words other than 0-10, 100 and 1,000.

<details><summary>Per topic</summary>

| Topic | Duplicate frames | Repeated phrases | Mid-clause breaks | Runtime error | Problems | Cost, USD |
|---|---|---|---|---|---|---|
| How did people start a fire before matches? | 0 | 0 | 0 | +5% | 3 | 0.047 |
| How did the first people cross the sea to Australia? | 0 | 0 | 0 | +4% | 3 | 0.055 |
| How did people survive an Ice Age winter without a house? | 0 | 0 | 0 | +5% | 2 | 0.041 |
| What would happen if you fell into Jupiter? | 0 | 0 | 0 | +3% | 3 | 0.049 |
| Why do we only ever see one side of the Moon? | 0 | 0 | 0 | +8% | 3 | 0.055 |
| What if you had a teaspoon of a neutron star? | 0 | 0 | 0 | +16% | 6 | 0.045 |
| How does an octopus change colour in a fraction of a second? | 0 | 0 | 0 | +9% | 5 | 0.053 |
| Why can't you tickle yourself? | 0 | 0 | 0 | +14% | 4 | 0.053 |
| What colours can bees see that we can't? | 0 | 0 | 0 | +17% | 4 | 0.050 |

</details>

Raw outputs and metrics: `eval/runs/20260930T185156Z/`.
