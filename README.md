# narration-pipeline

Writes the narration for six-minute illustrated explainer videos and cuts it into 60 frames. Facts come first, with a confidence status; a model writes prose to a word budget measured from the narrator's voice; about 25 checks in code send the script back with located problems; a dynamic program cuts the text into frames by speech duration. An optional second stage writes the image and animation prompts.

I built it in production for a media company with several explainer channels on YouTube. This repository is the generator extracted from that code, with the channels replaced by neutral examples (history, space, science).

## Results

In short: on today's model the pipeline's gain is timing and grounding, not duplicates. It lands within 4.3% of the target runtime instead of 9.7% and every figure it uses traces to the collected facts, at 2.6 times the cost. The duplicate frames that started the design no longer appear with one request either. Details below.

Measured in production; the production scripts are not published. The first comparison can be rerun on public topics with `python -m narration eval` (see [Reproduce](#reproduce)).

| | Before | After |
|---|---|---|
| Verbatim-duplicate frames per script | 15-19 of 80 (36 scripts, one request with 80 fixed frame slots) | 0 (prose first, frames cut by code, checks with repairs) |
| Concrete numbers in a script | 0 (same video, same model, no facts on input) | 15 (with facts on input) |
| Fact collection, one query | $0.337, 201 s, 1,828,447 input tokens (writing model with a web-search plugin) | $0.006, 7 s, 179 input tokens (`perplexity/sonar`), about 60x cheaper |

<!-- EVAL RESULTS -->

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

<!-- /EVAL RESULTS -->

**What the rerun shows.** The failure that started this design did not reproduce. With today's writing model, one request with fixed slots gave no duplicate frames on any of the nine topics, at 60 slots or at 80, the format the production numbers were measured on ([80-slot baseline run](eval/runs/20260930T185156Z/results.md)). I can't separate how much of that is the newer model and how much is the production templates, which were longer than the baseline prompt here. What the pipeline still buys on these topics is timing and grounding: the runtime lands within 4.3% of the target instead of 9.7%, picture and voice drift apart by at most 17.5 s instead of 36.1 s, and each script carries 2.4 figures, all traceable to the collected facts, against 0.1 in the baseline. It costs 2.6 times as much and takes 2.4 times as long.

The run also found a bug. A script written for a narrator spells numbers out, and the check for figures missing from the facts only read digits, so it never fired. It now reads "about twenty kilometers" as 20 and "nineteen sixty-nine" as 1969 (`numbers.py`).

## How it works

```
topic ─► facts with statuses ─► prose to a word budget ─► checks ──► cut into frames ─► [visuals ─► prompts] ─► JSON + Markdown
                                        ▲                   │
                                        └── repair (max 2) ◄┘
```

**Frames are cut by code.** The old approach asked the model for 80 frames in fixed slots. As soon as a model gets a form with N cells, it fills the form: repeated lines, filler phrases, paraphrases of the previous frame. Here the model writes continuous chapters to a word budget, and code decides where frames begin and end (`pipeline.py`).

**Facts with a status** (`research.py`). A search model returns six facts, each marked `confirmed`, `reported`, `disputed` or `speculation`, and each status comes with a rule for how firmly the script may state it. Two checks use the facts afterwards: every number in the script must trace to a fact (a rounding of a known value passes, a more precise figure does not, years must match exactly), and a disputed or speculative fact must not be stated flatly. I use Sonar here because it searches on the provider side and returns a summary with links; a web plugin on the writing model pulled whole pages into the context and cost 60 times more for facts of the same quality.

**Checks and repairs** (`checks.py`, `pipeline.py`). Every check exists because the defect shipped at least once: verbatim duplicates, repeated openings and four-word phrases, neighbours that retell each other, a hook that opens on a caveat or a citation, written rather than spoken register, a flat rhythm with no short sentences, abstract nouns, semicolons and long lists, an ending about the past instead of the viewer, a call to action that turned into advice, production words spoken aloud, and the length against the voice's budget. When a check fails, the whole script goes back to the model once or twice with the exact problems and their positions. Every version is scored and the one with the fewest problems ships, not the last one: in five cases out of five, a repair for length introduced new problems with the rhythm or the call to action. Notes that the text cannot fix (no facts found) are shown but never trigger a paid repair.

**Speech time from syllables, per voice** (`timing.py`). Duration is `syllables / rate + sentences * pause`, with both parameters measured per voice from two recordings. Across five production voices the rate ranged from 3.66 to 6.30 syllables per second; the same 1,420 words ran 8.0 minutes with one voice and 13.0 with another. So the word budget in the prompt is computed from the voice, not fixed.

**An exact cut** (`timing.py`). Each chapter gets frames in proportion to its duration, capped by how many pieces it can physically yield. Inside a chapter, a dynamic program splits sentences into exactly N frames, minimising the squared deviation from the frame length plus a penalty for ending a frame mid-sentence. A greedy pass gives almost the same total drift; the DP is there because it never breaks a sentence it does not have to, halves the spread of frame lengths, and works per chapter. Abbreviations, titles and names like "the Wow! Signal" do not end sentences.

**Pictures and prompts** (`visuals.py`, `frame_prompts.py`). Optional stage 2 decides per frame what is seen, whether the recurring character is in it, the shot size and one motion. The prompts are then assembled in code from those decisions: the reference sheets are attached and named by position, the character's looks are never described in words (on a test frame, a 4,282-character prompt matched the sheet's style worse than a 702-character one), and the animation prompt describes one action at one speed with a barely perceptible push-in, because in 77 clips a locked camera jittered in 30-56% of cases and a push-in in 10%.

## Reproduce

The tests run offline with a scripted fake model:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The eval needs an [OpenRouter](https://openrouter.ai) key and spends real money:

```bash
export OPENROUTER_API_KEY=...                    # PowerShell: $env:OPENROUTER_API_KEY = "..."
python -m narration eval                         # prints the estimate, spends nothing
python -m narration eval --yes --readme README.md
```

For each of the 9 topics in `eval/topics.yaml` it collects facts once, then runs:

- **baseline**: one request for 60 narration lines in fixed 6-second slots, with the same channel card, word budget and originality rules written into the prompt, the way the production templates had them. No facts, no checks, no repairs.
- **pipeline**: facts, prose, checks and up to two repairs, then the cut.

Both outputs are scored on their final frames by the same functions: duplicates, repeated phrases, paraphrased neighbours, numbers not in the facts, overstated claims, runtime error against the target, frames ending at a clause boundary and frames ending mid-clause, worst drift against a fixed 6-second grid, the number of problems the checks flag, tokens, cost and wall time. At the default model's list price the command estimates about $1.48, at most $2.31, prints that first and does nothing without `--yes`. The estimate is calibrated on the first run's token counts; the writing model spends most of its output on reasoning. The pipeline's tokens and cost include the facts query; the baseline's do not. It writes `eval/results.md`, raw outputs per topic under `eval/runs/<timestamp>/`, and with `--readme` puts the table right below the production numbers above.

Other flags: `--only id,id` runs a subset of topics; `--merge eval/runs/<stamp>` runs failed or missing topics into an existing run (three topics of the first run hit an expired key and were completed this way); `--rescore eval/runs/<stamp>` recomputes every metric from the saved outputs without calling a model; `--frames 80` changes the number of frames at the same 6 seconds per frame; `--baseline-only` skips facts and the pipeline (about $0.53 for nine topics at 80 frames).

## Usage

```bash
python -m narration research "Why do we only ever see one side of the Moon?" --channel space
python -m narration write "Why do we only ever see one side of the Moon?" --channel space --visuals
python -m narration check out/<id>/script.json --channel space --facts out/<id>/facts.json
python -m narration cut my_script.txt --channel history
```

`write` runs facts, the script with repairs and the cut, and with `--visuals` also stage 2. It prints an estimate first (about $0.20, at most $0.30 for one video with visuals), saves `facts.json` and `script.json` before anything that can fail, then writes the storyboard as JSON and Markdown. `check` and `cut` are free and also take a plain text file with one chapter per paragraph. A channel is a YAML card (`narration/channels/`): the role and direction for the writer, the recurring character and its description, the call to action, the reference sheet names and the measured pace of the voice. Pass a path to use your own.

Models are configurable with `--model` and `--research-model`; the defaults are `openai/gpt-5.6-luna-pro` for writing (strict JSON schema output, high reasoning effort) and `perplexity/sonar` for facts. There is deliberately no fallback model: a refusal or a timeout stops the run instead of silently switching to something else.

## Layout

```
narration/
  cards.py          channel and video cards, word budget per voice
  research.py       stage 0: facts with statuses, unsourced-number and overstatement checks
  script.py         stage 1 prompts: writing and repair
  pipeline.py       the writing loop (checks, repairs, best version) and the cut
  checks.py         deterministic checks
  timing.py         syllable-based duration, per-voice pace, exact partition
  visuals.py        stage 2: one visual decision per frame
  frame_prompts.py  image and animation prompts assembled in code
  storyboard.py     call-to-action frame, JSON and Markdown output
  baseline.py       the one-request, fixed-slot approach, for the eval
  evaluate.py       metrics, eval harness, report
  llm.py            OpenRouter client, prices and estimates
  cli.py
  channels/         history, space, science
eval/topics.yaml
tests/              offline; a fake model replays scripted replies
```

## Limitations

- The checks measure form, not truth. They catch repetition, figures that are not in the collected facts and claims stated more firmly than their status allows, but not a wrong fact from the search model or a clumsy sentence. In production a person still read every script before it went on.
- The eval scores both variants with the pipeline's own checks, and the pipeline repairs against those checks, so the "problems flagged" row favours it by construction and mixes real defects with style rules. The rows for duplicates, figures not in the facts, mid-clause breaks, drift and runtime error are counts of defects that do not depend on my style rules. It is one run per topic on nine topics, not a benchmark.
- The baseline prompt is my reconstruction of the production templates, not the templates themselves, so the eval cannot say why the production duplicates happened.
- A repair rewrites the whole script with a located list of problems; it does not touch only the affected chapters, so a repair can still disturb text that was fine. Keeping the best-scoring version limits the damage.
- The syllable counter is an English vowel-group heuristic, and the pace values in the cards were measured on specific production voices. A different voice needs its own two recordings.
- English narration only.

## What I'd do next

- Find what actually caused the production duplicates: run the original templates against the current model and against the model of that time, and keep whichever part of the pipeline the answer still justifies.
- Repair only the chapters a local problem points to and keep the rest unchanged, falling back to a whole-script repair for global problems such as length or rhythm.
- Add a blind preference test, human or model-judged, on top of the counts: the eval does not measure whether a script is interesting.
- Fit the pace of a new voice automatically from rendered audio instead of by hand.
- Run each topic several times and report the spread.
- Use a pronunciation dictionary for names and numbers in the syllable count.

## License

MIT
