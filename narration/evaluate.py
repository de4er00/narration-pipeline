"""Reproducible comparison: slot-filling baseline vs the full pipeline.

Both variants are scored by the same functions the pipeline uses for its own
checks, on the final list of frames each variant produces. The facts
collected for a topic are shared, so "numbers not in the facts" means the same
thing for both.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .baseline import write_baseline
from .cards import ChannelCard, VideoCard
from .checks import (check_narration, cta_follows_card, duplicate_frames,
                     ngram_repeats, paraphrased_neighbours)
from .llm import ChatClient, LLMError, estimate_usd
from .pipeline import MAX_REPAIRS, cut, write_script
from .research import Fact, collect, overstated, unsourced_numbers
from .storyboard import assemble
from .timing import Frame, Segmentation, TimingError, estimate_seconds

logger = logging.getLogger(__name__)

README_MARK = "<!-- EVAL RESULTS -->"
README_END = "<!-- /EVAL RESULTS -->"
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_ENDS_SENTENCE = re.compile(r"[.!?…][\"'”’)]*$")


@dataclass
class Topic:
    id: str
    channel: str
    title: str


def load_topics(path: Path) -> list[Topic]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Topic(id=str(t["id"]), channel=str(t["channel"]), title=str(t["title"]))
            for t in raw.get("topics", [])]


@dataclass
class Metrics:
    frames: int
    words: int
    duplicate_frames: int
    repeated_phrases: int
    paraphrased_neighbours: int
    numbers: int
    unsourced_numbers: int
    overstated_claims: int
    runtime_s: float
    runtime_error: float
    broken_sentences: int
    worst_drift_s: float
    problems: int
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0


def measure(frames: list[str], card: ChannelCard, facts: list[Fact], final_cta: str,
            *, ends_sentence: list[bool] | None = None) -> Metrics:
    """Score one variant's frames. Pure: no model, no network."""
    rate, pause = card.voice.syllables_per_second, card.voice.sentence_pause_s
    text = " ".join(frames)
    seconds = [estimate_seconds(t, rate=rate, pause=pause) for t in frames]
    if ends_sentence is None:
        ends_sentence = [bool(_ENDS_SENTENCE.search(t.strip())) for t in frames]
    grid = Segmentation(frames=[Frame(i, t, s, 0.0) for i, (t, s)
                                in enumerate(zip(frames, seconds), 1)],
                        target_seconds=card.frame_seconds)
    unsourced = unsourced_numbers(text, facts)
    stated = overstated(text, facts)
    flagged = [p for p in check_narration(frames, card, seconds) if not p.advisory]
    flagged += cta_follows_card(final_cta, card)
    runtime = estimate_seconds(text, rate=rate, pause=pause)
    return Metrics(
        frames=len(frames),
        words=len(text.split()),
        duplicate_frames=sum(len(p.at) - 1 for p in duplicate_frames(frames)),
        repeated_phrases=len(ngram_repeats(frames)),
        paraphrased_neighbours=len(paraphrased_neighbours(frames)),
        numbers=len(_NUMBER_RE.findall(text)),
        unsourced_numbers=len(unsourced),
        overstated_claims=len(stated),
        runtime_s=round(runtime, 1),
        runtime_error=round(runtime / card.target_seconds - 1, 4),
        broken_sentences=sum(1 for e in ends_sentence if not e),
        worst_drift_s=round(grid.worst_drift, 1),
        problems=len(flagged) + len(unsourced) + len(stated),
    )


def run_topic(topic: Topic, client: ChatClient, *, max_repairs: int = MAX_REPAIRS,
              research_model: str | None = None) -> dict:
    card = ChannelCard.load(topic.channel)
    video = VideoCard(video_id=topic.id, title=topic.title, channel=card)

    t0 = time.monotonic()
    kw = {"model": research_model} if research_model else {}
    res = collect(topic.title, card, client, **kw)
    video.key_facts = res.claims
    script = write_script(video, client, research=res, max_repairs=max_repairs)
    seg, owner = cut(script, card)
    board = assemble(video, script, seg, owner, research=res)
    pipe_seconds = time.monotonic() - t0

    t1 = time.monotonic()
    base = write_baseline(VideoCard(video_id=topic.id, title=topic.title, channel=card),
                          client)
    base_seconds = time.monotonic() - t1

    pipe = measure([f.narration for f in board.frames], card, res.facts,
                   script.final_cta, ends_sentence=[f.ends_sentence for f in board.frames])
    pipe.tokens_in = res.tokens_in + script.tokens_in
    pipe.tokens_out = res.tokens_out + script.tokens_out
    pipe.usd = round(res.usd + script.usd, 5)
    pipe.seconds = round(pipe_seconds, 1)

    last = base.frames[-1] if base.frames else ""
    b = measure(base.frames, card, res.facts, last)
    b.tokens_in, b.tokens_out = base.tokens_in, base.tokens_out
    b.usd, b.seconds = round(base.usd, 5), round(base_seconds, 1)

    return {
        "topic": asdict(topic),
        "setup": {"channel": card.id, "target_seconds": card.target_seconds,
                  "frames": card.frames},
        "facts": res.to_dict(),
        "baseline": {"frames": base.frames, "visuals": base.visuals,
                     "metrics": asdict(b)},
        "pipeline": {"script": script.to_dict(),
                     "frames": [f.narration for f in board.frames],
                     "metrics": asdict(pipe)},
    }


def estimate(n_topics: int, model: str, max_repairs: int = MAX_REPAIRS,
             ) -> tuple[float | None, float | None]:
    """(expected, upper bound) in USD. Expected assumes one repair per script."""
    def cost(repairs: float) -> float | None:
        return estimate_usd(model, {"script": n_topics, "repair": n_topics * repairs,
                                    "baseline": n_topics}, research_queries=n_topics)
    return cost(min(1, max_repairs)), cost(max_repairs)


def run_eval(topics: list[Topic], client: ChatClient, out_dir: Path, *,
             max_repairs: int = MAX_REPAIRS, research_model: str | None = None,
             stamp: str | None = None) -> tuple[list[dict], Path]:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, topic in enumerate(topics, 1):
        logger.info("eval %d/%d: %s", i, len(topics), topic.id)
        try:
            result = run_topic(topic, client, max_repairs=max_repairs,
                               research_model=research_model)
        except (LLMError, TimingError) as e:
            # Keep going: the topics already paid for are worth more than a
            # clean abort, and the failure is itself a result.
            logger.error("eval: %s failed: %s", topic.id, e)
            result = {"topic": asdict(topic), "error": str(e)}
        # Saved per topic, so an interrupted run keeps every finished topic.
        (run_dir / f"{topic.id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
    return results, run_dir


# --- report ------------------------------------------------------------------

_ROWS = [
    ("Frames", "frames", "{:.0f}"),
    ("Verbatim-duplicate frames", "duplicate_frames", "{:.1f}"),
    ("Four-word phrases used 3+ times", "repeated_phrases", "{:.1f}"),
    ("Neighbouring frames that paraphrase each other", "paraphrased_neighbours", "{:.1f}"),
    ("Numbers in the narration", "numbers", "{:.1f}"),
    ("Numbers not found in the collected facts", "unsourced_numbers", "{:.1f}"),
    ("Disputed or speculative facts stated as fact", "overstated_claims", "{:.1f}"),
    ("Runtime error vs target (absolute)", "runtime_error", "{:.1%}"),
    ("Frames ending mid-sentence", "broken_sentences", "{:.1f}"),
    ("Worst picture/voice drift on a fixed grid, s", "worst_drift_s", "{:.1f}"),
    ("Problems flagged by the checks", "problems", "{:.1f}"),
    ("Tokens in / out", "tokens", "{}"),
    ("Cost, USD", "usd", "{:.3f}"),
    ("Wall time, s", "seconds", "{:.0f}"),
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(results: list[dict]) -> dict[str, dict[str, float]]:
    ok = [r for r in results if "error" not in r]
    out: dict[str, dict[str, float]] = {}
    for variant in ("baseline", "pipeline"):
        ms = [r[variant]["metrics"] for r in ok]
        agg = {key: _mean([m[key] for m in ms]) for key in ms[0]} if ms else {}
        if ms:
            agg["runtime_error"] = _mean([abs(m["runtime_error"]) for m in ms])
        out[variant] = agg
    return out


def _cell(agg: dict, key: str, fmt: str) -> str:
    if not agg:
        return "-"
    if key == "tokens":
        return f"{agg['tokens_in']:,.0f} / {agg['tokens_out']:,.0f}"
    return fmt.format(agg[key])


def results_table(results: list[dict]) -> str:
    agg = summarise(results)
    n = sum(1 for r in results if "error" not in r)
    lines = [f"| Per script, mean over {n} topics | Baseline: one request, fixed "
             f"slots | Pipeline |", "|---|---:|---:|"]
    for label, key, fmt in _ROWS:
        lines.append(f"| {label} | {_cell(agg['baseline'], key, fmt)} | "
                     f"{_cell(agg['pipeline'], key, fmt)} |")
    return "\n".join(lines)


def _pair(b: dict, p: dict, key: str, fmt: str = "{}") -> str:
    return f"{fmt.format(b[key])} / {fmt.format(p[key])}"


def per_topic_table(results: list[dict]) -> str:
    lines = ["| Topic | Duplicate frames | Repeated phrases | Numbers not in facts "
             "| Runtime error | Problems | Cost, USD |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        title = r["topic"]["title"]
        if "error" in r:
            lines.append(f"| {title} | failed: {r['error'][:80]} | | | | | |")
            continue
        b, p = r["baseline"]["metrics"], r["pipeline"]["metrics"]
        cells = [_pair(b, p, "duplicate_frames"), _pair(b, p, "repeated_phrases"),
                 _pair(b, p, "unsourced_numbers"),
                 _pair(b, p, "runtime_error", "{:+.0%}"), _pair(b, p, "problems"),
                 _pair(b, p, "usd", "{:.3f}")]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(results: list[dict], *, model: str, research_model: str,
           run_dir: str, date: str) -> str:
    ok = [r for r in results if "error" not in r]
    total = sum(r[v]["metrics"]["usd"] for r in ok for v in ("baseline", "pipeline"))
    channels = sorted({r["setup"]["channel"] if "setup" in r
                       else Path(r["topic"]["channel"]).stem for r in results})
    shape = ""
    if ok:
        m, s = divmod(ok[0]["setup"]["target_seconds"], 60)
        shape = f" {m}:{s:02d} videos cut into {ok[0]['setup']['frames']} frames."
    return "\n".join([
        f"Eval run {date}: {len(results)} topics ({', '.join(channels)}).{shape} "
        f"Writing model `{model}`, facts from `{research_model}`. "
        f"Total cost ${total:.2f}.",
        "",
        results_table(results),
        "",
        "<details><summary>Per topic, baseline / pipeline</summary>",
        "",
        per_topic_table(results),
        "",
        "</details>",
        "",
        f"Raw outputs and metrics: `{run_dir}/`.",
    ]) + "\n"


def update_readme(readme: Path, block: str) -> None:
    """Put `block` right after the marker, replacing a previous insert."""
    text = readme.read_text(encoding="utf-8")
    if README_MARK not in text:
        raise ValueError(f"{readme} has no {README_MARK} marker")
    head, tail = text.split(README_MARK, 1)
    if README_END in tail:
        tail = tail.split(README_END, 1)[1]
    readme.write_text(f"{head}{README_MARK}\n\n{block.rstrip()}\n\n{README_END}{tail}",
                      encoding="utf-8")
