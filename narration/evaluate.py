"""Reproducible comparison: slot-filling baseline vs the full pipeline.

Both variants are scored by the functions the pipeline uses for its own checks,
on the final frames each produces, call to action included. The facts collected
for a topic are shared, so "figures not in the facts" means the same for both.
Scores are always recomputed from the saved raw outputs, so a run can be
re-scored after a metric changes without calling a model.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .baseline import write_baseline
from .cards import DEFAULT_FRAMES, ChannelCard, VideoCard
from .checks import (check_narration, cta_follows_card, duplicate_frames,
                     ngram_repeats, paraphrased_neighbours)
from .llm import (DEFAULT_MODEL, EXPECTED_REPAIRS, RESEARCH_MODEL, ChatClient, LLMError,
                  estimate_usd)
from .numbers import figures
from .pipeline import MAX_REPAIRS, cut, write_script
from .research import Fact, Research, collect, overstated, unsourced_numbers
from .storyboard import assemble
from .timing import Frame, Segmentation, TimingError, estimate_seconds

logger = logging.getLogger(__name__)

README_MARK = "<!-- EVAL RESULTS -->"
README_END = "<!-- /EVAL RESULTS -->"
FULL, BASELINE_ONLY = "full", "baseline-only"
_CLOSERS = "\"'”’)"


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
class Settings:
    """What a run did. `frames=None` keeps each channel card's own count."""
    mode: str = FULL
    frames: int | None = None
    model: str = DEFAULT_MODEL
    research_model: str = RESEARCH_MODEL
    max_repairs: int = MAX_REPAIRS
    started: str = ""
    merges: list[dict] = field(default_factory=list)

    @property
    def variants(self) -> tuple[str, ...]:
        return ("baseline",) if self.mode == BASELINE_ONLY else ("baseline", "pipeline")

    @property
    def is_main(self) -> bool:
        """The standard comparison, the one eval/results.md and the README show."""
        return self.mode == FULL and self.frames in (None, DEFAULT_FRAMES)

    def same_experiment(self, other: "Settings") -> bool:
        return (self.mode, self.frames or DEFAULT_FRAMES, self.model) == \
               (other.mode, other.frames or DEFAULT_FRAMES, other.model)


def channel_card(channel: str, frames: int | None = None,
                 target_seconds: int | None = None) -> ChannelCard:
    """The card with another frame count at the same frame length if asked."""
    card = ChannelCard.load(channel)
    if frames is None or frames == card.frames:
        return card
    return replace(card, frames=frames,
                   target_seconds=target_seconds or round(frames * card.frame_seconds))


@dataclass
class Metrics:
    frames: int
    words: int
    duplicate_frames: int
    repeated_phrases: int
    paraphrased_neighbours: int
    figures: int
    unsourced_numbers: int | None
    overstated_claims: int | None
    runtime_s: float
    runtime_error: float
    clause_breaks: int
    mid_clause_breaks: int
    worst_drift_s: float
    problems: int
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0


def frame_ending(text: str) -> str:
    """"sentence", "clause" (comma, colon, semicolon, dash) or "mid"."""
    end = text.strip().rstrip(_CLOSERS)
    if end.endswith((".", "!", "?", "…")):
        return "sentence"
    if end.endswith((",", ";", ":", "—", "–", "-")):
        return "clause"
    return "mid"


def measure(frames: list[str], card: ChannelCard, facts: list[Fact] | None,
            final_cta: str) -> Metrics:
    """Score one variant's frames. Pure: no model, no network.

    Without facts (a baseline-only run) the fact-based metrics are None.
    """
    rate, pause = card.voice.syllables_per_second, card.voice.sentence_pause_s
    text = " ".join(frames)
    seconds = [estimate_seconds(t, rate=rate, pause=pause) for t in frames]
    grid = Segmentation(frames=[Frame(i, t, s, 0.0) for i, (t, s)
                                in enumerate(zip(frames, seconds), 1)],
                        target_seconds=card.frame_seconds)
    unsourced = unsourced_numbers(text, facts) if facts is not None else None
    stated = overstated(text, facts) if facts is not None else None
    flagged = [p for p in check_narration(frames, card, seconds) if not p.advisory]
    flagged += cta_follows_card(final_cta, card)
    runtime = estimate_seconds(text, rate=rate, pause=pause)
    endings = [frame_ending(t) for t in frames]
    return Metrics(
        frames=len(frames),
        words=len(text.split()),
        duplicate_frames=sum(len(p.at) - 1 for p in duplicate_frames(frames)),
        repeated_phrases=len(ngram_repeats(frames)),
        paraphrased_neighbours=len(paraphrased_neighbours(frames)),
        figures=len(figures(text)),
        unsourced_numbers=None if unsourced is None else len(unsourced),
        overstated_claims=None if stated is None else len(stated),
        runtime_s=round(runtime, 1),
        runtime_error=round(runtime / card.target_seconds - 1, 4),
        clause_breaks=endings.count("clause"),
        mid_clause_breaks=endings.count("mid"),
        worst_drift_s=round(grid.worst_drift, 1),
        problems=len(flagged) + len(unsourced or []) + len(stated or []),
    )


# --- running -----------------------------------------------------------------

def run_topic(topic: Topic, client: ChatClient, settings: Settings) -> dict:
    card = channel_card(topic.channel, settings.frames)
    result: dict = {"topic": asdict(topic),
                    "setup": {"channel": card.id, "target_seconds": card.target_seconds,
                              "frames": card.frames}}
    res = None
    if settings.mode == FULL:
        video = VideoCard(video_id=topic.id, title=topic.title, channel=card)
        t0 = time.monotonic()
        res = collect(topic.title, card, client, model=settings.research_model)
        video.key_facts = res.claims
        script = write_script(video, client, research=res,
                              max_repairs=settings.max_repairs)
        seg, owner = cut(script, card)
        board = assemble(video, script, seg, owner, research=res)
        result["facts"] = res.to_dict()
        result["pipeline"] = {
            "script": script.to_dict(), "frames": [f.narration for f in board.frames],
            "cost": {"tokens_in": res.tokens_in + script.tokens_in,
                     "tokens_out": res.tokens_out + script.tokens_out,
                     "usd": round(res.usd + script.usd, 5),
                     "seconds": round(time.monotonic() - t0, 1)}}

    t1 = time.monotonic()
    base = write_baseline(VideoCard(video_id=topic.id, title=topic.title, channel=card),
                          client)
    result["baseline"] = {
        "frames": base.frames, "visuals": base.visuals,
        "cost": {"tokens_in": base.tokens_in, "tokens_out": base.tokens_out,
                 "usd": round(base.usd, 5), "seconds": round(time.monotonic() - t1, 1)}}
    return rescore(result)


def estimate(n_topics: int, model: str, max_repairs: int = MAX_REPAIRS, *,
             frames: int = DEFAULT_FRAMES, baseline_only: bool = False,
             ) -> tuple[float | None, float | None]:
    """(expected, high) in USD: facts + baseline + writing x (1 + repairs).

    Expected uses mean call sizes and the observed repair rate; high uses the
    largest calls seen and every repair spent.
    """
    scale = frames / DEFAULT_FRAMES

    def cost(repairs: float, high: bool) -> float | None:
        if baseline_only:
            return estimate_usd(model, {"baseline": n_topics}, high=high, scale=scale)
        return estimate_usd(model, {"baseline": n_topics,
                                    "writing": n_topics * (1 + repairs)},
                            research_queries=n_topics, high=high, scale=scale)

    return cost(min(EXPECTED_REPAIRS, max_repairs), False), cost(max_repairs, True)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_eval(topics: list[Topic], client: ChatClient, run_dir: Path,
             settings: Settings) -> list[dict]:
    """Run `topics` into `run_dir`. An existing run is merged into: its topic
    files are replaced one by one and the merge is recorded in run.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = read_settings(run_dir)
    if saved is not None:
        settings = saved
    if saved is not None or load_results(run_dir):
        # Runs made before run.json existed are dated by their directory name.
        settings.started = settings.started or stamp_date(run_dir)
        settings.merges.append({"at": _now(), "topics": [t.id for t in topics]})
    else:
        settings.started = settings.started or _now()
    write_settings(run_dir, settings)
    results = []
    for i, topic in enumerate(topics, 1):
        logger.info("eval %d/%d: %s", i, len(topics), topic.id)
        try:
            result = run_topic(topic, client, settings)
        except (LLMError, TimingError) as e:
            # Keep going: the topics already paid for are worth more than a
            # clean abort, and the failure is itself a result.
            logger.error("eval: %s failed: %s", topic.id, e)
            result = {"topic": asdict(topic), "error": str(e)}
        # Saved per topic, so an interrupted run keeps every finished topic.
        (run_dir / f"{topic.id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
    return results


# --- saved runs --------------------------------------------------------------

def read_settings(run_dir: Path) -> Settings | None:
    path = run_dir / "run.json"
    if not path.is_file():
        return None
    return Settings(**json.loads(path.read_text(encoding="utf-8")))


def write_settings(run_dir: Path, settings: Settings) -> None:
    (run_dir / "run.json").write_text(json.dumps(asdict(settings), indent=2),
                                      encoding="utf-8")


def load_results(run_dir: Path, order: list[str] | None = None) -> list[dict]:
    """Every topic file of a run, in `order` first, then by name."""
    by_id = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.name in ("run.json", "summary.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id[data["topic"]["id"]] = data
    rank = {tid: i for i, tid in enumerate(order or [])}
    return [by_id[k] for k in sorted(by_id, key=lambda k: (rank.get(k, len(rank)), k))]


def needs_run(run_dir: Path, topic_id: str) -> bool:
    path = run_dir / f"{topic_id}.json"
    return not path.is_file() or "error" in json.loads(path.read_text(encoding="utf-8"))


def stamp_date(run_dir: Path) -> str:
    m = re.match(r"(\d{4})(\d{2})(\d{2})T", run_dir.name)
    return f"{m[1]}-{m[2]}-{m[3]}" if m else run_dir.name


def _cost(variant: dict) -> dict:
    # Early runs kept cost inside the stored metrics.
    src = variant.get("cost") or variant.get("metrics") or {}
    return {k: src.get(k, 0) for k in ("tokens_in", "tokens_out", "usd", "seconds")}


def rescore(result: dict) -> dict:
    """Fresh metrics from the raw frames of a saved result."""
    if "error" in result:
        return result
    out = dict(result)
    setup = result["setup"]
    card = channel_card(result["topic"]["channel"], setup["frames"],
                        setup["target_seconds"])
    facts = Research.from_dict(result["facts"]).facts if "facts" in result else None
    if "pipeline" in result:
        p = result["pipeline"]
        m = measure(p["frames"], card, facts, p["script"]["final_cta"])
        out["pipeline"] = {**p, "metrics": {**asdict(m), **_cost(p)}}
    b = result["baseline"]
    m = measure(b["frames"], card, facts, b["frames"][-1] if b["frames"] else "")
    out["baseline"] = {**b, "metrics": {**asdict(m), **_cost(b)}}
    return out


# --- report ------------------------------------------------------------------

_ROWS = [
    ("Frames, call to action included", "frames", "{:.0f}"),
    ("Verbatim-duplicate frames", "duplicate_frames", "{:.1f}"),
    ("Four-word phrases used 3+ times", "repeated_phrases", "{:.1f}"),
    ("Neighbouring frames that paraphrase each other", "paraphrased_neighbours", "{:.1f}"),
    ("Figures in the narration, digits or words", "figures", "{:.1f}"),
    ("Figures not found in the collected facts", "unsourced_numbers", "{:.1f}"),
    ("Disputed or speculative facts stated as fact", "overstated_claims", "{:.1f}"),
    ("Runtime error vs target (absolute)", "runtime_error", "{:.1%}"),
    ("Frames ending at a clause boundary (comma, colon, dash)", "clause_breaks", "{:.1f}"),
    ("Frames ending mid-clause", "mid_clause_breaks", "{:.1f}"),
    ("Worst picture/voice drift on a fixed grid, s", "worst_drift_s", "{:.1f}"),
    ("Problems flagged by the checks", "problems", "{:.1f}"),
    ("Tokens in / out", "tokens", "{}"),
    ("Cost, USD", "usd", "{:.3f}"),
    ("Wall time, s", "seconds", "{:.0f}"),
]


def failure_note(error: str) -> str:
    if re.search(r"HTTP \d{3}|OpenRouter|attempts|network|API key", error):
        return "not run: API error"
    if "frames are needed" in error or "too short" in error:
        return "failed: script too short to cut"
    return "failed: see the run file"


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise(results: list[dict], variants: tuple[str, ...] = ("baseline", "pipeline"),
              ) -> dict[str, dict[str, float | None]]:
    ok = [r for r in results if "error" not in r]
    out: dict[str, dict[str, float | None]] = {}
    for variant in variants:
        ms = [r[variant]["metrics"] for r in ok]
        agg = {key: _mean([m[key] for m in ms]) for key in ms[0]} if ms else {}
        if ms:
            agg["runtime_error"] = _mean([abs(m["runtime_error"]) for m in ms])
        out[variant] = agg
    return out


def _cell(agg: dict, key: str, fmt: str) -> str:
    if key == "tokens":
        if agg.get("tokens_in") is None:
            return "-"
        return f"{agg['tokens_in']:,.0f} / {agg['tokens_out']:,.0f}"
    value = agg.get(key)
    return "-" if value is None else fmt.format(value)


def _heading(variant: str, frames: int) -> str:
    return (f"Baseline: one request, {frames} fixed slots" if variant == "baseline"
            else "Pipeline")


def results_table(results: list[dict], variants: tuple[str, ...], frames: int) -> str:
    agg = summarise(results, variants)
    n = sum(1 for r in results if "error" not in r)
    lines = [f"| Per script, mean over {n} topics | "
             + " | ".join(_heading(v, frames) for v in variants) + " |",
             "|---|" + "---:|" * len(variants)]
    for label, key, fmt in _ROWS:
        if len(variants) == 1 and key in ("unsourced_numbers", "overstated_claims"):
            continue
        lines.append(f"| {label} | "
                     + " | ".join(_cell(agg[v], key, fmt) for v in variants) + " |")
    return "\n".join(lines)


_TOPIC_COLUMNS = [
    ("Duplicate frames", "duplicate_frames", "{}"),
    ("Repeated phrases", "repeated_phrases", "{}"),
    ("Figures not in facts", "unsourced_numbers", "{}"),
    ("Mid-clause breaks", "mid_clause_breaks", "{}"),
    ("Runtime error", "runtime_error", "{:+.0%}"),
    ("Problems", "problems", "{}"),
    ("Cost, USD", "usd", "{:.3f}"),
]


def per_topic_table(results: list[dict], variants: tuple[str, ...]) -> str:
    cols = [c for c in _TOPIC_COLUMNS
            if len(variants) > 1 or c[1] != "unsourced_numbers"]
    lines = ["| Topic | " + " | ".join(c[0] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for r in results:
        title = r["topic"]["title"]
        if "error" in r:
            lines.append(f"| {title} | {failure_note(r['error'])} |" + " |" * (len(cols) - 1))
            continue
        cells = [" / ".join(fmt.format(r[v]["metrics"][key]) for v in variants)
                 for _, key, fmt in cols]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(results: list[dict], settings: Settings, *, run_dir: str, date: str) -> str:
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    variants = settings.variants
    total = sum(r[v]["metrics"]["usd"] for r in ok for v in variants)
    channels = sorted({r["setup"]["channel"] if "setup" in r
                       else Path(r["topic"]["channel"]).stem for r in results})
    frames = ok[0]["setup"]["frames"] if ok else (settings.frames or DEFAULT_FRAMES)
    seconds = ok[0]["setup"]["target_seconds"] if ok else frames * 6
    m, s = divmod(seconds, 60)

    status = f"{len(ok)} finished"
    if failed:
        notes = sorted({failure_note(r["error"]) for r in failed})
        status += f", {len(failed)} {'; '.join(notes)}"
    merged = "".join(f", {len(x['topics'])} rerun {x['at'][:10]}" for x in settings.merges)
    if settings.mode == BASELINE_ONLY:
        what = (f"Baseline only: one request per topic for {frames} fixed slots "
                f"({m}:{s:02d}), model `{settings.model}`.")
        cta = "The call to action sits inside the last slots."
    else:
        what = (f"{m}:{s:02d} videos, {frames} frames. Writing model `{settings.model}`, "
                f"facts from `{settings.research_model}`.")
        cta = (f"Both variants are scored on their whole narration, call to action "
               f"included: the pipeline speaks it as an extra frame after its {frames} "
               f"cut frames, the baseline inside its {frames} slots. The pipeline's "
               f"tokens and cost include the facts query.")
    figures_note = ("Figures are numbers in digits or words other than 0-10, 100 and "
                    "1,000.")
    return "\n".join([
        f"Eval run {date}{merged}: {len(results)} topics ({', '.join(channels)}), "
        f"{status}. {what} Total cost ${total:.2f}.",
        "",
        results_table(results, variants, frames),
        "",
        f"{cta} {figures_note}",
        "",
        "<details><summary>Per topic" + (", baseline / pipeline" if len(variants) > 1
                                         else "") + "</summary>",
        "",
        per_topic_table(results, variants),
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
