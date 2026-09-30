"""Command line: python -m narration research|write|check|cut|eval."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .cards import ChannelCard, VideoCard, available
from .checks import check_narration, check_visuals, sentences_of
from .llm import DEFAULT_MODEL, RESEARCH_MODEL, LLMClient, LLMError, estimate_usd
from .pipeline import MAX_REPAIRS, Script, cut, script_problems, write_script
from .research import Research, collect
from .storyboard import assemble, write_all
from .timing import TimingError, segment_sections

log = logging.getLogger("narration")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "video"


def _client(args: argparse.Namespace) -> LLMClient:
    return LLMClient(model=args.model)


def _load_script(path: Path) -> Script:
    """A script JSON written by `write`, or plain text with one chapter per
    paragraph block."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        return Script.from_dict(data.get("script", data))
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    return Script(main_idea="", key_facts=[], final_cta="",
                  chapters=[{"title": f"Part {i}", "narration": " ".join(b.split())}
                            for i, b in enumerate(blocks, 1)])


def cmd_research(args: argparse.Namespace) -> int:
    card = ChannelCard.load(args.channel)
    res = collect(args.topic, card, _client(args), count=args.count,
                  model=args.research_model)
    for f in res.facts:
        print(f.as_prompt_line())
    path = res.save(Path(args.out) / f"{slugify(args.topic)}.facts.json")
    print(f"\n{len(res.facts)} facts, {len(res.sources)} sources, ${res.usd:.4f} -> {path}")
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    card = ChannelCard.load(args.channel)
    video = VideoCard(video_id=args.id or slugify(args.topic), title=args.topic,
                      channel=card, main_idea=args.idea or "")
    print(video.summary())
    calls = {"script": 1, "repair": args.repairs}
    if args.visuals:
        calls["visuals"] = 1
    queries = 0 if (args.facts or args.no_research) else 1
    usd = estimate_usd(args.model, calls, research_queries=queries)
    print(f"Estimated cost: up to ${usd:.2f}" if usd is not None
          else f"No price known for {args.model}; cost will be reported after the run")

    client = _client(args)
    out = Path(args.out) / video.video_id
    res: Research | None = None
    if args.facts:
        res = Research.load(Path(args.facts))
    elif not args.no_research:
        res = collect(args.topic, card, client, model=args.research_model)
        res.save(out / "facts.json")
    if res is not None:
        video.key_facts = res.claims
        print(f"Facts: {len(res.facts)} {res.by_status}")

    script = write_script(video, client, research=res, siblings=args.sibling,
                          max_repairs=args.repairs)
    # The script is the expensive part; it is saved before anything that can fail.
    out.mkdir(parents=True, exist_ok=True)
    (out / "script.json").write_text(json.dumps(script.to_dict(), ensure_ascii=False,
                                                indent=2), encoding="utf-8")
    print(f"Script: {len(script.chapters)} chapters, {len(script.text.split())} words "
          f"(budget {card.words}), repairs {script.repairs}, ${script.usd:.4f}")
    for p in script.problems:
        print(f"  ! {p}")

    seg, owner = cut(script, card)
    print(f"Cut: {len(seg.frames)} frames, {seg.total_seconds / 60:.2f} min, "
          f"broken sentences {seg.broken_sentences}")

    visuals, extra, usd_total = None, [], script.usd + (res.usd if res else 0.0)
    if args.visuals:
        from .visuals import design_visuals
        visuals, reply = design_visuals(video, seg, owner, script, client)
        extra = check_visuals([v.scene_concept for v in visuals],
                              [v.character_present for v in visuals], card)
        usd_total += reply.usd
        for p in extra:
            print(f"  ! {p}")
    board = assemble(video, script, seg, owner, visuals=visuals, research=res,
                     extra_problems=extra, usd=usd_total)
    for kind, path in write_all(board, out).items():
        print(f"{kind:<9}{path}")
    print(f"Total ${board.usd:.4f}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Free: run every check on an existing script."""
    card = ChannelCard.load(args.channel)
    script = _load_script(Path(args.file))
    res = Research.load(Path(args.facts)) if args.facts else None
    if args.frames:
        texts = [t for t in Path(args.file).read_text(encoding="utf-8").splitlines()
                 if t.strip()]
        problems = check_narration(texts, card)
    else:
        problems = script_problems(script.chapters, card, res,
                                   final_cta=script.final_cta or None)
        texts = sentences_of(script.chapters)
    print(f"{len(texts)} lines, {len(' '.join(texts).split())} words")
    for p in problems:
        print(f"  {p}")
    blocking = [p for p in problems if not p.advisory]
    print("clean" if not blocking else f"{len(blocking)} problems")
    return 1 if blocking else 0


def cmd_cut(args: argparse.Namespace) -> int:
    """Free: lay a script out on frames and print the timeline."""
    card = ChannelCard.load(args.channel)
    script = _load_script(Path(args.file))
    n = args.frames or card.frames
    try:
        seg, _ = segment_sections(script.sections, n,
                                      frame_seconds=card.target_seconds / n,
                                      rate=card.voice.syllables_per_second,
                                      pause=card.voice.sentence_pause_s,
                                      uniform=False)
    except TimingError as e:
        print(f"cannot cut: {e}")
        return 1
    for f in seg.frames:
        mark = "" if f.ends_sentence else "  [mid-sentence]"
        print(f"{f.number:03d} {f.start:6.1f}s {f.seconds:4.1f}s  {f.text}{mark}")
    print(f"\n{len(seg.frames)} frames, {seg.total_seconds:.0f}s spoken, "
          f"target {card.target_seconds}s, broken sentences {seg.broken_sentences}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from . import evaluate as ev

    topics = ev.load_topics(Path(args.topics))
    if args.only:
        wanted = set(args.only.split(","))
        topics = [t for t in topics if t.id in wanted]
    if args.limit:
        topics = topics[:args.limit]
    if not topics:
        print("no topics selected")
        return 1
    expected, upper = ev.estimate(len(topics), args.model, args.repairs)
    print(f"{len(topics)} topics, writing model {args.model}, facts from "
          f"{args.research_model}, up to {args.repairs} repairs per script")
    if expected is None:
        print(f"No price known for {args.model}; the real cost is reported after the run.")
    else:
        print(f"Estimated cost: about ${expected:.2f}, at most ${upper:.2f}")
    if not args.yes:
        print("Nothing was spent. Re-run with --yes to call the models.")
        return 1

    out = Path(args.out)
    results, run_dir = ev.run_eval(topics, _client(args), out, max_repairs=args.repairs,
                                   research_model=args.research_model)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = ev.report(results, model=args.model, research_model=args.research_model,
                     run_dir=run_dir.as_posix(), date=date)
    (out / "results.md").write_text(text, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(ev.summarise(results), indent=2),
                                          encoding="utf-8")
    print(text)
    if args.readme:
        ev.update_readme(Path(args.readme), text)
        print(f"Inserted into {args.readme}")
    return 0 if all("error" not in r for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m narration", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    channels = f"bundled: {', '.join(available())}, or a path to a card .yaml"

    def paid(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--research-model", default=RESEARCH_MODEL)

    p = sub.add_parser("research", help="collect facts with statuses (~$0.006)")
    p.add_argument("topic")
    p.add_argument("--channel", required=True, help=channels)
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--out", default="out")
    paid(p)
    p.set_defaults(fn=cmd_research)

    p = sub.add_parser("write", help="facts, script, checks, repairs, cut")
    p.add_argument("topic")
    p.add_argument("--channel", required=True, help=channels)
    p.add_argument("--id", help="output name; defaults to a slug of the topic")
    p.add_argument("--idea", help="main idea, if you have one")
    p.add_argument("--facts", help="facts JSON from `research` instead of a new search")
    p.add_argument("--no-research", action="store_true",
                   help="no facts: the script will carry no numbers")
    p.add_argument("--sibling", action="append", default=[],
                   help="title of another video of the channel; repeatable")
    p.add_argument("--repairs", type=int, default=MAX_REPAIRS)
    p.add_argument("--visuals", action="store_true",
                   help="also run stage 2 and build image and animation prompts")
    p.add_argument("--out", default="out")
    paid(p)
    p.set_defaults(fn=cmd_write)

    p = sub.add_parser("check", help="run the checks on a script (free)")
    p.add_argument("file", help="script.json from `write`, or a text file")
    p.add_argument("--channel", required=True, help=channels)
    p.add_argument("--facts", help="facts JSON to verify numbers and statuses")
    p.add_argument("--frames", action="store_true",
                   help="treat each non-empty line of a text file as one frame")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("cut", help="cut a script into frames (free)")
    p.add_argument("file", help="script.json from `write`, or a text file")
    p.add_argument("--channel", required=True, help=channels)
    p.add_argument("--frames", type=int, help="override the channel's frame count")
    p.set_defaults(fn=cmd_cut)

    p = sub.add_parser("eval", help="baseline vs pipeline on eval/topics.yaml")
    p.add_argument("--topics", default="eval/topics.yaml")
    p.add_argument("--out", default="eval")
    p.add_argument("--only", help="comma-separated topic ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--repairs", type=int, default=MAX_REPAIRS)
    p.add_argument("--readme", help="insert the results table into this README")
    p.add_argument("--yes", action="store_true", help="actually spend money")
    paid(p)
    p.set_defaults(fn=cmd_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
