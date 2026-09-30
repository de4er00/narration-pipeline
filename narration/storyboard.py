"""Final assembly: frames, the call-to-action frame, JSON and Markdown output."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cards import VideoCard
from .checks import Problem
from .frame_prompts import FrameVisual, build_animation_prompt, build_image_prompt, reference_inputs
from .pipeline import Script
from .research import Research
from .timing import Segmentation, estimate_seconds

_CLOSING_QUOTES = "\"'”’»"


@dataclass
class StoryFrame:
    number: int
    start: float
    end: float
    narration: str
    chapter: str
    ends_sentence: bool = True
    visual: dict = field(default_factory=dict)
    image_prompt: str = ""
    animation_prompt: str = ""
    references: list[str] = field(default_factory=list)


@dataclass
class Storyboard:
    video_id: str
    title: str
    channel: str
    main_idea: str
    key_facts: list[str]
    final_cta: str
    next_video: str
    chapters: list[str]
    frames: list[StoryFrame]
    facts: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    usd: float = 0.0

    @property
    def narration(self) -> str:
        return " ".join(f.narration for f in self.frames)

    @property
    def runtime(self) -> float:
        return self.frames[-1].end if self.frames else 0.0


def normalise_cta(text: str) -> str:
    """The call to action is a spoken line and must end with a mark.

    A closing quote is its own case: a call ending in a quoted title with a
    full stop inside got a second one ('Profitable.".'). The stop moves
    outside the quote; a question or exclamation mark stays inside, since a
    title can be a question, and a stop follows it.
    """
    s = (text or "").strip()
    if not s:
        return s
    if s.endswith("."):
        inner = s[:-1].rstrip(_CLOSING_QUOTES)
        if s[len(inner):-1] and inner.endswith("."):
            return inner[:-1] + s[len(inner):-1] + "."
    body = s.rstrip(_CLOSING_QUOTES)
    quotes = s[len(body):]
    if quotes and body:
        return (body[:-1] + quotes + ".") if body[-1] == "." else s + "."
    return s if s[-1] in ".!?…" else s + "."


def cta_already_spoken(cta: str, frames: list[StoryFrame], tail: int = 3) -> bool:
    """Did the model already put the call to action at the end of a chapter?

    The model often writes the call into the last chapter as well as into
    `final_cta`, and the viewer then hears it twice. Compared by significant
    words, because the two versions usually differ by a word or two.
    """
    def words(t: str) -> set[str]:
        return {w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3}

    want = words(cta)
    if len(want) < 4:
        return False
    spoken = words(" ".join(f.narration for f in frames[-tail:]))
    return len(want & spoken) / len(want) >= 0.6


def cta_visual(video: VideoCard, number: int) -> FrameVisual:
    """The closing frame is built by code: the same sign-off every video.

    No hands or eyes are assumed, because one character may have pincers and
    another a camera lens for a face.
    """
    name = video.channel.character_name
    return FrameVisual(
        number=number,
        scene_concept=(f"{name} turns to face the viewer directly at the very end "
                       f"of the video and invites an answer"),
        character_present=True,
        shot="close",
        character_action=(f"{name} faces the viewer straight on in an open, "
                          f"inviting pose, turned toward the camera"),
        object_placement=("the character is centred with generous empty space "
                          "around; nothing competes for attention"),
        main_motion="the character settles into a still, open posture",
    )


def _with_prompts(frame: StoryFrame, video: VideoCard, vis: FrameVisual) -> StoryFrame:
    card = video.channel
    frame.visual = asdict(vis)
    frame.image_prompt = build_image_prompt(card, vis)
    frame.animation_prompt = build_animation_prompt(card, vis)
    frame.references = reference_inputs(card, vis)
    return frame


def assemble(video: VideoCard, script: Script, seg: Segmentation, owner: list[int],
             *, visuals: list[FrameVisual] | None = None,
             research: Research | None = None,
             extra_problems: list[Problem] | None = None,
             usd: float = 0.0) -> Storyboard:
    """Frames plus the call to action as the last frame and the last words."""
    titles = [t for t, _ in script.sections]
    frames = [StoryFrame(number=f.number, start=round(f.start, 2),
                         end=round(f.start + f.seconds, 2), narration=f.text,
                         chapter=titles[sec], ends_sentence=f.ends_sentence)
              for f, sec in zip(seg.frames, owner)]
    if visuals is not None:
        frames = [_with_prompts(fr, video, v) for fr, v in zip(frames, visuals)]

    cta = normalise_cta(script.final_cta)
    if cta and frames and not cta_already_spoken(cta, frames):
        card = video.channel
        start = frames[-1].end
        secs = estimate_seconds(cta, rate=card.voice.syllables_per_second,
                                pause=card.voice.sentence_pause_s)
        # The call has no chapter of its own: it is recognised as the last
        # frame, and an extra chapter marker would be a lie on the video page.
        last = StoryFrame(number=len(frames) + 1, start=start,
                          end=round(start + secs, 2), narration=cta,
                          chapter=frames[-1].chapter)
        if visuals is not None:
            last = _with_prompts(last, video, cta_visual(video, last.number))
        frames.append(last)

    problems = list(script.problems) + list(extra_problems or [])
    return Storyboard(
        video_id=video.video_id, title=video.title, channel=video.channel.id,
        main_idea=script.main_idea, key_facts=script.key_facts, final_cta=cta,
        next_video=script.next_video, chapters=titles, frames=frames,
        facts=[asdict(f) for f in research.facts] if research else [],
        problems=[str(p) for p in problems],
        usd=round(usd or script.usd, 5))


def write_json(board: Storyboard, path: Path) -> Path:
    payload = {**asdict(board), "narration": board.narration,
               "runtime_seconds": round(board.runtime, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def to_markdown(board: Storyboard) -> str:
    lines = [f"# {board.title}", "",
             f"Channel: {board.channel} · {len(board.frames)} frames · "
             f"runtime {_clock(board.runtime)} · cost ${board.usd:.3f}", ""]
    if board.main_idea:
        lines += [f"**Main idea.** {board.main_idea}", ""]
    if board.facts:
        lines += ["## Facts", ""]
        lines += [f"- [{f['status']}] {f['claim']} ({f['source']})" for f in board.facts]
        lines.append("")
    if board.problems:
        lines += ["## Open problems", ""] + [f"- {p}" for p in board.problems] + [""]

    lines += ["## Narration", ""]
    current = None
    for f in board.frames:
        if f.chapter != current:
            current = f.chapter
            lines += [f"### {current}", ""]
        mark = "" if f.ends_sentence else " (continues)"
        lines.append(f"**{f.number:03d}** `{_clock(f.start)}-{_clock(f.end)}` "
                     f"{f.narration}{mark}")
        lines.append("")
    if board.next_video:
        lines += [f"Next video: {board.next_video}", ""]

    if any(f.image_prompt for f in board.frames):
        lines += ["## Frame prompts", ""]
        for f in board.frames:
            refs = ", ".join(f.references)
            lines += [f"### Frame {f.number:03d}", "", f"Attach: {refs}", "",
                      "```text", f.image_prompt, "```", "",
                      "```text", f.animation_prompt, "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_all(board: Storyboard, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{board.video_id}.md"
    md.write_text(to_markdown(board), encoding="utf-8")
    return {"json": write_json(board, out_dir / f"{board.video_id}.json"), "markdown": md}
