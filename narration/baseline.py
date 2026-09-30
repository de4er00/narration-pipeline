"""The approach this pipeline replaced: one request, N fixed frame slots.

It gets the same channel card, word budget and originality rules in words,
the way the production templates stated them. It gets no facts, no checks and
no repairs, and the model decides the frame boundaries by filling the slots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cards import VideoCard
from .checks import BANNED_SCAFFOLDS
from .llm import ChatClient, LLMError

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frames"],
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["number", "narration", "visual"],
                "properties": {
                    "number": {"type": "integer"},
                    "narration": {"type": "string"},
                    "visual": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM = (
    "You are a senior script writer for an explainer video channel. You write "
    "spoken US English for a single narrator. Respond with JSON only."
)


@dataclass
class BaselineScript:
    frames: list[str]
    visuals: list[str] = field(default_factory=list)
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(self.frames)


def baseline_prompt(video: VideoCard) -> str:
    ch = video.channel
    n = ch.frames
    m, s = divmod(ch.target_seconds, 60)
    per_frame = round(ch.words / n)
    scaffolds = "\n".join(f'  - "{x}"' for x in BANNED_SCAFFOLDS)
    return f"""ROLE
{ch.role}

THE VIDEO
Title: {video.title}
Runtime: {m}:{s:02d}, narrated by one voice.

FRAME CONTRACT
Write the video as exactly {n} frames, numbered 1 to {n}. Each frame is a
{ch.frame_seconds:.0f}-second slot on screen: one narration line of about
{per_frame} words and one short description of the picture. The whole script
is about {ch.words} words.

NARRATIVE DIRECTION
{ch.narrative_direction}

HOOK
Open inside a situation, in the second person and the present tense. No
caveats, no sources and no organisation names in the first thirty seconds.

NARRATION ORIGINALITY LOCK
Every frame says something new. Never repeat a line, never restate the
previous frame, and use no opening phrase of four or more words more than
twice. These phrases are forbidden:
{scaffolds}

VOICE
Spoken register: contract (you're, don't, it's), no semicolons, no lists of
four or more. Address the viewer as "you" and ask at least three questions
spread across the video. Invent no quotations, studies or exact figures.

ENDING
Turn the subject onto the viewer's own life today, then close with one
question to the viewer that asks exactly this: {ch.cta}

OUTPUT
JSON with `frames`: {n} entries of number, narration and visual, in order."""


def write_baseline(video: VideoCard, client: ChatClient) -> BaselineScript:
    reply = client.ask_json(SYSTEM, baseline_prompt(video), SCHEMA, name="baseline")
    frames = reply.data.get("frames") or []
    if not frames:
        raise LLMError("the baseline returned no frames")
    frames = sorted(frames, key=lambda f: int(f.get("number", 0)))
    return BaselineScript(frames=[str(f.get("narration", "")).strip() for f in frames],
                          visuals=[str(f.get("visual", "")).strip() for f in frames],
                          usd=reply.usd, tokens_in=reply.tokens_in,
                          tokens_out=reply.tokens_out, seconds=reply.seconds)
