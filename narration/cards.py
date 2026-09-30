"""Channel and video cards: everything the generator gets as input."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .timing import words_budget

CHANNELS_DIR = Path(__file__).resolve().parent / "channels"

DEFAULT_TARGET_SECONDS = 360
DEFAULT_FRAMES = 60

# Average sentence length in production scripts. The prompt asks for a
# sentence count as well as a word count: sixty frames cannot be filled with
# forty long sentences.
WORDS_PER_SENTENCE = 14.4


class CardError(RuntimeError):
    """A card cannot be read or misses a required field."""


@dataclass
class VoiceProfile:
    """The narrator voice and its measured pace (see `timing`)."""
    label: str
    syllables_per_second: float
    sentence_pause_s: float

    @classmethod
    def from_dict(cls, raw: dict, where: str) -> "VoiceProfile":
        v = raw or {}
        for key in ("syllables_per_second", "sentence_pause_s"):
            if v.get(key) is None:
                raise CardError(
                    f"{where}: voice.{key} is missing. Pace is measured from a "
                    f"recording, never guessed: voices differ almost twofold.")
        return cls(label=str(v.get("label") or "unnamed voice").strip(),
                   syllables_per_second=float(v["syllables_per_second"]),
                   sentence_pause_s=float(v["sentence_pause_s"]))


@dataclass
class ChannelCard:
    """What stays the same across a channel's videos."""
    id: str
    character_name: str
    character_share: tuple[float, float]
    character_bible: str
    role: str
    narrative_direction: str
    cta: str
    voice: VoiceProfile
    character_sheet: str = ""
    style_sheet: str = ""
    target_seconds: int = DEFAULT_TARGET_SECONDS
    frames: int = DEFAULT_FRAMES

    @property
    def frame_seconds(self) -> float:
        return self.target_seconds / self.frames

    @property
    def words(self) -> int:
        """Word budget for this runtime and this voice.

        A fixed 800-900 words for every channel was right for only two of five
        voices: 850 words ran 4:47 with the fastest and 7:32 with the slowest
        instead of 6:00.
        """
        return words_budget(self.frames, self.frame_seconds,
                            self.voice.syllables_per_second,
                            self.voice.sentence_pause_s)

    @property
    def sentences(self) -> int:
        """Target sentence count, with 10% headroom over the frame count.

        With exactly one sentence per frame the cutter has no choice and must
        break sentences at clauses; a test run at parity produced three broken
        frames out of sixty.
        """
        return max(int(round(self.frames * 1.1)),
                   int(round(self.words / WORDS_PER_SENTENCE)))

    @property
    def words_per_sentence(self) -> float:
        return round(self.words / max(1, self.sentences), 1)

    @property
    def character_frames(self) -> tuple[int, int]:
        lo, hi = self.character_share
        return math.floor(self.frames * lo), math.ceil(self.frames * hi)

    @classmethod
    def from_file(cls, path: Path) -> "ChannelCard":
        if not path.is_file():
            raise CardError(f"no channel card at {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        share = raw.get("character_share") or [0.25, 0.40]
        return cls(
            id=str(raw.get("id") or path.stem),
            character_name=_text(raw, "character_name", path),
            character_share=(float(share[0]), float(share[1])),
            character_bible=_text(raw, "character_bible", path),
            role=_text(raw, "role", path),
            narrative_direction=_text(raw, "narrative_direction", path),
            cta=_text(raw, "cta", path),
            voice=VoiceProfile.from_dict(raw.get("voice"), path.name),
            character_sheet=str(raw.get("character_sheet") or "").strip(),
            style_sheet=str(raw.get("style_sheet") or "").strip(),
            target_seconds=int(raw.get("target_seconds") or DEFAULT_TARGET_SECONDS),
            frames=int(raw.get("frames") or DEFAULT_FRAMES),
        )

    @classmethod
    def load(cls, channel: str) -> "ChannelCard":
        """A bundled channel by id, or a path to any card file."""
        path = Path(channel)
        if path.suffix in (".yaml", ".yml") and path.is_file():
            return cls.from_file(path)
        bundled = CHANNELS_DIR / f"{channel}.yaml"
        if not bundled.is_file():
            raise CardError(f"unknown channel {channel!r}; bundled channels: "
                            f"{', '.join(available())}")
        return cls.from_file(bundled)


def _text(raw: dict, key: str, path: Path) -> str:
    val = " ".join(str(raw.get(key) or "").split())
    if not val:
        raise CardError(f"{path.name}: field {key!r} is empty")
    return val


def available() -> list[str]:
    return sorted(p.stem for p in CHANNELS_DIR.glob("*.yaml"))


@dataclass
class VideoCard:
    """One video: the topic plus anything the operator adds by hand."""
    video_id: str
    title: str
    channel: ChannelCard
    main_idea: str = ""
    key_facts: list[str] = field(default_factory=list)
    extra_restrictions: str = ""

    def summary(self) -> str:
        ch = self.channel
        m, s = divmod(ch.target_seconds, 60)
        lo, hi = ch.character_frames
        return "\n".join([
            f"{self.video_id}: {self.title}",
            f"Channel: {ch.id} ({ch.character_name})",
            f"Frames: {ch.frames} x {ch.frame_seconds:.1f}s, runtime {m}:{s:02d}",
            f"Voice: {ch.voice.label}, {ch.voice.syllables_per_second} syll/s, "
            f"pause {ch.voice.sentence_pause_s}s",
            f"Budget: {ch.words} words, about {ch.sentences} sentences",
            f"Character in {lo}-{hi} frames",
        ])
