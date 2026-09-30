"""Deterministic checks for a finished script.

Each check targets a defect that shipped at least once: a rule in the prompt
is a request, a check in code is a guarantee. Every function takes lines
(frames or sentences) and returns problems; empty means clean. Thresholds were
set on production scripts and on transcripts of channels that hold viewers.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .cards import ChannelCard
from .numbers import figures
from .timing import _WORD_RE, estimate_seconds, split_sentences

BANNED_SCAFFOLDS = (
    "the hidden consequence is",
    "the evidence becomes clearer",
    "the practical conclusion is",
    "this matters because",
    "the key point is",
    "it is important to note",
    "looks harmless in isolation",
    # "Stay with me" appeared in 14% of scripts, right at 0:18-0:21, where
    # retention curves lost half the audience. A plea to stay tells the viewer
    # the next part is boring.
    "stay with me",
    "the surprising part is not",
)

# None of the nine competitor channels studied opens on a source; they open
# on a scene. An instrument reads as a citation too: an opening built on
# "For six years, Hubble watched..." kept 39% of viewers at 0:21.
CITATION_MARKERS = (
    "according to", "reports that", "researchers", "a study",
    "nasa", "noaa", "bureau of", "university of",
    "institute", "the journal", "survey found", "data from",
    "hubble", "james webb", "voyager", "kepler telescope",
)
# Whole words only: a short acronym matched inside an ordinary word once sent
# a clean script into two paid repairs.
_CITATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in CITATION_MARKERS) + r")\b")

HEDGES = (
    "thought experiment", "it is important to note", "we should separate",
    "this is hypothetical", "cannot actually", "of course this",
    "before we begin", "in this video we will", "let us first define",
)

PRODUCTION_WORDS = ("the script", "this script", "the narration",
                    "the storyboard", "this frame", "the voiceover",
                    "the material here", "this material",
                    "the material we have", "the material above")

MAX_OPENING_REPEATS = 2
NGRAM = 4
MAX_NGRAM_REPEATS = 2

# Contractions per 1,000 words: 3.5 in the old scripts, 13-27 in competitors.
# First sentence: 16 words in ours, 5 in the most-watched competitor.
MIN_CONTRACTIONS_PER_1000 = 10.0
MAX_FIRST_SENTENCE_WORDS = 12

# Share of sentences of six words or fewer: 0-3% in ours, 6-39% in channels
# that hold viewers. The generator reliably reaches 6% when asked by count,
# so the floor is 5%: it catches the flat band without failing good scripts.
SHORT_SENTENCE_WORDS = 6
MIN_SHORT_SENTENCE_SHARE = 0.05

# Abstract nouns per 100 words: 4.5 in the old scripts, 1.1-2.4 in
# competitors. The new prompt gives 3.2-3.3, so the ceiling catches a return
# to report language rather than arguing over tenths.
MAX_ABSTRACT_PER_100 = 3.5

_ABSTRACT_RE = re.compile(
    r"[A-Za-z]+(?:tion|sion|ment|ness|ity|ance|ence|ism)\b", re.I)
_CONTRACTION_RE = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b")
_LONG_LIST_RE = re.compile(r"(?:\b[\w-]+\b,\s+){3,}(?:and|or)\s+\b[\w-]+\b")
# An introductory word before a comma is not a list item: "Without it,
# warmth, light, and cooked food" is a list of three.
_LIST_INTRO = frozenset(
    "it this that them him her us you me there then now so yes no well still "
    "instead however today first second finally".split())
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Problem:
    """One finding; `at` holds 1-based positions in the checked list.

    Advisory problems are shown but never buy a paid repair: they depend on
    missing material, and a rewrite can only fix them by inventing.
    """
    kind: str
    message: str
    at: list[int] = field(default_factory=list)
    advisory: bool = False

    def __str__(self) -> str:
        tail = f" (at {', '.join(map(str, self.at[:12]))})" if self.at else ""
        mark = "~" if self.advisory else ""
        return f"{mark}[{self.kind}] {self.message}{tail}"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(texts: list[str]) -> list[str]:
    return [x.strip() for t in texts for x in _SENTENCE_END.split(t) if x.strip()]


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / min(len(a), len(b))


# --- degeneration ------------------------------------------------------------

def duplicate_frames(texts: list[str]) -> list[Problem]:
    """Lines repeated verbatim. Slot-filled scripts had 15-19 in 80."""
    seen: dict[str, int] = {}
    dupes: dict[str, list[int]] = {}
    for i, t in enumerate(texts, start=1):
        key = _norm(t)
        if key in seen:
            dupes.setdefault(key, [seen[key]]).append(i)
        else:
            seen[key] = i
    return [Problem("duplicate", f'line repeated verbatim: "{key[:70]}"', at=nums)
            for key, nums in dupes.items()]


def banned_scaffolds(texts: list[str]) -> list[Problem]:
    out = []
    for phrase in BANNED_SCAFFOLDS:
        hits = [i for i, t in enumerate(texts, start=1) if phrase in _norm(t)]
        if hits:
            out.append(Problem("scaffold", f'banned filler phrase "{phrase}"',
                               at=hits))
    return out


def repeated_openings(texts: list[str],
                      limit: int = MAX_OPENING_REPEATS) -> list[Problem]:
    heads: dict[str, list[int]] = {}
    for i, t in enumerate(texts, start=1):
        toks = _tokens(t)[:NGRAM]
        if len(toks) == NGRAM:
            heads.setdefault(" ".join(toks), []).append(i)
    return [Problem("opening", f'"{h}" opens {len(n)} lines', at=n)
            for h, n in heads.items() if len(n) > limit]


def ngram_repeats(texts: list[str], n: int = NGRAM,
                  limit: int = MAX_NGRAM_REPEATS) -> list[tuple[str, int]]:
    """Every n-word phrase used more than `limit` times, most frequent first.

    Phrases with a proper noun are exempt: names repeat legitimately. This
    measure alone separated the early batches (0 hits) from slot-filled ones.
    """
    joined = " ".join(texts)
    toks = _tokens(joined)
    counts = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    proper = {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}", joined)}
    bad = [(" ".join(g), c) for g, c in counts.items()
           if c > limit and not (set(g) & proper)]
    return sorted(bad, key=lambda x: -x[1])


def repeated_ngrams(texts: list[str], n: int = NGRAM,
                    limit: int = MAX_NGRAM_REPEATS) -> list[Problem]:
    return [Problem("ngram", f'phrase "{g}" repeats {c} times')
            for g, c in ngram_repeats(texts, n, limit)[:10]]


def paraphrased_neighbours(texts: list[str], threshold: float = 0.6,
                           min_shared: int = 4) -> list[Problem]:
    """Adjacent lines that retell each other.

    Honest development of a thought shares 0.2-0.45 of its words. The four
    shared words rule spares a deliberate short echo after a long sentence;
    on 72 production scripts every hit without it was a false one.
    """
    out = []
    for i in range(len(texts) - 1):
        a, b = set(_tokens(texts[i])), set(_tokens(texts[i + 1]))
        if len(a) < 5 or len(b) < 5 or len(a & b) < min_shared:
            continue
        overlap = _overlap(a, b)
        if overlap >= threshold:
            out.append(Problem("paraphrase",
                               f"neighbouring lines overlap by {overlap:.0%}",
                               at=[i + 1, i + 2]))
    return out


# --- content -----------------------------------------------------------------

def _window(texts: list[str], seconds: list[float], window: float,
            from_end: bool = False) -> list[str]:
    pairs = list(zip(texts, seconds))
    if from_end:
        pairs.reverse()
    out, clock = [], 0.0
    for t, s in pairs:
        if clock >= window:
            break
        out.append(t)
        clock += s
    return out[::-1] if from_end else out


def hook_quality(texts: list[str], seconds: list[float],
                 window: float = 30.0) -> list[Problem]:
    """The first thirty seconds: a situation, not a caveat or a citation.

    When a number counted as a hook, 61% of scripts cited a figure there and
    retention at 0:21 fell to 39-44%, so the check asks for the viewer instead.
    """
    head = _window(texts, seconds, window)
    joined = _norm(" ".join(head))
    out = [Problem("hook", f'caveat "{h}" in the first {window:.0f} s defuses '
                           f"the premise before it is stated")
           for h in HEDGES if h in joined]
    if not head:
        return out
    if "you" not in joined:
        out.append(Problem("hook", f"the first {window:.0f} s never address "
                                   f"the viewer"))
    hit = _CITATION_RE.search(joined)
    if hit:
        out.append(Problem("hook", f'source "{hit.group()}" in the first '
                                   f"{window:.0f} s: viewers leave before they "
                                   f"learn why it matters"))
    first = _SENTENCE_END.split(" ".join(head).strip())[0]
    if len(first.split()) > MAX_FIRST_SENTENCE_WORDS:
        out.append(Problem("hook", f"the first sentence has {len(first.split())} "
                                   f"words, ceiling {MAX_FIRST_SENTENCE_WORDS}"))
    return out


def spoken_register(texts: list[str],
                    per_1000: float = MIN_CONTRACTIONS_PER_1000) -> list[Problem]:
    """A narrator reading "you are" instead of "you're" sounds like a report."""
    joined = " ".join(texts)
    words = len(_WORD_RE.findall(joined)) or 1
    got = len(_CONTRACTION_RE.findall(joined)) / words * 1000
    if got >= per_1000:
        return []
    return [Problem("register", f"{got:.1f} contractions per 1,000 words, "
                                f"minimum {per_1000:.0f}: written, not spoken")]


def sentence_rhythm(texts: list[str],
                    floor: float = MIN_SHORT_SENTENCE_SHARE) -> list[Problem]:
    """Short sentences are the beats; without them the voice runs flat."""
    sents = _sentences(texts)
    if len(sents) < 20:
        return []
    short = sum(1 for x in sents if len(_WORD_RE.findall(x)) <= SHORT_SENTENCE_WORDS)
    share = short / len(sents)
    if share >= floor:
        return []
    return [Problem("rhythm", f"only {share:.0%} of sentences have "
                              f"{SHORT_SENTENCE_WORDS} words or fewer, minimum "
                              f"{floor:.0%}: no beats, one flat band")]


def abstract_density(texts: list[str],
                     ceiling: float = MAX_ABSTRACT_PER_100) -> list[Problem]:
    joined = " ".join(texts)
    words = len(_WORD_RE.findall(joined)) or 1
    found = _ABSTRACT_RE.findall(joined)
    got = len(found) / words * 100
    if got <= ceiling:
        return []
    sample = sorted({m.lower() for m in found})[:6]
    return [Problem("abstract", f"{got:.1f} abstract nouns per 100 words, "
                                f"ceiling {ceiling:.1f}: {', '.join(sample)}")]


def ending_repeats_itself(texts: list[str], last: int = 4,
                          threshold: float = 0.5) -> list[Problem]:
    """The closing sentences saying the same thing twice, checked by sentence
    because a repair tends to append a second call to action in the same frame."""
    tail = _sentences(texts)[-last:]
    for i in range(len(tail) - 1):
        for j in range(i + 1, len(tail)):
            a, b = set(_tokens(tail[i])), set(_tokens(tail[j]))
            if len(a) >= 4 and len(b) >= 4 and _overlap(a, b) >= threshold:
                return [Problem("ending", f'the last sentences repeat each other: '
                                          f'"{tail[i][:48]}" and "{tail[j][:48]}"')]
    return []


def unspeakable_syntax(texts: list[str]) -> list[Problem]:
    """Semicolons and lists of four or more: on the ear that is noise."""
    out: list[Problem] = []
    for i, t in enumerate(texts, 1):
        if ";" in t:
            out.append(Problem("syntax", "semicolon in a spoken line", at=[i]))
        if _has_long_list(t):
            out.append(Problem("syntax", "a list of four or more items", at=[i]))
    return out


def _has_long_list(text: str) -> bool:
    for m in _LONG_LIST_RE.finditer(text):
        items = [w.lower() for w in re.findall(r"\b([\w-]+)\b,\s+", m.group())]
        while items and items[0] in _LIST_INTRO:
            items.pop(0)
        if len(items) >= 3:
            return True
    return False


def finale_returns_to_viewer(texts: list[str], seconds: list[float],
                             window: float = 30.0) -> list[Problem]:
    """The last thirty seconds must turn to the viewer; a correct summary of
    the past is a dead ending."""
    joined = _norm(" ".join(_window(texts, seconds, window, from_end=True)))
    if not joined:
        return []
    if re.search(r"\b(you|your|yours|we|us|our)\b", joined):
        return []
    return [Problem("finale", f"the last {window:.0f} s never turn to the viewer: "
                              f"the video ends on the past, not on their life")]


def cta_follows_card(final_cta: str, card: ChannelCard) -> list[Problem]:
    """The call to action is the channel's question, not advice for the day.

    23 of 58 scripts ended on advice or a subscribe request instead. If the
    card names the character, so must the call (advice never does); otherwise
    the call has to ask something.
    """
    cta = " ".join((final_cta or "").split())
    if not cta:
        return [Problem("cta", "there is no call to action")]
    name = (card.character_name or "").lower()

    def named(text: str) -> bool:
        return bool(name) and bool(re.search(rf"\b{re.escape(name)}\b", text.lower()))

    if named(card.cta):
        if named(cta):
            return []
        return [Problem("cta", f"the call to action drifted from the channel's "
                               f"question: {card.cta}")]
    asks = "?" in cta or re.search(
        r"\b(tell|comment|let us know)\b.*\b(which|what|how|whether)\b", cta, re.I)
    if not asks:
        return [Problem("cta", f"the call to action asks nothing; the channel "
                               f"asks: {card.cta}")]
    return []


def questions_on_the_seams(texts: list[str], minimum: int = 3) -> list[Problem]:
    """Questions belong on the seams between chapters, not bunched at the end."""
    marks = [i for i, x in enumerate(texts, start=1) if "?" in x]
    if len(marks) < minimum:
        return [Problem("seams", f"{len(marks)} questions, minimum {minimum}: a "
                                 f"question bridges chapters, not only the end")]
    quarter = max(1, len(texts) * 3 // 4)
    if all(i > quarter for i in marks):
        return [Problem("seams", "all questions are bunched in the last quarter")]
    return []


def viewer_address(texts: list[str], per_100: float = 0.25) -> list[Problem]:
    """Slot-filled scripts addressed the viewer exactly zero times."""
    words = _tokens(" ".join(texts))
    if not words:
        return []
    density = sum(1 for w in words if w in ("you", "your", "yours")) / len(words) * 100
    if density >= per_100:
        return []
    return [Problem("address", f"{density:.2f} addresses to the viewer per 100 "
                               f"words, minimum {per_100}")]


def questions(texts: list[str], minimum: int = 1) -> list[Problem]:
    n = sum(t.count("?") for t in texts)
    if n >= minimum:
        return []
    return [Problem("questions", f"{n} question marks, minimum {minimum}: one of "
                                 f"the few ways to break synthetic monotony")]


def concreteness(texts: list[str], ceiling_per_100: float = 1.60) -> list[Problem]:
    """A ceiling on figures and no floor: 32 numbers in 823 words once made a
    script sound like a report read aloud. Spelled-out figures count too."""
    words = _tokens(" ".join(texts))
    if not words:
        return []
    density = len(figures(" ".join(texts))) / len(words) * 100
    if density <= ceiling_per_100:
        return []
    return [Problem("overload", f"{density:.2f} numbers per 100 words, ceiling "
                                f"{ceiling_per_100}: keep the most vivid, turn the "
                                f"rest into mechanisms or comparisons")]


# Editorial rules spoken aloud. A repair for "disputed fact stated as fact"
# once wrote "Neither side can be presented as settled fact here" into the
# narration. Those sentences contain none of PRODUCTION_WORDS.
_EDITORIAL_RE = re.compile(
    r"\b(?:presented|framed|treated|stated)\s+as\s+(?:a\s+)?"
    r"(?:settled|established|confirmed|proven)\s+fact"
    r"|\b(?:must|should|has to|needs to|have to)\s+(?:remain|stay|be kept)\s+visible\b"
    r"|\b(?:can|could|should|must|cannot|can't|shouldn't)\s+(?:not\s+)?be\s+"
    r"(?:presented|framed)\s+as\b"
    r"|\bseparate\s+(?:the\s+)?(?:confirmed\s+)?facts?\s+from\s+(?:the\s+)?"
    r"(?:possibilit|speculation|hypothes|guess|opinion)",
    re.IGNORECASE)


def production_leak(texts: list[str]) -> list[Problem]:
    """The viewer does not know a script, frames or "material" exist."""
    out = []
    for w in PRODUCTION_WORDS:
        hits = [i for i, t in enumerate(texts, start=1) if w in _norm(t)]
        if hits:
            out.append(Problem("production", f'production word "{w}"', at=hits))
    hits = [i for i, t in enumerate(texts, start=1) if _EDITORIAL_RE.search(t)]
    if hits:
        out.append(Problem("production", "an editorial rule about presenting facts "
                                         "is spoken aloud; tell the disagreement "
                                         "through the story instead", at=hits))
    return out


# --- length ------------------------------------------------------------------

def budget(texts: list[str], card: ChannelCard,
           tolerance: float = 0.08) -> list[Problem]:
    joined = " ".join(texts)
    secs = estimate_seconds(joined, rate=card.voice.syllables_per_second,
                            pause=card.voice.sentence_pause_s)
    off = secs / card.target_seconds - 1
    if abs(off) <= tolerance:
        return []
    m, s = divmod(int(secs), 60)
    tm, ts = divmod(card.target_seconds, 60)
    words = len(_WORD_RE.findall(joined))
    return [Problem("budget", f"{words} words run {m}:{s:02d} against a target "
                              f"of {tm}:{ts:02d} ({off:+.0%}); the budget for this "
                              f"voice is {card.words} words")]


# --- visuals -----------------------------------------------------------------

def visual_variety(concepts: list[str], min_unique_ratio: float = 0.8,
                   ) -> list[Problem]:
    """A hand-made storyboard used 20 objects for 80 frames, each four times."""
    out: list[Problem] = []
    uniq = len({_norm(c) for c in concepts})
    if concepts and uniq / len(concepts) < min_unique_ratio:
        out.append(Problem("visual", f"{uniq} unique scenes out of {len(concepts)}, "
                                     f"below {min_unique_ratio:.0%}"))
    for i in range(len(concepts) - 1):
        a, b = set(_tokens(concepts[i])), set(_tokens(concepts[i + 1]))
        if len(a) >= 4 and len(b) >= 4 and _overlap(a, b) >= 0.8:
            out.append(Problem("visual", "neighbouring frames describe one scene",
                               at=[i + 1, i + 2]))
    return out


def character_balance(present: list[bool], card: ChannelCard) -> list[Problem]:
    lo, hi = card.character_frames
    n = sum(present)
    if not lo <= n <= hi:
        return [Problem("character", f"{card.character_name} in {n} frames, "
                                     f"channel range {lo}-{hi}")]
    run = best = 0
    for p in present:
        run = run + 1 if p else 0
        best = max(best, run)
    if best > max(4, card.frames // 10):
        return [Problem("character", f"{card.character_name} in {best} frames in "
                                     f"a row; spread it across the video")]
    return []


# --- all together ------------------------------------------------------------

def check_narration(texts: list[str], card: ChannelCard,
                    seconds: list[float] | None = None) -> list[Problem]:
    if seconds is None:
        seconds = [estimate_seconds(t, rate=card.voice.syllables_per_second,
                                    pause=card.voice.sentence_pause_s)
                   for t in texts]
    return [
        *duplicate_frames(texts),
        *banned_scaffolds(texts),
        *repeated_openings(texts),
        *repeated_ngrams(texts),
        *paraphrased_neighbours(texts),
        *hook_quality(texts, seconds),
        *spoken_register(texts),
        *sentence_rhythm(texts),
        *abstract_density(texts),
        *ending_repeats_itself(texts),
        *unspeakable_syntax(texts),
        *viewer_address(texts),
        *questions(texts),
        *questions_on_the_seams(texts),
        *finale_returns_to_viewer(texts, seconds),
        *concreteness(texts),
        *production_leak(texts),
        *budget(texts, card),
    ]


def check_visuals(concepts: list[str], present: list[bool],
                  card: ChannelCard) -> list[Problem]:
    return visual_variety(concepts) + character_balance(present, card)


def missing_grounding(key_facts: list[str]) -> list[Problem]:
    """No facts is a note, not a repair: text cannot fix missing material."""
    if key_facts:
        return []
    return [Problem("grounding", "no facts supplied: the script will carry no "
                                 "numbers, because inventing them is forbidden",
                    advisory=True)]


def sentences_of(chapters: list[dict]) -> list[str]:
    """Chapter narration as sentences, the way the cutter will see it."""
    out: list[str] = []
    for c in chapters:
        out += split_sentences(str(c.get("narration", "")))
    return out
