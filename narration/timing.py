"""Speech duration from syllables, and the exact partition into frames.

A frame never breaks a sentence unless it has to: in hand-cut production
scripts every frame ended on a full stop. Clause breaks are a fallback for
sentences longer than a frame or texts with fewer sentences than frames.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Measured medians on eleven production scripts written in the spoken
# register. Contractions ("you're") remove a syllable per word, which moved
# the density from 1.76 to 1.52; on the old value scripts written exactly to
# budget came out up to 10% short. The two constants are a pair: the pause is
# charged per sentence, so changing one without the other shifts the budget.
SYLLABLES_PER_WORD = 1.52
SYLLABLES_PER_SENTENCE = 20.7

# Pace is a property of the voice. Measured per voice from two recordings
# (long sentences and short ones), which gives two equations for two unknowns:
#
#     duration = syllables / rate + sentences * pause
#
# Across five production voices the rate ranged from 3.66 to 6.30 syllables
# per second and the pause from 0.47 to 0.81 s, so the same 1,420 words ran
# 8.0 minutes with one voice and 13.0 with another. These defaults are only
# for tests; real values come from the channel card.
SYLLABLES_PER_SECOND = 4.75
SENTENCE_PAUSE_S = 0.62

_VOWELS = "aeiouy"
_WORD_RE = re.compile(r"[A-Za-z']+")

_COORD = r"(?:and|but|or|so|yet|for|nor)"
_SUBORD = (r"(?:because|when|while|if|although|though|since|after|before|"
           r"unless|until|whereas|as)")
# "as" is excluded from bare breaks: it lives in comparisons ("as big as a
# city"), and splitting there destroys the phrase the sentence exists for.
_SUBORD_BARE = (r"(?:because|when|while|although|though|since|unless|until|"
                r"whereas)")
# Ordered by how much meaning a break at that point tears. Group 1 captures
# the punctuation so it stays in the left chunk: the narration is rebuilt by
# joining frames, and a lost semicolon makes the sentence ungrammatical.
_SOFT_BREAKS: tuple[tuple[str, int], ...] = (
    (r"([;:])\s+", 1),
    (r"\s+([—–])\s+", 2),
    (rf"(,)\s+(?={_COORD}\s)", 3),
    (rf"(,)\s+(?={_SUBORD}\s)", 4),
    (r"(,)\s+", 5),
    (rf"\s+()(?={_SUBORD_BARE}\s)", 6),
)

# A two-word fragment on its own sounds like a recording glitch, not a frame.
MIN_CHUNK_WORDS = 4


class TimingError(RuntimeError):
    """The narration cannot be laid out on the requested number of frames."""


def count_syllables(word: str) -> int:
    """Vowel-group heuristic. Per-word errors average out over a frame."""
    w = word.lower().strip("'")
    if not w:
        return 0
    groups = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


def syllables_in(text: str) -> int:
    return sum(count_syllables(w) for w in _WORD_RE.findall(text))


def estimate_seconds(text: str, *, rate: float = SYLLABLES_PER_SECOND,
                     pause: float = SENTENCE_PAUSE_S) -> float:
    if not text.strip():
        return 0.0
    sentences = len(_SENTENCE_SPLIT.findall(text)) + 1
    return syllables_in(text) / rate + sentences * pause


# A full stop after an abbreviation is not a sentence end. "U.S. Bank says..."
# once produced a one-word, one-second frame. Titles always precede a name, so
# they never end a sentence; "Inc.", "Ltd." and "etc." often do and are left
# out on purpose. Erring towards not splitting is safer: two merged sentences
# make a longer frame, a wrong split makes a broken one.
_ABBREV = "".join(
    rf"(?<!{a})" for a in (
        r"[A-Za-z]\.[A-Za-z]\.",
        r"\bMr\.", r"\bMs\.", r"\bMrs\.", r"\bDr\.", r"\bProf\.",
        r"\bSt\.", r"\bJr\.", r"\bSr\.", r"\bNo\.",
        r"\bLt\.", r"\bCmdr\.", r"\bCapt\.", r"\bCol\.", r"\bGen\.",
        r"\bSgt\.", r"\bMaj\.", r"\bAdm\.", r"\bGov\.", r"\bSen\.",
        r"\bRep\.", r"\bCpl\.", r"\bPvt\.", r"\bRev\.", r"\bHon\.",
    )
)
_SENTENCE_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])[\"'”’)]*\s+")

# Second line of defence for abbreviations the list does not know: a chunk
# made of a single abbreviation is a fragment of the next sentence. Checked by
# shape, not length, because short sentences ("Why?") are wanted.
_ABBREV_ONLY_RE = re.compile(
    r"^(?:[A-Za-z]\.){2,}$"
    r"|^(?:Mr|Ms|Mrs|Dr|Prof|St|Jr|Sr|No|Etc|Inc|Ltd|Co|vs|Lt|Cmdr|Capt|Col"
    r"|Gen|Sgt|Maj|Adm|Gov|Sen|Rep|Cpl|Pvt|Rev|Hon)\.$",
    re.IGNORECASE,
)


@dataclass
class Unit:
    """An indivisible piece: a sentence, or a clause when one had to be split.

    `hard` means the piece ends a sentence. Breaking a frame there is free;
    breaking after a soft piece costs a penalty.
    """
    text: str
    seconds: float
    hard: bool = True

    @property
    def words(self) -> int:
        return len(_WORD_RE.findall(self.text))


def _exclamation_continues(prev: str, nxt: str) -> bool:
    """"the Wow! Signal": an exclamation mark inside a name, not an ending."""
    if not prev.endswith("!") or not nxt:
        return False
    if nxt[:1].islower():
        return True
    words = prev[:-1].split()
    last = words[-1].strip("\"'“”‘’(") if words else ""
    return len(words) >= 2 and last[:1].isupper() and nxt[:1].isupper()


def split_sentences(text: str) -> list[str]:
    """Text to sentences. Regrouping only: not a character of text changes."""
    parts = [p for p in
             (p.strip() for p in _SENTENCE_SPLIT.split(" ".join(text.split())))
             if p]
    merged: list[str] = []
    for p in parts:
        if merged and _exclamation_continues(merged[-1], p):
            merged[-1] = f"{merged[-1]} {p}"
        else:
            merged.append(p)
    out: list[str] = []
    carry = ""
    for p in merged:
        if carry:
            p, carry = f"{carry} {p}", ""
        if _ABBREV_ONLY_RE.match(p):
            carry = p
            continue
        out.append(p)
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


def _split_once(sentence: str) -> tuple[str, str] | None:
    """Split at the best available clause, as close to the middle as possible."""
    if len(_WORD_RE.findall(sentence)) < MIN_CHUNK_WORDS * 2:
        return None
    best: tuple[int, float, int] | None = None
    mid = len(sentence) / 2
    for pattern, prio in _SOFT_BREAKS:
        for m in re.finditer(pattern, sentence):
            pos = m.end(1)
            left, right = sentence[:pos].strip(), sentence[pos:].strip()
            if (len(_WORD_RE.findall(left)) < MIN_CHUNK_WORDS
                    or len(_WORD_RE.findall(right)) < MIN_CHUNK_WORDS):
                continue
            key = (prio, abs(pos - mid), pos)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    pos = best[2]
    return sentence[:pos].strip(), sentence[pos:].strip()


def build_units(text: str, *, max_seconds: float, min_units: int = 0,
                rate: float = SYLLABLES_PER_SECOND,
                pause: float = SENTENCE_PAUSE_S) -> list[Unit]:
    """Text to pieces a frame can take whole.

    A sentence is split when it is longer than `max_seconds`, or when there
    are fewer pieces than frames and nothing else can fill them.
    """
    def unit(s: str, hard: bool) -> Unit:
        return Unit(s, estimate_seconds(s, rate=rate, pause=pause), hard)

    units = [unit(s, True) for s in split_sentences(text)]

    changed = True
    while changed:
        changed = False
        out: list[Unit] = []
        for u in units:
            if u.seconds > max_seconds and (pair := _split_once(u.text)):
                out += [unit(pair[0], False), unit(pair[1], u.hard)]
                changed = True
            else:
                out.append(u)
        units = out

    while len(units) < min_units:
        # Take the longest piece that can actually split. Taking the longest
        # piece blindly failed whenever it had no clause boundary, even though
        # shorter pieces could still be divided.
        idx = pair = None
        for i in sorted(range(len(units)), key=lambda i: -units[i].seconds):
            pair = _split_once(units[i].text)
            if pair is not None:
                idx = i
                break
        if pair is None or idx is None:
            raise TimingError(
                f"the narration splits into only {len(units)} pieces but "
                f"{min_units} frames are needed; the text is too short")
        units[idx:idx + 1] = [unit(pair[0], False), unit(pair[1], units[idx].hard)]
    return units


# Penalty for a frame that ends mid-sentence, in seconds squared: break a
# sentence only if it saves more than two seconds of deviation. At 1.0 the
# cutter broke sentences where a human editor did not; at 4.0 it matched them.
SOFT_BREAK_PENALTY = 4.0

# 30% of production sentences were longer than a six-second frame and all of
# them sat in one frame, so only sentences twice the frame length are pre-split.
PRESPLIT_RATIO = 2.0


@dataclass
class Frame:
    number: int
    text: str
    seconds: float
    start: float
    ends_sentence: bool = True

    @property
    def words(self) -> int:
        return len(_WORD_RE.findall(self.text))


@dataclass
class Segmentation:
    frames: list[Frame] = field(default_factory=list)
    target_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return sum(f.seconds for f in self.frames)

    @property
    def worst_drift(self) -> float:
        """Largest gap between picture and voice if every frame gets
        `target_seconds` on screen. Slot-filled scripts reached 163 s."""
        worst = spoken = 0.0
        for i, f in enumerate(self.frames, start=1):
            spoken += f.seconds
            worst = max(worst, abs(spoken - i * self.target_seconds))
        return worst

    @property
    def broken_sentences(self) -> int:
        return sum(1 for f in self.frames if not f.ends_sentence)


def partition(units: list[Unit], n_frames: int, target_seconds: float,
              ) -> list[list[Unit]]:
    """Split units into exactly `n_frames` contiguous groups, optimally.

    Minimises squared deviation from the target duration plus a penalty for
    each group that ends mid-sentence; exact DP in O(n_frames * m^2). Against
    a greedy pass the total drift is about the same (the text length sets it);
    the DP breaks fewer sentences and halves the spread of frame lengths.
    """
    m = len(units)
    if n_frames <= 0:
        raise TimingError("the number of frames must be positive")
    if m < n_frames:
        raise TimingError(f"{m} pieces cannot fill {n_frames} frames")

    pref = [0.0] * (m + 1)
    for i, u in enumerate(units):
        pref[i + 1] = pref[i] + u.seconds

    def group_cost(lo: int, hi: int) -> float:
        dev = (pref[hi] - pref[lo]) - target_seconds
        return dev * dev + (0.0 if units[hi - 1].hard else SOFT_BREAK_PENALTY)

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n_frames + 1)]
    back = [[0] * (m + 1) for _ in range(n_frames + 1)]
    dp[0][0] = 0.0
    for k in range(1, n_frames + 1):
        for i in range(k, m - (n_frames - k) + 1):
            best, best_j = inf, k - 1
            for j in range(k - 1, i):
                prev = dp[k - 1][j]
                if prev == inf:
                    continue
                c = prev + group_cost(j, i)
                if c < best:
                    best, best_j = c, j
            dp[k][i] = best
            back[k][i] = best_j

    if dp[n_frames][m] == inf:
        raise TimingError("no partition found")
    groups: list[list[Unit]] = []
    i = m
    for k in range(n_frames, 0, -1):
        j = back[k][i]
        groups.append(units[j:i])
        i = j
    groups.reverse()
    return groups


def _frames_from_groups(groups: list[list[Unit]], first_number: int,
                        clock: float, frame_seconds: float,
                        uniform: bool) -> list[Frame]:
    frames = []
    for n, g in enumerate(groups, start=first_number):
        secs = sum(u.seconds for u in g)
        start = (n - 1) * frame_seconds if uniform else clock
        frames.append(Frame(number=n, text=" ".join(u.text for u in g),
                            seconds=secs, start=start,
                            ends_sentence=g[-1].hard))
        clock += secs
    return frames


def segment(text: str, n_frames: int, *, frame_seconds: float = 6.0,
            rate: float = SYLLABLES_PER_SECOND,
            pause: float = SENTENCE_PAUSE_S, uniform: bool = True) -> Segmentation:
    """Narration to frames.

    `uniform=True` puts frames on a regular grid; `uniform=False` uses each
    frame's measured duration, so the timeline cannot drift by construction.
    """
    units = build_units(text, max_seconds=frame_seconds * PRESPLIT_RATIO,
                        min_units=n_frames, rate=rate, pause=pause)
    groups = partition(units, n_frames, frame_seconds)
    return Segmentation(
        frames=_frames_from_groups(groups, 1, 0.0, frame_seconds, uniform),
        target_seconds=frame_seconds)


def max_units(text: str, *, rate: float = SYLLABLES_PER_SECOND,
              pause: float = SENTENCE_PAUSE_S) -> int:
    """How many pieces the text can physically yield (a section's capacity)."""
    return len(build_units(text, max_seconds=0.0, rate=rate, pause=pause))


def allocate_frames(section_seconds: list[float], n_frames: int,
                    capacity: list[int] | None = None) -> list[int]:
    """Distribute frames over sections in proportion to their duration.

    Largest-remainder method with at least one frame per section. `capacity`
    caps each section at the number of pieces it can yield: without it the
    proportion once gave a chapter eight frames when it could split into
    seven, and the cut failed after the script had been paid for.
    """
    k = len(section_seconds)
    if k == 0:
        raise TimingError("no sections")
    if n_frames < k:
        raise TimingError(f"{n_frames} frames for {k} sections: not one each")
    if capacity is not None:
        if len(capacity) != k:
            raise TimingError("capacity length does not match the sections")
        if sum(capacity) < n_frames:
            raise TimingError(
                f"the sections split into at most {sum(capacity)} pieces but "
                f"{n_frames} frames are needed; the script needs shorter "
                f"sentences")
    total = sum(section_seconds)
    if total <= 0:
        raise TimingError("the sections have zero total duration")

    exact = [s / total * n_frames for s in section_seconds]
    base = [max(1, int(x)) for x in exact]
    while sum(base) > n_frames:
        spare = [i for i in range(k) if base[i] > 1]
        if not spare:
            raise TimingError(f"{n_frames} frames cannot cover {k} sections")
        i = max(spare, key=lambda i: base[i] - exact[i])
        base[i] -= 1
    cap = capacity or [n_frames] * k
    for i in range(k):
        base[i] = min(base[i], max(1, cap[i]))
    order = sorted(range(k), key=lambda i: exact[i] - base[i], reverse=True)
    pos = guard = 0
    while sum(base) < n_frames:
        i = order[pos % k]
        pos += 1
        guard += 1
        if base[i] < cap[i]:
            base[i] += 1
            guard = 0
        elif guard > k:
            raise TimingError(
                f"frames do not fit: free capacity {sum(cap)}, needed {n_frames}")
    return base


def segment_sections(sections: list[tuple[str, str]], n_frames: int, *,
                     frame_seconds: float = 6.0,
                     rate: float = SYLLABLES_PER_SECOND,
                     pause: float = SENTENCE_PAUSE_S,
                     uniform: bool = True) -> tuple[Segmentation, list[int]]:
    """(title, text) sections to frames; no frame crosses a section boundary.

    A frame that holds the end of one chapter and the start of the next has
    two subjects and cannot be illustrated. Returns the cut and the section
    index of every frame.
    """
    if not sections:
        raise TimingError("no sections")
    secs = [estimate_seconds(text, rate=rate, pause=pause) for _, text in sections]
    cap = [max_units(text, rate=rate, pause=pause) for _, text in sections]
    quota = allocate_frames(secs, n_frames, cap)

    frames: list[Frame] = []
    owner: list[int] = []
    for idx, ((_, text), want) in enumerate(zip(sections, quota)):
        units = build_units(text, max_seconds=frame_seconds * PRESPLIT_RATIO,
                            min_units=want, rate=rate, pause=pause)
        clock = sum(f.seconds for f in frames)
        frames += _frames_from_groups(partition(units, want, frame_seconds),
                                      len(frames) + 1, clock, frame_seconds,
                                      uniform)
        owner += [idx] * want
    return Segmentation(frames=frames, target_seconds=frame_seconds), owner


def words_budget(n_frames: int, frame_seconds: float, rate: float,
                 pause: float = SENTENCE_PAUSE_S, *,
                 syllables_per_word: float = SYLLABLES_PER_WORD,
                 syllables_per_sentence: float = SYLLABLES_PER_SENTENCE) -> int:
    """Words that fill the runtime with this voice. Goes into the prompt."""
    seconds_per_syllable = 1 / rate + pause / syllables_per_sentence
    syllables = n_frames * frame_seconds / seconds_per_syllable
    return int(round(syllables / syllables_per_word))
