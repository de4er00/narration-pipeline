"""Numbers in English text, whether written as digits or spelled out.

Scripts written for a narrator spell numbers out ("about twenty kilometers
across", "two hundred fifty times"), while facts usually carry digits. The
number checks compare both on the same canonical value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine".split())}
_TEENS = {w: i + 10 for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen".split())}
_TENS = {w: (i + 2) * 10 for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9,
           "trillion": 10 ** 12}
# "second" is left out: it is far more often the unit of time.
_ORDINALS = {
    "first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
    "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
    "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_ERAS = {"ad", "bc", "bce", "ce"}

# Counts that say nothing about the world ("three ways", "one gate").
HARMLESS = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                      "100", "1000"})

_TOKEN_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<join>[ \t\r\n]*-[ \t\r\n]*|[ \t\r\n]+)"
    r"|(?P<other>.)", re.S)


@dataclass(frozen=True)
class Number:
    value: str
    text: str


@dataclass
class _Tok:
    kind: str
    text: str
    start: int
    end: int

    @property
    def low(self) -> str:
        return self.text.lower()


def _canon(d: Decimal) -> str:
    d = d.normalize()
    return str(int(d)) if d == d.to_integral_value() else format(d, "f")


def _two_digit(toks: list[_Tok], k: int) -> tuple[int, int] | None:
    """A 10-99 group starting at k ("forty-seven", "nineteen") -> (value, next)."""
    if k >= len(toks) or toks[k].kind != "word":
        return None
    w = toks[k].low
    if w in _TEENS:
        return _TEENS[w], k + 1
    if w not in _TENS:
        return None
    value, nxt = _TENS[w], k + 1
    j = _skip(toks, nxt)
    if j < len(toks) and toks[j].low in _UNITS and toks[j].low != "zero":
        return value + _UNITS[toks[j].low], j + 1
    return value, nxt


def _skip(toks: list[_Tok], k: int) -> int:
    while k < len(toks) and toks[k].kind == "join":
        k += 1
    return k


def _read(toks: list[_Tok], i: int) -> tuple[Decimal, int] | None:
    """Read one number starting at token i -> (value, index after it)."""
    total, group = Decimal(0), Decimal(0)
    last: str | None = None
    last_scale = 10 ** 15
    end = i
    k = i
    while True:
        k = _skip(toks, k)
        if k >= len(toks):
            break
        t, w = toks[k], toks[k].low
        if t.kind == "num":
            if last is not None:
                break
            group, last, end = Decimal(t.text.replace(",", "")), "digit", k + 1
        elif t.kind != "word":
            break
        elif w in ("a", "an") and last is None:
            j = _skip(toks, k + 1)
            if j < len(toks) and (toks[j].low == "hundred" or toks[j].low in _SCALES):
                group, last = Decimal(1), "a"
            else:
                break
        elif w in _UNITS:
            if last is None or last in ("hundred", "scale", "and"):
                group += _UNITS[w]
            elif last == "tens" and group % 10 == 0:
                group += _UNITS[w]
            else:
                break
            last, end = "unit", k + 1
        elif w in _TEENS or w in _TENS:
            if last in ("unit", "teen", "tens") and total == 0 and group < 100:
                # Year style: "nineteen forty-seven", "twenty twenty-four", and
                # "five seventy-seven" only before an era ("five seventy-seven AD").
                pair = _two_digit(toks, k)
                if pair and (group >= 10 or _era_follows(toks, pair[1])):
                    return group * 100 + pair[0], pair[1]
                break
            if last is None or last in ("hundred", "scale", "and"):
                group += _TEENS.get(w) or _TENS[w]
            else:
                break
            last, end = ("teen" if w in _TEENS else "tens"), k + 1
        elif w == "hundred":
            if last not in ("unit", "teen", "tens", "a", "digit") or not 0 < group < 100:
                break
            group, last, end = group * 100, "hundred", k + 1
        elif w in _SCALES:
            if last not in ("unit", "teen", "tens", "hundred", "a", "digit") \
                    or _SCALES[w] >= last_scale or group == 0:
                break
            total += group * _SCALES[w]
            group, last, last_scale, end = Decimal(0), "scale", _SCALES[w], k + 1
        elif w == "and":
            j = _skip(toks, k + 1)
            nxt = toks[j].low if j < len(toks) else ""
            if last not in ("hundred", "scale") or not (nxt in _UNITS or nxt in _TEENS or nxt in _TENS):
                break
            last = "and"
        elif w == "point" and last in ("unit", "teen", "tens"):
            digits, j = "", _skip(toks, k + 1)
            while j < len(toks) and toks[j].low in _UNITS:
                digits += str(_UNITS[toks[j].low])
                end, j = j + 1, _skip(toks, j + 1)
            if not digits:
                break
            value = group + Decimal("0." + digits)
            j = _skip(toks, end)
            # "one point four million tonnes": the scale applies to the whole decimal.
            if j < len(toks) and toks[j].low in _SCALES and _SCALES[toks[j].low] < last_scale:
                return total + value * _SCALES[toks[j].low], j + 1
            return total + value, end
        elif w in _ORDINALS:
            v = _ORDINALS[w]
            if last is None or last in ("hundred", "scale", "and") or (
                    last == "tens" and group % 10 == 0 and v < 10):
                return total + group + v, k + 1
            break
        else:
            break
        k += 1
    if end == i:
        return None
    return total + group, end


def _era_follows(toks: list[_Tok], k: int) -> bool:
    letters = ""
    for t in toks[_skip(toks, k):_skip(toks, k) + 4]:
        if t.kind == "word":
            letters += t.low
        elif t.text != ".":
            break
    return letters in _ERAS


def find_numbers(text: str) -> list[Number]:
    """Every number in the text, in order, with its canonical value."""
    toks = [_Tok(m.lastgroup or "other", m.group(), m.start(), m.end())
            for m in _TOKEN_RE.finditer(text or "")]
    out: list[Number] = []
    i = 0
    while i < len(toks):
        got = _read(toks, i) if toks[i].kind in ("num", "word") else None
        if got is None:
            i += 1
            continue
        value, nxt = got
        out.append(Number(_canon(value), text[toks[i].start:toks[nxt - 1].end]))
        i = nxt
    return out


def figures(text: str) -> list[Number]:
    """Numbers that claim something about the world: not the small counts."""
    return [n for n in find_numbers(text) if n.value not in HARMLESS]
