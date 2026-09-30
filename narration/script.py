"""Stage 1 prompts: continuous prose under a word budget, and its repair."""
from __future__ import annotations

from .cards import VideoCard
from .checks import BANNED_SCAFFOLDS, CITATION_MARKERS, MIN_SHORT_SENTENCE_SHARE, Problem
from .research import Research

SCRIPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    # Strict structured output requires every property to be listed as
    # required, or the provider rejects the schema itself.
    "required": ["main_idea", "key_facts", "chapters", "final_cta", "next_video"],
    "properties": {
        "main_idea": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "narration"],
                "properties": {
                    "title": {"type": "string"},
                    "narration": {"type": "string"},
                },
            },
        },
        "final_cta": {"type": "string"},
        "next_video": {"type": "integer"},
    },
}

SYSTEM = (
    "You are a senior script writer for an explainer video channel. You write "
    "spoken US English for a single narrator. Respond with JSON only."
)


def _facts_block(video: VideoCard, research: Research | None) -> str:
    if research is not None:
        return research.as_prompt_block()
    return "\n".join(f"- {f}" for f in video.key_facts) or "- none supplied"


def max_numbers(target_seconds: int) -> int:
    """About one number per forty seconds of speech."""
    return max(4, round(target_seconds / 40))


def short_sentences_needed(sentences: int) -> int:
    # A count, not a share: asked for "one in six", the model twice resolved
    # the conflict with the sentence-count contract in favour of the contract.
    return max(6, round(sentences * MIN_SHORT_SENTENCE_SHARE * 1.5))


def next_video_block(siblings: list[str]) -> tuple[str, str]:
    """Hand-off to another video of the channel: (prompt block, output field).

    Empty for a new channel: promising a video that does not exist is worse
    than promising nothing.
    """
    if not siblings:
        return "", ""
    listed = "\n".join(f"  {i}. {title}" for i, title in enumerate(siblings, 1))
    block = f"""
AFTER the call to action, hand the viewer one more video from this channel.
Choose the ONE below whose subject follows from this ending — not the newest,
not a random one — and say in one spoken sentence why that video is the next
thing to watch. Return its NUMBER, not its text.

{listed}

Return 0 if none of them genuinely connects. A forced link is worse than none.
"""
    # A number, not a title: asked for a title, the model returned one that
    # was not on the list. A number is either in range or it is not.
    return block, ", next_video (the NUMBER from the list above, or 0)"


def resolve_next_video(raw: object, siblings: list[str]) -> str:
    try:
        i = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return siblings[i - 1] if 1 <= i <= len(siblings) else ""


def script_prompt(video: VideoCard, research: Research | None = None,
                  siblings: list[str] | None = None) -> str:
    ch = video.channel
    next_video, next_field = next_video_block(siblings or [])
    m, s = divmod(ch.target_seconds, 60)
    # Banned phrases are built from the constants the checks use, so the
    # prompt and the check can never drift apart.
    scaffolds = "\n".join(f'  - "{x}"' for x in BANNED_SCAFFOLDS)
    extra = (f"Extra restrictions: {video.extra_restrictions}"
             if video.extra_restrictions else "")
    return f"""ROLE
{ch.role}

THE VIDEO
Title: {video.title}
Main idea: {video.main_idea or "derive it from the title"}
MATERIAL YOU MAY USE
Each line carries a status and how firmly you are allowed to speak about
it. Obey the status. A hypothesis stated as fact costs the channel its
credibility; a hypothesis stated as a hypothesis costs nothing and is
just as interesting.
{_facts_block(video, research)}
{extra}

LENGTH CONTRACT — THIS IS A HARD REQUIREMENT
Write approximately {ch.words} words, within five percent. This number is not
a style preference: it is measured from this
channel's actual narrator voice ({ch.voice.label}, {ch.voice.syllables_per_second}
syllables per second), and it is what fills {m}:{s:02d} of speech. A different
number produces a video of the wrong length.

Write about {ch.sentences} sentences, averaging {ch.words_per_sentence} words
each. The video is cut into {ch.frames} separate frames and every frame needs
its own line, so a script built from long paragraph-length sentences cannot be
distributed — the cut is then forced to break a phrase in half, and the
narrator reads the fragment as a fragment. Short punchy sentences are welcome;
one twenty-five-word sentence in place of two is not.

RHYTHM — COUNT THESE BEFORE YOU ANSWER
At least {short_sentences_needed(ch.sentences)} of your {ch.sentences} sentences must be SIX WORDS OR
FEWER. Not merely shorter than the rest — six words or fewer, counted.

This does not break the length contract above: the word total stays the same
because the remaining sentences carry it. A script where every sentence lands
within a word or two of the average is exactly what monotony sounds like, and
a synthesized voice makes it worse.

A short sentence is a beat. Put one after something lands, before a turn, and
wherever the viewer needs half a second to feel what you just said. Repeating
a short opening across two or three consecutive sentences is a device, not a
defect — use it when the point is an accumulation.

Vary across the WHOLE video, not only at the start.

STRUCTURE
Write {8 if ch.frames < 70 else 10}-14 chapters of DELIBERATELY VARIED length.
Never use a fixed number of frames or sentences per chapter. Some chapters
carry one idea in two sentences; others develop across eight.

Chapter titles are viewer-facing: they become chapter markers on the video
page, so write them as short curiosity phrases, not as production labels.

NARRATIVE DIRECTION
{ch.narrative_direction}

THE HOOK — most viewers who leave, leave before 0:21
Open by putting the viewer INSIDE a situation, in the present tense, in the
second person. Not a statement about the subject — a place they are standing,
a body they are in, a moment happening to them right now. Build it from THIS
video's material; do not reach for a generic scene.

First sentence: SIX WORDS OR FEWER is ideal, twelve is the ceiling. It has to
be enterable at a run. Then two or three more that keep them inside the
situation before anything is explained.

BANNED for the first thirty seconds, without exception:
  - any organisation, agency, university, journal, telescope or study by name;
  - "according to", "researchers", "a study found", "data from";
  - any figure that reads as a citation rather than as a sensation.
A number is allowed there only when the viewer feels it on their own body —
their weight, their night's sleep, how long ago this was in human lifetimes.
A figure carrying a source, an instrument or a confirmation is the other kind,
and it is exactly where viewers leave.

Never ask the viewer to stay. A plea to keep watching tells them the next part
is boring. Hold attention by opening a gap and withholding the answer.

Earn the next thirty seconds with a promise, then keep withholding it.

BANNED FROM THE OPENING: any caveat, disclaimer, definition, methodology note
or framing sentence. Never open by explaining that something is hypothetical,
that evidence is limited, or that we should separate fact from speculation.
If the subject genuinely needs that framing, it goes AFTER the hook has
landed, around the thirty-second mark. A video that opens by defusing its own
premise loses the viewer before it starts.

ORIGINALITY — this is where form-filled scripts fail
Write every sentence for its exact moment. Do not expand one fact across
several sentences with a fixed pattern.

These scaffolds are forbidden outright:
{scaffolds}

No opening phrase of four or more consecutive words may appear more than
twice. Never restate the previous sentence to fill time: if a fact needs one
sentence, give it one and move on.

VOICE — spoken aloud by a narrator, not read off a page
A sentence that looks correct in an article sounds like a report when a voice
reads it.

  - ALWAYS contract: you're, don't, it's, that's, they'd, we've, can't.
    Write "you are" only when the stress genuinely falls on "are".
  - One-word and two-word sentences are correct here. Use them for the beat
    after something lands.
  - NEVER use a semicolon. Nobody speaks a semicolon.
  - Never list four or more things in a row. Three is a rhythm, five is noise.
  - Let the narrator interrupt himself, undercut himself, be wry. One dry
    modern aside per chapter is worth more than a paragraph of explanation —
    typically by measuring the past against an ordinary modern convenience,
    or by naming the blunt physical truth of a situation. Invent the joke from
    THIS video's material. Be funny.

Address the viewer directly as "you" — at least once every hundred words.
Ask at least three real questions, and SPREAD THEM ACROSS THE SCRIPT rather
than saving them for the end. A question belongs on the seam between chapters:
it closes what you just showed and opens what comes next, and it is built from
what that chapter just established. A narrator who never asks anything reads
as a machine, and a question is one of the few ways to break the monotony of
synthesized speech.

CONCRETENESS — and its ceiling
Prefer the concrete: mechanisms, comparisons a viewer can picture, named
consequences. Numbers work, but they are seasoning, not the dish.

NOUNS ENDING IN -TION, -SION, -MENT, -NESS, -ITY, -ANCE, -ISM ARE THE PROBLEM.
They are how a sentence stops being about anything you can see. Name the thing
happening, not the -tion of the thing: someone decides rather than makes a
decision, the fire goes out rather than experiences extinction, it gets colder
rather than undergoes a reduction in temperature. Give the verb back to
whoever or whatever is doing it.

Use AT MOST {max_numbers(ch.target_seconds)} numbers in the entire script — roughly one every
forty seconds of speech. There is no minimum. Use the figures the material
actually gives you and no others: if it offers two, use two and carry the rest
of the video on mechanism, comparison and consequence. Never stretch a figure
across several sentences to make the script look better sourced, and never
reach for a number the material does not contain.
Say every figure the way a narrator says it out loud: rounded to two or three
significant digits, never a long decimal or a full accounting figure. A
rounded figure from the material is still that figure.
A script that recites figures sounds like a report being read aloud, and that
is the one failure that matters here — the video has to stay interesting.
Dates and organisation names are the easiest way to sound informative and the
fastest way to lose a viewer.
If two pieces of material make the same point, take the more vivid one and
drop the other.

Every number you use must come from the material above or be an explicitly
framed example. Invent no quotations, studies, documents or exact figures.
Where the material marks something disputed, build the disagreement into
the story rather than hiding it — the disagreement is usually the better
scene.

Never mention the script, the narration, the storyboard or the frames. The
viewer does not know any of that exists.

ENDING — the last thirty seconds decide whether the video felt worth it
Resolve the promise made in the hook. Then do the thing that separates a video
people remember from a summary: turn the subject back onto the viewer's own
life today. Explain what this means for them and why they feel or behave the
way they do. The video must stop being only about its subject and become about
the person watching.

Do not end on a tidy conclusion about the topic. A sentence that correctly
summarises what the video established is still a dead ending: it closes the
subject instead of opening the viewer.

Then close with the call to action, and let it GROW OUT OF that ending rather
than being bolted on. It is one question to the viewer, and it asks exactly
this: {ch.cta}
Say it in your own spoken words, but never swap it for advice, a task for the
viewer's day, or a request to share or subscribe.
{next_video}
OUTPUT
JSON with: main_idea (one sentence), key_facts (array of the concrete claims
the finished script actually makes), chapters (array of title + narration),
final_cta{next_field}.
The narration field of each chapter is continuous prose, not a list."""


def repair_prompt(video: VideoCard, chapters: list[dict], problems: list[Problem],
                  final_cta: str = "", research: Research | None = None,
                  siblings: list[str] | None = None) -> str:
    """The whole script back to the model with a located list of problems.

    The repair sees the same material, siblings, call to action and rhythm
    rule as the first pass. Without them it fixed "unsourced number" blind,
    topped up length with invention, rewrote the call to action as advice and
    merged short sentences, and the second repair went on fixing what the
    first one broke.
    """
    ch = video.channel
    next_video, _ = next_video_block(siblings or [])
    listed = "\n".join(f"- {p}" for p in problems)
    body = "\n\n".join(f"### {c['title']}\n{c['narration']}" for c in chapters)
    scaffolds = "\n".join(f'   - "{x}"' for x in BANNED_SCAFFOLDS)
    citations = "\n".join(f'   - "{x}"' for x in CITATION_MARKERS)
    return f"""Rewrite this script so it satisfies EVERY requirement below at
once. Keep its facts, its structure and everything that already works.
Invent nothing new.

WHAT IS WRONG NOW (positions count sentences from the start of the script):
{listed}

REQUIREMENTS THAT MUST ALL HOLD IN YOUR OUTPUT:
1. About {ch.words} words total, within five percent, and about
   {ch.sentences} sentences. This is a floor as much as a ceiling. Fixing a
   repetition means REPLACING the weak sentence with something else from the
   material below or from the script itself, never deleting it: a shorter
   script is a failed repair, not a fixed one. Count
   your words before you answer.
2. No sentence repeated. No opening phrase of four or more words used more
   than twice. None of the banned scaffolds.
2a. Use at most {max_numbers(ch.target_seconds)} numbers in the whole
   script. There is no minimum — a script with no figures at all is fine if
   the material has none. Turn surplus figures into mechanisms or comparisons.
   Say each figure rounded, the way a person says it aloud.
2b. The opening puts the viewer inside a scene in the second person and the
   present tense. First sentence twelve words at the outside, six is better.
2c. The first thirty seconds name no organisation, university, agency, journal,
   telescope or study, and contain none of these:
{citations}
   A number is allowed there only if the viewer feels it on their own body,
   never as a citation.
2d. None of these phrases anywhere, and no plea to keep watching:
{scaffolds}
3. No sentence that merely paraphrases the one before it.
4. No caveat, disclaimer or methodology note in the first thirty seconds.
4a. Spoken register throughout: contract wherever a person would (you're,
   don't, it's), no semicolons at all, no list of four or more items in a row.
   Fragments are welcome. This is read aloud.
5. At least one "you" per hundred words.
6. Never mention the script, narration, storyboard or frames.
7. Chapter titles stay viewer-facing curiosity phrases.
8. At least three questions, spread across the script and not bunched at the
   end — each on a seam between chapters.
9. The last thirty seconds speak to the viewer's own life today, not to the
   past, and the call to action grows out of that.
10. The call to action is one question to the viewer, and it asks exactly
   this: {ch.cta} If the current one below already does, return it unchanged.
   Never swap it for advice, a task for the viewer's day, or a request to
   share or subscribe.
11. At least {short_sentences_needed(ch.sentences)} sentences of six words or fewer, spread across the
   whole script. When you add words, add them to the long sentences, never by
   gluing the short ones together.

Return the same JSON shape: main_idea, key_facts, chapters, final_cta,
next_video.
{next_video}
MATERIAL YOU MAY USE — the only facts and figures allowed; obey each status:
{_facts_block(video, research)}

CURRENT CALL TO ACTION:
{final_cta or "(none yet)"}

CURRENT SCRIPT:
{body}"""
