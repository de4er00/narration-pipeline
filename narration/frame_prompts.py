"""Image and animation prompts per frame, assembled by code without a model.

400 hand-made production prompts had no place that needed a decision, only
the channel's constant text plus stage 2's choices. In code, the motion fields
cannot go missing (they were absent from all 400) and nothing can leak in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cards import ChannelCard

NEGATIVE_IMAGE = (
    "no subtitles, no subtitle text, no caption bar, no closed captions, no "
    "burned-in timecode, no timestamp overlay, no narration text on screen, "
    "no quote text rendered on image, no transcript text, no video player UI, "
    "no progress bar, no filename, no project code, no non-Latin text, no "
    "watermark, no generator signature, no unintended text, no real brand "
    "logos, no duplicate character, no extra limbs, no duplicate hands, no "
    "fused fingers, no detached hands, no malformed anatomy, no cropped key "
    "object, no style drift"
)

# Shorter than the image negative on purpose. An image-to-video model
# reproduces the frame it is given, so it does not invent subtitles or
# watermarks, but it does read every noun listed here. The video models this
# was tuned on have no negative-prompt parameter: the text is appended to the
# prompt, and "no sparkles" puts sparkles into the condition. Add to it only
# after measuring on real clips. The dust and particle bans are measured: one
# model added over 2,000 small bright specks to a single clip.
NEGATIVE_ANIMATION = (
    "no morphing, no identity drift, no scene redesign, no new characters, "
    "no new objects, no deformed or duplicated hands, no lip sync, "
    "no 3D conversion, no random camera movement, no strong motion blur, "
    "no camera shake, no floating dust, no drifting particles, no sparkles, "
    "no added glow, no film grain"
)
NEGATIVE_ANIMATION_TEXT = (
    ", no text changes, no number changes, no warped typography, "
    "no flickering text"
)

# Hand-made prompts used one shot size for all sixty frames of a video. Stage
# 2 picks a label; the wording is fixed so every frame of a type reads alike.
SHOTS: dict[str, str] = {
    "wide": "wide establishing shot, the subject occupies about a quarter of "
            "the frame height, generous surrounding space",
    "medium": "medium editorial shot, the subject occupies about half of the "
              "frame height, readable foreground and background separation",
    "close": "close detail shot, one element fills most of the frame, "
             "background reduced to a simple field",
}
DEFAULT_SHOT = "medium"

_FORBIDDEN = (
    (re.compile(r"\[?\b\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?\b\]?"), "a timecode"),
    (re.compile(r"[\u0400-\u04FF]"), "Cyrillic text"),
    (re.compile(r"[\"“”«»][^\"“”«»]{8,}[\"“”«»]"), "a quoted narration line"),
    # A frame file name, not any ".png": the reference sheets are named in the
    # prompt on purpose.
    (re.compile(r"_F\d{3}_\d{4}\.png", re.I), "a frame file name"),
)


class PromptError(RuntimeError):
    """A prompt was assembled with forbidden content. It never reaches a model."""


@dataclass
class FrameVisual:
    """What stage 2 decided about one frame. Everything else is the card's."""
    number: int
    scene_concept: str
    character_present: bool
    shot: str = DEFAULT_SHOT
    character_action: str = ""
    object_placement: str = ""
    visible_text: str = ""
    main_motion: str = ""
    secondary_motion: str = ""
    camera: str = "locked camera"

    def __post_init__(self) -> None:
        if self.shot not in SHOTS:
            self.shot = DEFAULT_SHOT
        # Silence is worse than a wrong guess here: the image model would give
        # the character an arbitrary action that contradicts the line.
        if self.character_present and not self.character_action:
            raise PromptError(f"frame {self.number}: the character is present "
                              f"but has no action")


def reference_inputs(card: ChannelCard, visual: FrameVisual) -> list[str]:
    """Sheets to attach, in order. The character sheet goes only where the
    character is: attached to an object frame it asks for the character."""
    refs = [card.character_sheet] if visual.character_present else []
    return refs + [card.style_sheet]


def _guard(text: str, where: str) -> str:
    for rx, what in _FORBIDDEN:
        m = rx.search(text)
        if m:
            raise PromptError(f"{where}: the prompt contains {what}: "
                              f"{m.group(0)!r}")
    return text


def build_image_prompt(card: ChannelCard, visual: FrameVisual) -> str:
    """The attached sheets plus what makes THIS frame different.

    No words describe the style or the character: text next to a sheet became
    a second voice that sometimes contradicted it, and a 4,282-character
    prompt matched the sheet's style worse than a 702-character one.
    """
    name = card.character_name
    # Sheets are referred to by position, not by file name: the model gets an
    # unnamed array of images. The same frame pointed at "the first" and "the
    # second" sheet matched the style on the first try; pointed at by file
    # name, it lost both the canvas and the character's face.
    if visual.character_present:
        refs = (f"REFERENCE INPUTS: Two sheets are attached in this order, and they "
                f"are the whole specification. The FIRST is the character sheet "
                f"({card.character_sheet}): copy {name}'s design from it exactly — "
                f"every part, colour and proportion it shows, and nothing it does "
                f"not. The SECOND is the style sheet ({card.style_sheet}): copy "
                f"its medium, paper, line quality, palette, shading and detail "
                f"level exactly. Do not redesign either sheet")
        presence = (f"CHARACTER PRESENCE: PRESENT. {name} is copied from the "
                    f"character sheet exactly and is never redesigned.")
        action = f"CHARACTER POSITION AND ACTION: {visual.character_action.rstrip('.')}."
    else:
        refs = (f"REFERENCE INPUTS: One sheet is attached and it is the whole "
                f"specification: the style sheet ({card.style_sheet}). Copy its "
                f"medium, paper, line quality, palette, shading and detail level "
                f"exactly. Do not redesign the sheet")
        presence = (f"CHARACTER PRESENCE: ABSENT. Do not add {name} or any "
                    f"substitute presenter. People, animals and other figures "
                    f"appear only as the scene concept describes them.")
        action = "CHARACTER POSITION AND ACTION: NONE."
    placement = (f"OBJECT PLACEMENT: {visual.object_placement.rstrip('.')}."
                 if visual.object_placement else
                 "OBJECT PLACEMENT: the dominant object sits in the centre of "
                 "the frame, clearly separated from the background.")
    text = visual.visible_text.strip()
    parts = [
        # No pixel size: the size is a request parameter, and the models in
        # use return 1376x768 and 1672x941, not the 1920x1080 old templates named.
        "FORMAT LOCK: One standalone horizontal 16:9 video frame, not a collage.",
        refs + " and never use a reference from another channel.",
        presence,
        f"SCENE CONCEPT: {visual.scene_concept.rstrip('.')}.",
        f"COMPOSITION: {SHOTS[visual.shot]}; one focal idea, no more than three "
        f"supporting objects, generous safe margins.",
        action,
        placement,
        f"VISIBLE TEXT: {text}." if text else "VISIBLE TEXT: NONE.",
        f"NEGATIVE PROMPT: {NEGATIVE_IMAGE}",
    ]
    return _guard("\n".join(parts), f"frame {visual.number}, image prompt")


# A tail like ", then stops" contradicts the constant-speed line. A separable
# tail is cut; an inseparable one ("slows to a stop") stays, and the speed line
# is softened instead, because a contradiction is worse than a slowdown.
_STOP_TAIL = re.compile(
    r",?\s*(?:and\s+)?then\s+(?:stops?|pauses?|holds?|freezes?|rests?|settles?)\b.*$"
    r"|,?\s*and\s+(?:stops?|pauses?|freezes?)\b[^,]*$",
    re.IGNORECASE)
_STOP_WORDS = re.compile(
    r"\b(?:stops?|stopping|pauses?|slows?|settles?|holds?\s+still)\b", re.IGNORECASE)
_SPEED_EVEN = (
    "Speed: constant and uniform throughout the shot, already moving at the "
    "first frame and continuing through the final frame; no visible start, "
    "stop, acceleration, deceleration, reversal or loop.")
_SPEED_SOFT = "Speed: even and unhurried, without abrupt changes, reversal or loop."


def single_action(text: str) -> str:
    main = (text or "").strip() or "hold the composition nearly static"
    cut = _STOP_TAIL.sub("", main).strip(" ,;")
    return cut if len(cut.split()) >= 3 else main


def build_animation_prompt(card: ChannelCard, visual: FrameVisual) -> str:
    """Motion only: the video model also receives the frame itself.

    Hand-made prompts spent 53% of the text retelling the picture and 6% on
    motion. One action at one speed, because two movements or a stop inside
    the shot make the model redraw every frame and the clip shakes. The camera
    is a barely perceptible push-in: on 77 clips whole-frame jitter hit 10% of
    push-in clips and 30-56% of locked-camera ones.
    """
    main = single_action(visual.main_motion)
    text = visual.visible_text.strip()
    parts = [
        "Animate the attached still image as a moving version of this exact "
        "drawing. Keep its composition, crop, palette, linework, texture, "
        "typography and every existing figure and object unchanged.",
        f"Single action: {main.rstrip('.')}.",
        _SPEED_SOFT if _STOP_WORDS.search(main) else _SPEED_EVEN,
        "Secondary motion: none.",
        "Camera: a very slow, barely perceptible push-in at one constant "
        "speed from the first frame to the last.",
    ]
    # The character is not named: the video model sees the frame but has no
    # idea who the name refers to. Point at what it can see.
    if visual.character_present:
        parts += [
            "Characters: keep every figure exactly as drawn in the source "
            "frame — same face, hair, clothing, proportions and line weight. "
            "Do not redraw, restyle or re-pose them beyond the single action "
            "above.",
            "Props: only a prop the figure already holds may follow it as part "
            "of that same mechanically linked action; every other prop stays "
            "fixed.",
        ]
    else:
        parts.append(
            "Characters: keep every figure the frame already shows exactly as "
            "drawn. Do not introduce a new person, creature, presenter or figure.")
    parts.append(
        "Apart from the camera, keep the background, papers, particles, clothing "
        "details, diagrams and every element outside the single action "
        "perfectly static.")
    # Text is mentioned only when the frame has some; otherwise it is an
    # invitation to add it. It is quoted because video models rarely erase a
    # caption, they reshuffle its letters.
    if text:
        parts.append(
            f"Visible text: the frame already contains the text "
            f"{text.rstrip('.')} — keep it exactly as drawn, same wording, same "
            f"lettering, static and legible; do not re-letter, warp, translate "
            f"or animate it.")
    parts.append(
        "Output: smooth clean non-looping motion, horizontal 16:9, flat "
        "hand-drawn 2D look throughout; no added glow, lighting, depth or 3D.")
    negative = NEGATIVE_ANIMATION + (NEGATIVE_ANIMATION_TEXT if text else "")
    parts.append(f"Negative prompt: {negative}")
    return _guard("\n".join(parts), f"frame {visual.number}, animation prompt")


@dataclass
class PromptSet:
    number: int
    image: str
    animation: str
    references: list[str] = field(default_factory=list)


def build_all(card: ChannelCard, visuals: list[FrameVisual]) -> list[PromptSet]:
    return [PromptSet(number=v.number, image=build_image_prompt(card, v),
                      animation=build_animation_prompt(card, v),
                      references=reference_inputs(card, v))
            for v in visuals]
