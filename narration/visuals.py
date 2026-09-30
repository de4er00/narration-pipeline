"""Stage 2: one visual decision per frame, made after the cut."""
from __future__ import annotations

from .cards import VideoCard
from .frame_prompts import SHOTS, FrameVisual
from .llm import ChatClient, LLMError, Reply
from .pipeline import Script
from .timing import Segmentation

_FIELDS = ("number", "scene_concept", "character_present", "shot",
           "character_action", "object_placement", "visible_text",
           "main_motion", "secondary_motion", "camera")

VISUAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["frames"],
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_FIELDS),
                "properties": {
                    "number": {"type": "integer"},
                    "scene_concept": {"type": "string"},
                    "character_present": {"type": "boolean"},
                    "shot": {"type": "string", "enum": sorted(SHOTS)},
                    "character_action": {"type": "string"},
                    "object_placement": {"type": "string"},
                    "visible_text": {"type": "string"},
                    "main_motion": {"type": "string"},
                    "secondary_motion": {"type": "string"},
                    "camera": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM = (
    "You are an art director turning an approved script into a shot list for "
    "a 2D illustrated video. Respond with JSON only."
)


def visual_prompt(video: VideoCard, seg: Segmentation, owner: list[int],
                  script: Script) -> str:
    """The channel's lists of typical objects and motions are not included.

    Given such a list, the model picked frames from the list instead of from
    the line: the same campfire across different videos, "distance map"
    diagrams for a line about fear. The character description is included,
    because without it the model wrote hands for a character that has pincers.
    """
    ch = video.channel
    lo, hi = ch.character_frames
    n = len(seg.frames)
    titles = [t for t, _ in script.sections]
    numbered = "\n".join(f"{f.number}. [{titles[sec]}] {f.text}"
                         for f, sec in zip(seg.frames, owner))
    shots = "\n".join(f"- {k}: {v}" for k, v in sorted(SHOTS.items()))
    return f"""Design one distinct visual for each of the {n}
frames below. Each frame shows one still illustration while its line is spoken.

THE SCRIPT IS SPOKEN, AND THE PICTURES HAVE TO FEEL LIKE IT
The narration talks to the viewer, jokes, and puts the viewer inside a
situation. The frames carry the same feeling. When a line is funny, its frame
is funny about the same thing. When a line puts the viewer somewhere, its
frame shows that place at that moment, from inside, not a diagram about it. A
picture that is merely correct and calm is the weakest frame this video can
have.

THE RECURRING CHARACTER
{ch.character_name}: {ch.character_bible}

Put the character in {lo}-{hi} of the {n} frames — no more, no
fewer — and SPREAD those frames across the whole video. Never more than four
character frames in a row. Frames without the character are needed for
variety, and they show a real place or thing at that moment. A diagram is the
last resort, for a line that is literally about a number or a relationship,
and even then it is built out of things that exist in the scene rather than
out of abstract marks, arrows and panels.

When character_present is true, character_action must say where the
character is in the frame, what it is doing, and how it reacts to what the
line says. The reaction is drawn at comic-strip size: at the size of a phone
screen a small expression reads as no expression. Refer only to the parts
this character actually has, as described above, and never describe how the
character looks: its appearance is fixed by the character sheet the image
model receives, and a description in words competes with that sheet instead
of helping it. Say what the character does, not what it is.
When character_present is false, leave character_action empty. People,
animals and any other figures the line needs may be in scene_concept; only
the recurring character stays out of these frames.

SHOT SIZES — vary them, never use one size for the whole video
{shots}

RULES THAT DECIDE WHETHER THIS IS ANY GOOD
- Every scene_concept visualizes the exact meaning of ITS line. Never fall
  back on generic evidence, consequence or comparison imagery.
- scene_concept names only what can be SEEN in the picture: the place, who
  and what is in it, and what is happening. It never says what the picture
  means, suggests, stands for, or should make the viewer feel. An image model
  cannot draw an explanation: it turns the words of one into extra objects or
  into lettering on the frame. A joke or a feeling has to be visible in the
  situation itself — describe the situation, not the joke.
- Adjacent frames must differ in at least two of: subject, action,
  composition, scale, location, camera distance, metaphor. A changed pose,
  crop, arrow or zoom alone is NOT a new concept.
- Across the whole video, do not reuse one object as the dominant subject of
  more than three frames. Twenty objects spread over eighty frames reads as
  the same picture four times over.
- object_placement states positive positions for at most three objects,
  each placed by where it sits in the frame or what is holding it.
- visible_text is empty for almost every frame. Fill it only where one short
  English number or word is genuinely necessary to understand the image, and
  in no more than {max(4, n // 8)} frames of the whole video.
  Never put the spoken line on screen.
- main_motion describes the ONE thing that MOVES inside a picture that
  ALREADY EXISTS. The animation pass hands the finished frame to a video
  model together with this field, so anything named here that the frame does
  not contain is invented on screen instead of moved.
  * Name only movement of things the frame itself shows. Never ask for
    something to appear, be built, assemble, construct, rearrange, reconnect
    or dissolve: that contradicts holding the composition, and a video model
    resolves the contradiction by redrawing the picture.
  * Never ask for glow, pulse, shimmer, sparkle, flash, highlight or
    illumination. The channel styles are flat and forbid added light; a video
    model reads these words as permission to add it anyway.
  * ONE action per frame and nothing else. Two movements at once, a moving
    camera or a start-and-stop inside the shot make the video model rebuild
    the drawing frame by frame, and the finished clip shakes. Therefore:
    secondary_motion is always empty, and camera is always "locked camera".
  * main_motion must run at one steady speed along one simple path for the
    whole shot: no starting, stopping, bouncing back or repeating.
  * When the character is in the frame, main_motion is the character's own
    movement: the action or the reaction from character_action, carried
    through. Character frames are the ones picked for animation, and a
    character frame where only the background moves wastes the shot.
  * main_motion always names one real movement of something already in the
    frame, however small. A nearly static frame is still a frame where one
    thing moves a little: stillness is a choice, but "none" is not a motion.
- Write no non-English text, no timestamps, no file names and no project
  codes anywhere.

THE FRAMES
{numbered}

OUTPUT
JSON with `frames`: one entry per frame, numbers 1..{n} in order."""


def design_visuals(video: VideoCard, seg: Segmentation, owner: list[int],
                   script: Script, client: ChatClient,
                   ) -> tuple[list[FrameVisual], Reply]:
    reply = client.ask_json(SYSTEM, visual_prompt(video, seg, owner, script),
                            VISUAL_SCHEMA, name="visuals")
    frames = reply.data.get("frames") or []
    if len(frames) != len(seg.frames):
        raise LLMError(f"visuals: got {len(frames)} frames instead of "
                       f"{len(seg.frames)}; a storyboard with holes is useless")
    return [FrameVisual(**{k: f[k] for k in _FIELDS}) for f in frames], reply
