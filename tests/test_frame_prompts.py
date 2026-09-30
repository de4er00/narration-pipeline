from __future__ import annotations

import re

import pytest

from narration.cards import ChannelCard, available
from narration.frame_prompts import (NEGATIVE_ANIMATION, NEGATIVE_IMAGE, SHOTS,
                                     FrameVisual, PromptError, build_all,
                                     build_animation_prompt, build_image_prompt,
                                     reference_inputs, single_action)

IMAGE_ORDER = [
    "FORMAT LOCK:", "REFERENCE INPUTS:", "CHARACTER PRESENCE:", "SCENE CONCEPT:",
    "COMPOSITION:", "CHARACTER POSITION AND ACTION:", "OBJECT PLACEMENT:",
    "VISIBLE TEXT:", "NEGATIVE PROMPT:",
]


@pytest.fixture(params=available())
def card(request) -> ChannelCard:
    return ChannelCard.load(request.param)


def _visual(**kw) -> FrameVisual:
    base = dict(number=1,
                scene_concept="one durable item beside a chain of broken replacements",
                character_present=True,
                character_action="the character stands on the left and points at the chain",
                object_placement="the durable item on the right, the chain in front")
    return FrameVisual(**{**base, **kw})


def _absent(**kw) -> FrameVisual:
    return _visual(character_present=False, character_action="", **kw)


# --- image prompt ------------------------------------------------------------

def test_image_prompt_keeps_the_fixed_block_order(card):
    p = build_image_prompt(card, _visual())
    pos = [p.index(b) for b in IMAGE_ORDER]
    assert pos == sorted(pos)


def test_prompt_never_describes_the_character_in_words(card):
    for v in (_visual(), _absent()):
        p = build_image_prompt(card, v)
        assert card.character_bible not in p
        assert "STYLE IDENTITY LOCK" not in p and "CHARACTER IDENTITY LOCK" not in p


def test_character_frame_gets_both_sheets_character_first(card):
    assert reference_inputs(card, _visual()) == [card.character_sheet, card.style_sheet]


def test_object_frame_gets_the_style_sheet_only(card):
    assert reference_inputs(card, _absent()) == [card.style_sheet]


def test_reference_clause_counts_the_attachments(card):
    def attached(prompt: str) -> list[str]:
        line = next(x for x in prompt.split("\n") if x.startswith("REFERENCE INPUTS:"))
        return re.findall(r"\S+\.png", line)

    alone = build_image_prompt(card, _absent())
    assert "Do not redesign the sheet" in alone and "either sheet" not in alone
    assert len(attached(alone)) == 1
    both = build_image_prompt(card, _visual())
    assert "Do not redesign either sheet" in both and len(attached(both)) == 2


def test_reference_clause_points_by_order_not_by_filename(card):
    both = build_image_prompt(card, _visual())
    assert "FIRST is the character sheet" in both and "SECOND is the style sheet" in both
    alone = build_image_prompt(card, _absent())
    assert "One sheet is attached" in alone and "FIRST" not in alone


def test_reference_clause_does_not_list_human_parts(card):
    line = next(x for x in build_image_prompt(card, _visual()).split("\n")
                if x.startswith("REFERENCE INPUTS:"))
    assert "face, hair" not in line and "clothing" not in line


def test_frame_without_the_character_allows_people_but_no_substitute(card):
    p = build_image_prompt(card, _absent(scene_concept="two hunters climb a rocky ridge"))
    assert "CHARACTER PRESENCE: ABSENT" in p
    assert f"Do not add {card.character_name}" in p and "substitute presenter" in p
    assert "People, animals and other figures" in p


def test_character_present_frame_requires_an_action():
    with pytest.raises(PromptError, match="no action"):
        FrameVisual(number=3, scene_concept="x", character_present=True)


@pytest.mark.parametrize("shot", sorted(SHOTS))
def test_every_shot_size_reaches_the_prompt(shot):
    assert SHOTS[shot] in build_image_prompt(ChannelCard.load("space"), _visual(shot=shot))


def test_unknown_shot_falls_back_to_medium():
    assert FrameVisual(number=1, scene_concept="x", character_present=False,
                       shot="cinematic dolly").shot == "medium"


def test_visible_text_defaults_to_none_and_passes_through():
    card = ChannelCard.load("history")
    assert "VISIBLE TEXT: NONE." in build_image_prompt(card, _visual())
    p = build_image_prompt(card, _visual(visible_text="the number 40 on the jar"))
    assert "VISIBLE TEXT: the number 40 on the jar." in p


def test_default_placement_when_none_is_given():
    p = build_image_prompt(ChannelCard.load("space"), _absent(object_placement=""))
    assert "the dominant object sits in the centre" in p


def test_prompts_name_a_ratio_but_no_pixel_size(card):
    for p in (build_image_prompt(card, _visual()), build_animation_prompt(card, _visual())):
        assert "16:9" in p
        assert "1920" not in p and "1080" not in p


def test_image_prompt_fits_model_limits(card):
    assert len(build_image_prompt(card, _visual())) < 10_000


# --- forbidden content -------------------------------------------------------

def test_no_forbidden_content_in_any_prompt(card):
    for v in (_visual(), _absent()):
        for p in (build_image_prompt(card, v), build_animation_prompt(card, v)):
            assert not re.search(r"[\u0400-\u04FF]", p)
            assert not re.search(r"\b\d{1,2}:\d{2}\b", p)
            assert not re.search(r"_F\d{3}_\d{4}\.png", p)


@pytest.mark.parametrize("field,value,what", [
    ("scene_concept", "a clock reading 0:06", "timecode"),
    ("object_placement", "\u043c\u043e\u043d\u0435\u0442\u0430 in the left hand", "Cyrillic"),
    ("scene_concept", "see video_F012_0106.png", "frame file name"),
    ("scene_concept", 'a sign saying "you are falling and nobody knows"', "quoted narration"),
])
def test_builder_refuses_smuggled_content(field, value, what):
    with pytest.raises(PromptError, match=what):
        build_image_prompt(ChannelCard.load("space"), _visual(**{field: value}))


def test_reference_sheet_names_are_allowed_although_they_end_in_png():
    assert build_image_prompt(ChannelCard.load("space"), _visual()).count(".png") >= 2


def test_mandatory_negative_blocks_are_verbatim(card):
    assert NEGATIVE_IMAGE in build_image_prompt(card, _visual())
    assert NEGATIVE_ANIMATION in build_animation_prompt(card, _visual())


# --- animation prompt --------------------------------------------------------

def test_animation_prompt_carries_the_motion_fields(card):
    p = build_animation_prompt(card, _visual(main_motion="the probe turns its lens"))
    for field in ("Single action: the probe turns its lens.", "Speed:",
                  "Secondary motion: none.", "Camera: a very slow, barely perceptible push-in"):
        assert field in p


def test_animation_never_asks_for_a_second_motion_or_a_moving_camera():
    p = build_animation_prompt(ChannelCard.load("history"), _visual(
        secondary_motion="nearby cords sway", camera="slow orbit left"))
    body = p.split("Negative prompt:")[0]
    assert "nearby cords sway" not in body and "orbit" not in body


def test_camera_line_does_not_name_the_artifacts():
    # A negative list of camera moves put "shake" into the condition: 56% of
    # locked-camera clips shook, 10% of push-in clips.
    p = build_animation_prompt(ChannelCard.load("history"), _visual())
    cam = next(line for line in p.splitlines() if line.startswith("Camera:"))
    assert "push-in" in cam
    for word in ("shake", "parallax", "locked", "tilt", "orbit"):
        assert word not in cam


def test_animation_prompt_does_not_repeat_what_the_frame_shows(card):
    vis = _visual()
    p = build_animation_prompt(card, vis)
    for text in (card.character_bible, vis.scene_concept, vis.object_placement,
                 vis.character_action, card.character_name):
        assert text not in p


def test_animation_prompt_leaves_duration_to_the_request(card):
    body = build_animation_prompt(card, _visual()).split("Negative prompt:")[0]
    assert not re.search(r"\bsecond", body)


def test_animation_mentions_text_only_when_the_frame_has_text():
    card = ChannelCard.load("space")
    without = build_animation_prompt(card, _visual(visible_text=""))
    assert "Visible text:" not in without and "no text changes" not in without
    with_text = build_animation_prompt(card, _visual(visible_text="1 in 200"))
    assert "the frame already contains the text 1 in 200" in with_text
    assert "no text changes" in with_text


def test_animation_negative_bans_the_particles_video_models_add():
    p = build_animation_prompt(ChannelCard.load("space"), _visual())
    for ban in ("no floating dust", "no drifting particles", "no sparkles", "no film grain"):
        assert ban in p


def test_frames_without_the_character_keep_existing_figures():
    p = build_animation_prompt(ChannelCard.load("history"), _absent(
        scene_concept="two hunters climb a rocky ridge", main_motion="the hunters step up"))
    assert "keep every figure the frame already shows" in p
    assert "Props:" not in p


def test_animation_prompt_is_mostly_motion(card):
    p = build_animation_prompt(card, _visual(
        main_motion="the scraper advances a short distance across the hide"))
    body = p.split("Negative prompt:")[0]
    motion = sum(len(line) for line in body.split("\n")
                 if line.startswith(("Single action:", "Speed:", "Secondary motion:", "Camera:")))
    assert motion / len(body) >= 0.2


@pytest.mark.parametrize("raw,expected", [
    ("the lid lifts slowly, then stops", "the lid lifts slowly"),
    ("the lid lifts slowly and pauses at the top", "the lid lifts slowly"),
    ("it stops", "it stops"),
    ("", "hold the composition nearly static"),
])
def test_a_stop_tail_is_cut_when_something_is_left(raw, expected):
    assert single_action(raw) == expected


def test_an_inseparable_stop_softens_the_speed_line():
    p = build_animation_prompt(ChannelCard.load("space"),
                               _visual(main_motion="the probe drifts and slows to a stop"))
    assert "Speed: even and unhurried" in p
    even = build_animation_prompt(ChannelCard.load("space"),
                                  _visual(main_motion="the probe drifts to the right"))
    assert "Speed: constant and uniform" in even


def test_build_all_returns_one_set_per_frame():
    card = ChannelCard.load("science")
    sets = build_all(card, [_visual(number=n) for n in range(1, 11)])
    assert [s.number for s in sets] == list(range(1, 11))
    assert all(s.image and s.animation and s.references for s in sets)
