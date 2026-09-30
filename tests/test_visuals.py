"""Contracts of the stage 2 prompt and its parsing."""
from __future__ import annotations

import pytest

from conftest import CLEAN_CHAPTERS
from fakes import FakeLLM
from narration.llm import LLMError
from narration.pipeline import Script, cut
from narration.visuals import design_visuals, visual_prompt


@pytest.fixture
def cut_script(mini):
    script = Script(main_idea="", key_facts=[], chapters=CLEAN_CHAPTERS, final_cta="")
    seg, owner = cut(script, mini)
    return script, seg, owner


def _prompt(video, cut_script) -> str:
    script, seg, owner = cut_script
    return " ".join(visual_prompt(video, seg, owner, script).split())


def _frame(n: int, present: bool) -> dict:
    return {"number": n, "scene_concept": f"scene {n}", "character_present": present,
            "shot": "wide", "character_action": "Kip tilts its lens" if present else "",
            "object_placement": "", "visible_text": "", "main_motion": "clouds roll",
            "secondary_motion": "", "camera": "locked camera"}


def test_every_frame_is_listed_with_its_chapter(video, cut_script):
    p = _prompt(video, cut_script)
    assert "1. [Falling Without a Floor] You're falling" in p
    assert f"numbers 1..{len(cut_script[1].frames)}" in p


def test_character_is_described_by_its_own_sheet(video, cut_script):
    p = _prompt(video, cut_script)
    assert video.channel.character_bible in p
    assert "only to the parts this character actually has" in p
    assert "never describe how the character looks" in p


def test_frames_carry_the_feeling_of_the_spoken_script(video, cut_script):
    p = _prompt(video, cut_script)
    assert "THE SCRIPT IS SPOKEN" in p and "how it reacts to what the line says" in p


def test_scene_concept_describes_only_what_can_be_seen(video, cut_script):
    p = _prompt(video, cut_script)
    assert "names only what can be SEEN in the picture" in p
    assert "describe the situation, not the joke" in p


def test_motion_rules_forbid_what_a_flat_frame_cannot_do(video, cut_script):
    p = _prompt(video, cut_script)
    for word in ("construct", "rearrange", "dissolve", "glow", "pulse", "shimmer", "sparkle"):
        assert word in p
    assert 'camera is always "locked camera"' in p
    assert "secondary_motion is always empty" in p
    assert "main_motion always names one real movement" in p
    assert "main_motion is the character's own movement" in p


def test_people_are_allowed_in_frames_without_the_character(video, cut_script):
    p = _prompt(video, cut_script)
    assert "only the recurring character stays out of these frames" in p


def test_diagrams_are_a_last_resort(video, cut_script):
    assert "last resort" in _prompt(video, cut_script)


def test_character_share_comes_from_the_card(video, cut_script):
    lo, hi = video.channel.character_frames
    assert f"Put the character in {lo}-{hi} of the" in _prompt(video, cut_script)


def test_design_visuals_parses_every_frame(video, cut_script):
    script, seg, owner = cut_script
    frames = [_frame(n, n % 3 == 0) for n in range(1, len(seg.frames) + 1)]
    visuals, reply = design_visuals(video, seg, owner, script,
                                    FakeLLM(visuals=[{"frames": frames}]))
    assert [v.number for v in visuals] == list(range(1, len(seg.frames) + 1))
    assert sum(v.character_present for v in visuals) == len(seg.frames) // 3
    assert reply.usd > 0


def test_a_storyboard_with_holes_is_refused(video, cut_script):
    script, seg, owner = cut_script
    with pytest.raises(LLMError, match="instead of"):
        design_visuals(video, seg, owner, script,
                       FakeLLM(visuals=[{"frames": [_frame(1, False)]}]))
