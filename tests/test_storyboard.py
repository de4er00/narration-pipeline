from __future__ import annotations

import json

import pytest

from conftest import CLEAN_CHAPTERS, CLEAN_CTA
from fakes import make_research
from narration.checks import Problem
from narration.frame_prompts import FrameVisual
from narration.pipeline import Script, cut
from narration.storyboard import (StoryFrame, assemble, cta_already_spoken,
                                  normalise_cta, to_markdown, write_all)


def _script(cta: str = CLEAN_CTA, chapters=CLEAN_CHAPTERS) -> Script:
    return Script(main_idea="Falling never ends.", key_facts=["400 mph winds"],
                  chapters=chapters, final_cta=cta, usd=0.05,
                  problems=[Problem("grounding", "note", advisory=True)])


def _visuals(n: int) -> list[FrameVisual]:
    return [FrameVisual(number=i, scene_concept=f"a distinct scene number {i}",
                        character_present=i % 4 == 0,
                        character_action="Kip tilts its lens" if i % 4 == 0 else "",
                        main_motion="the clouds roll left")
            for i in range(1, n + 1)]


@pytest.mark.parametrize("raw,expected", [
    ("subscribe for more", "subscribe for more."),
    ("what would you do?", "what would you do?"),
    ("", ""),
    ("watch “How the Moon Formed.”", "watch “How the Moon Formed”."),
    ("watch “Why Do We Dream?”", "watch “Why Do We Dream?”."),
    ("watch “The Gym”", "watch “The Gym”."),
    ("watch “How the Moon Formed.”.", "watch “How the Moon Formed”."),
    ("watch “Why Do We Dream?”.", "watch “Why Do We Dream?”."),
])
def test_the_call_to_action_always_ends_with_one_mark(raw, expected):
    assert normalise_cta(raw) == expected


def test_a_call_already_spoken_in_the_chapter_is_not_repeated():
    cta = ("Which hidden part of early human life should Tuk look into next? Tell us "
           "below and subscribe for the next prehistoric puzzle.")
    doubled = [StoryFrame(1, 0, 1, "Which part of Tuk's hidden life should we look into next?", "c"),
               StoryFrame(2, 1, 2, "Tell us below, and subscribe for the next prehistoric puzzle.", "c"),
               StoryFrame(3, 2, 3, "Then watch our winter survival video.", "c")]
    assert cta_already_spoken(cta, doubled)
    clean = [StoryFrame(1, 0, 1, "The fire is out, and the night is long.", "c"),
             StoryFrame(2, 1, 2, "That is what your ancestors solved for you.", "c")]
    assert not cta_already_spoken(cta, clean)


def test_a_short_call_is_never_judged_as_spoken():
    assert not cta_already_spoken("Why?", [StoryFrame(1, 0, 1, "Why?", "c")])


def test_the_call_to_action_is_the_last_frame_and_the_last_words(video):
    script = _script()
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner)
    last = board.frames[-1]
    assert len(board.frames) == video.channel.frames + 1
    assert last.narration == CLEAN_CTA and board.narration.endswith(CLEAN_CTA)
    # It has no chapter of its own.
    assert last.chapter == board.frames[-2].chapter
    assert last.start == board.frames[-2].end and last.end > last.start


def test_a_call_already_in_the_script_is_not_added_twice(video):
    chapters = [dict(c) for c in CLEAN_CHAPTERS]
    chapters[-1] = {**chapters[-1], "narration": chapters[-1]["narration"]
                    + " So which place out there should Kip visit next?"}
    script = _script(chapters=chapters)
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner)
    assert len(board.frames) == video.channel.frames


def test_narration_survives_assembly_word_for_word(video):
    script = _script()
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner)
    assert board.narration.split() == (script.text + " " + CLEAN_CTA).split()


def test_with_visuals_every_frame_gets_prompts_and_the_closing_frame_the_character(video):
    script = _script()
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner, visuals=_visuals(len(seg.frames)))
    assert all(f.image_prompt and f.animation_prompt and f.references for f in board.frames)
    last = board.frames[-1]
    assert last.visual["character_present"] and last.references[0].endswith("character.png")
    # The closing frame assumes no hands and no eyes.
    assert "hand" not in last.visual["character_action"]
    assert "eyes" not in last.visual["character_action"]


def test_problems_and_facts_travel_with_the_storyboard(video):
    script = _script()
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner, research=make_research(),
                     extra_problems=[Problem("visual", "same scene")])
    assert board.facts[1]["status"] == "disputed"
    assert board.problems == ["~[grounding] note", "[visual] same scene"]
    assert board.usd == pytest.approx(0.05)


def test_json_and_markdown_are_written(video, tmp_path):
    script = _script()
    seg, owner = cut(script, video.channel)
    board = assemble(video, script, seg, owner, visuals=_visuals(len(seg.frames)),
                     research=make_research())
    paths = write_all(board, tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["video_id"] == "jupiter-fall"
    assert len(data["frames"]) == len(board.frames)
    assert data["narration"] == board.narration
    assert data["frames"][0]["visual"]["scene_concept"] == "a distinct scene number 1"
    md = paths["markdown"].read_text(encoding="utf-8")
    assert md.startswith("# What would happen if you fell into Jupiter?")
    assert "### Falling Without a Floor" in md
    assert "**001** `0:00-" in md
    assert "## Frame prompts" in md and "[disputed]" in md


def test_markdown_without_visuals_has_no_prompt_section(video):
    script = _script()
    seg, owner = cut(script, video.channel)
    md = to_markdown(assemble(video, script, seg, owner))
    assert "## Frame prompts" not in md
    assert md.count("**0") == len(seg.frames) + 1
