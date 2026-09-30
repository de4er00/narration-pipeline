"""Contracts of the stage 1 prompts."""
from __future__ import annotations

import re

import pytest

from fakes import make_research
from narration.cards import ChannelCard, VideoCard, available
from narration.checks import BANNED_SCAFFOLDS, CITATION_MARKERS, Problem
from narration.script import (next_video_block, repair_prompt, resolve_next_video,
                              script_prompt, short_sentences_needed)


@pytest.fixture(params=available())
def card(request) -> ChannelCard:
    return ChannelCard.load(request.param)


def test_prompt_carries_the_budget_of_this_voice(card):
    p = script_prompt(VideoCard(video_id="v", title="T", channel=card))
    assert f"approximately {card.words} words" in p
    assert card.voice.label in p
    assert f"At least {short_sentences_needed(card.sentences)}" in p


def test_the_prompt_has_no_contradicting_word_count(card):
    text = script_prompt(VideoCard(video_id="v", title="T", channel=card))
    assert "Not 800" not in text and "not 900" not in text
    # "what they inherited" in an ending rule turned into "You inherit..." in
    # six scripts out of eight.
    assert "inherit" not in text


def test_prompt_carries_facts_with_their_status(video):
    p = script_prompt(video, make_research())
    assert "[disputed] Jupiter may have a diluted" in p
    assert "name the disagreement" in p


def test_prompt_without_facts_says_so(video):
    assert "- none supplied" in script_prompt(video)


def test_prompt_carries_no_imitable_example_phrases(card):
    # The model copies phrases from its instructions verbatim. Only phrases the
    # prompt forbids may be quoted, and those exactly as the checks spell them.
    video = VideoCard(video_id="v", title="T", channel=card, key_facts=["a"])
    allowed = {s.lower() for s in BANNED_SCAFFOLDS + CITATION_MARKERS}
    bad = []
    for line in script_prompt(video).splitlines():
        for quoted in re.findall(r'"([^"]{6,})"', line):
            if len(quoted.split()) >= 4 and quoted.strip().lower().strip(".,") not in allowed:
                bad.append(quoted)
    assert not bad


def test_every_banned_scaffold_is_named_in_the_prompt(video):
    p = script_prompt(video)
    assert all(f'"{s}"' in p for s in BANNED_SCAFFOLDS)


def test_repair_sees_the_call_to_action_and_the_rhythm(video):
    current = "Which place should Kip visit next?"
    p = repair_prompt(video, [{"title": "One", "narration": "You fall."}],
                      [Problem("budget", "short")], final_cta=current)
    assert current in p and video.channel.cta in p
    assert "six words or fewer" in p


def test_repair_sees_the_material_and_the_sibling_list(video):
    p = repair_prompt(video, [{"title": "A", "narration": "You fall."}],
                      [Problem("source", "x")], research=make_research(),
                      siblings=["The first title"])
    assert "about 400 miles per hour" in p
    assert "The first title" in p


def test_repair_lists_every_problem_with_its_position(video):
    p = repair_prompt(video, [{"title": "A", "narration": "You fall."}],
                      [Problem("duplicate", 'line repeated verbatim: "x"', at=[2, 9])])
    assert '[duplicate] line repeated verbatim: "x" (at 2, 9)' in p


def test_repair_requirements_do_not_contradict_each_other(video):
    p = repair_prompt(video, [{"title": "A", "narration": "You fall."}],
                      [Problem("budget", "short")])
    assert "two questions" not in p and "three questions" in p


def test_repair_prompt_does_not_demand_a_minimum_number_count(video):
    p = repair_prompt(video, [{"title": "A", "narration": "One."}],
                      [Problem("paraphrase", "x")])
    assert "not zero" not in p and "There is no minimum" in p


def test_repair_prompt_forbids_shortening_the_script(video):
    # Two repairs for "paraphrase" once cut a script to 663 words against a
    # budget of 786: they deleted sentences instead of replacing them.
    p = repair_prompt(video, [{"title": "A", "narration": "One."}],
                      [Problem("paraphrase", "x")])
    assert "floor as much as a ceiling" in p and "REPLACING" in p


def test_repair_shows_the_current_script_in_order(video):
    chapters = [{"title": "A", "narration": "First."}, {"title": "B", "narration": "Second."}]
    p = repair_prompt(video, chapters, [])
    assert p.endswith("CURRENT SCRIPT:\n### A\nFirst.\n\n### B\nSecond.")


# --- the hand-off to another video -------------------------------------------

def test_no_siblings_no_block():
    assert next_video_block([]) == ("", "")


def test_the_next_video_block_asks_for_a_number_and_gives_no_wording():
    block, field = next_video_block(["Title one", "Title two"])
    assert "1. Title one" in block and "NUMBER" in block and "next_video" in field
    assert "naturally" not in block


@pytest.mark.parametrize("raw,expected", [
    (2, "Title two"), ("1", "Title one"), (0, ""), (3, ""), (-1, ""),
    (None, ""), ("Title one", ""),
])
def test_the_number_resolves_to_a_real_title_or_nothing(raw, expected):
    assert resolve_next_video(raw, ["Title one", "Title two"]) == expected
