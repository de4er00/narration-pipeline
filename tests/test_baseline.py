from __future__ import annotations

import pytest

from fakes import BASELINE_FRAMES, FakeLLM
from narration.baseline import baseline_prompt, write_baseline
from narration.checks import BANNED_SCAFFOLDS
from narration.llm import LLMError


def test_baseline_asks_for_exactly_the_frame_count_in_slots(video):
    p = baseline_prompt(video)
    ch = video.channel
    assert f"exactly {ch.frames} frames" in p
    assert f"about {ch.words} words" in p
    assert "6-second slot" in p


def test_baseline_gets_the_same_rules_in_words_but_no_facts(video):
    p = baseline_prompt(video)
    assert all(f'"{s}"' in p for s in BANNED_SCAFFOLDS)
    assert video.channel.cta in p and video.channel.role in p
    assert "MATERIAL" not in p and "[confirmed]" not in p


def test_baseline_returns_frames_in_number_order(video):
    frames = [{"number": i, "narration": f"line {i}", "visual": f"v{i}"} for i in (2, 1, 3)]
    base = write_baseline(video, FakeLLM(baseline=[{"frames": frames}]))
    assert base.frames == ["line 1", "line 2", "line 3"]
    assert base.visuals == ["v1", "v2", "v3"]
    assert base.text == "line 1 line 2 line 3"
    assert base.usd > 0 and base.tokens_out > 0


def test_baseline_keeps_what_the_model_returned_even_if_it_repeats(video):
    frames = [{"number": i, "narration": t, "visual": ""}
              for i, t in enumerate(BASELINE_FRAMES, 1)]
    base = write_baseline(video, FakeLLM(baseline=[{"frames": frames}]))
    assert base.frames == BASELINE_FRAMES


def test_an_empty_baseline_is_an_error(video):
    with pytest.raises(LLMError):
        write_baseline(video, FakeLLM(baseline=[{"frames": []}]))
