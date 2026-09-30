from __future__ import annotations

import pytest

from conftest import CLEAN_CHAPTERS
from fakes import FakeLLM, dirty_chapters, make_research, script_reply
from narration import pipeline
from narration.checks import Problem
from narration.llm import LLMError
from narration.pipeline import Script, cut, script_problems, write_script


def _counting(counts: dict[str, int]):
    def fake(chapters, card, research, **kw):
        return [Problem("x", str(i)) for i in range(counts[chapters[0]["narration"]])]
    return fake


def _replies(*names: str) -> list[dict]:
    return [script_reply([{"title": "t", "narration": n}]) for n in names]


def test_a_clean_draft_needs_no_repair(video):
    llm = FakeLLM(script=[script_reply(CLEAN_CHAPTERS)])
    script = write_script(video, llm, research=make_research())
    assert script.repairs == 0 and script.problems == []
    assert llm.count("script") == 1
    assert script.history == [0]


def test_a_dirty_draft_is_repaired_to_clean(video):
    llm = FakeLLM(script=[script_reply(dirty_chapters()), script_reply(CLEAN_CHAPTERS)])
    script = write_script(video, llm, research=make_research())
    assert script.repairs == 1 and script.problems == []
    assert script.chapters == CLEAN_CHAPTERS
    repair = llm.calls[1][1]
    assert "WHAT IS WRONG NOW" in repair
    assert "[scaffold]" in repair and "[duplicate]" in repair
    assert script.history[0] > 0 and script.history[-1] == 0


def test_cost_and_tokens_add_up_over_every_call(video):
    llm = FakeLLM(script=[script_reply(dirty_chapters()), script_reply(CLEAN_CHAPTERS)])
    script = write_script(video, llm, research=make_research())
    assert script.usd == pytest.approx(2 * 0.0062)
    assert script.tokens_out == 2 * 5000


def test_the_best_version_ships_not_the_last(video, monkeypatch):
    monkeypatch.setattr(pipeline, "script_problems",
                        _counting({"draft": 1, "repair1": 3, "repair2": 2}))
    llm = FakeLLM(script=_replies("draft", "repair1", "repair2"))
    script = write_script(video, llm, max_repairs=2)
    assert script.chapters[0]["narration"] == "draft"
    assert script.repairs == 2 and len(script.problems) == 1
    assert script.history == [1, 3, 2]


def test_a_repair_that_wins_is_kept(video, monkeypatch):
    monkeypatch.setattr(pipeline, "script_problems", _counting({"draft": 3, "repair1": 0}))
    script = write_script(video, FakeLLM(script=_replies("draft", "repair1")))
    assert script.chapters[0]["narration"] == "repair1" and script.repairs == 1


def test_a_tie_keeps_the_earlier_version(video, monkeypatch):
    monkeypatch.setattr(pipeline, "script_problems", _counting({"draft": 2, "repair1": 2}))
    script = write_script(video, FakeLLM(script=_replies("draft", "repair1")), max_repairs=1)
    assert script.chapters[0]["narration"] == "draft"


def test_the_second_repair_starts_from_the_best_version(video, monkeypatch):
    monkeypatch.setattr(pipeline, "script_problems",
                        _counting({"draft": 1, "repair1": 3, "repair2": 1}))
    llm = FakeLLM(script=_replies("draft", "repair1", "repair2"))
    write_script(video, llm, max_repairs=2)
    assert "CURRENT SCRIPT:\n### t\ndraft" in llm.calls[2][1]


def test_advisory_notes_never_buy_a_repair(video):
    # No research and no key facts: a grounding note, but no repair call.
    llm = FakeLLM(script=[script_reply(CLEAN_CHAPTERS)])
    script = write_script(video, llm)
    assert llm.count("script") == 1
    assert [p.kind for p in script.problems] == ["grounding"]


def test_repairs_stop_at_the_limit(video, monkeypatch):
    monkeypatch.setattr(pipeline, "script_problems", _counting({"draft": 2}))
    llm = FakeLLM(script=_replies("draft"))
    script = write_script(video, llm, max_repairs=2)
    assert llm.count("script") == 3 and script.repairs == 2


def test_zero_repairs_means_one_call(video):
    llm = FakeLLM(script=[script_reply(dirty_chapters())])
    script = write_script(video, llm, research=make_research(), max_repairs=0)
    assert llm.count("script") == 1 and script.problems


def test_a_draft_without_chapters_is_an_error(video):
    with pytest.raises(LLMError, match="without chapters"):
        write_script(video, FakeLLM(script=[script_reply([])]))


def test_a_repair_without_chapters_keeps_the_previous_version(video):
    llm = FakeLLM(script=[script_reply(dirty_chapters()), script_reply([])])
    script = write_script(video, llm, research=make_research())
    assert script.chapters == dirty_chapters()
    assert llm.count("script") == 2


def test_the_next_video_number_becomes_a_real_title(video):
    llm = FakeLLM(script=[script_reply(CLEAN_CHAPTERS, next_video=2)])
    script = write_script(video, llm, research=make_research(),
                          siblings=["Why Saturn has rings", "How deep is Jupiter"])
    assert script.next_video == "How deep is Jupiter"
    assert "1. Why Saturn has rings" in llm.calls[0][1]


def test_an_invented_number_is_sent_back(mini):
    chapters = [dict(c) for c in CLEAN_CHAPTERS]
    chapters[2] = {**chapters[2], "narration": chapters[2]["narration"].replace(
        "about an hour", "about 58 minutes")}
    kinds = [p.kind for p in script_problems(chapters, mini, make_research())]
    assert "source" in kinds


def test_a_disputed_fact_stated_as_fact_is_sent_back(mini):
    chapters = [dict(c) for c in CLEAN_CHAPTERS]
    chapters[2] = {**chapters[2], "narration": chapters[2]["narration"].replace(
        "The core may be fuzzy and spread out, or it may be a hard lump.",
        "Jupiter has a diluted, fuzzy core rather than a compact one.")}
    kinds = [p.kind for p in script_problems(chapters, mini, make_research())]
    assert "status" in kinds


def test_script_round_trips_through_a_dict():
    s = Script(main_idea="m", key_facts=["f"], chapters=CLEAN_CHAPTERS, final_cta="q?",
               problems=[Problem("rhythm", "x", at=[1], advisory=True)])
    assert Script.from_dict(s.to_dict()) == s


# --- the cut -----------------------------------------------------------------

def test_the_cut_gives_the_channel_frame_count_without_breaking_sentences(mini):
    script = Script(main_idea="", key_facts=[], chapters=CLEAN_CHAPTERS, final_cta="")
    seg, owner = cut(script, mini)
    assert len(seg.frames) == mini.frames
    assert seg.broken_sentences == 0
    assert owner == sorted(owner) and set(owner) == {0, 1, 2}
    assert " ".join(f.text for f in seg.frames).split() == script.text.split()


def test_the_cut_uses_measured_durations_by_default(mini):
    script = Script(main_idea="", key_facts=[], chapters=CLEAN_CHAPTERS, final_cta="")
    seg, _ = cut(script, mini)
    for prev, cur in zip(seg.frames, seg.frames[1:]):
        assert cur.start == pytest.approx(prev.start + prev.seconds)
