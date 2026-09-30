from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CLEAN_CHAPTERS, CLEAN_CTA, MINI
from fakes import BASELINE_FRAMES, FACTS, FakeLLM, dirty_chapters, research_text, script_reply
from narration import evaluate as ev
from narration.checks import sentences_of
from narration.llm import DEFAULT_MODEL, LLMError, estimate_usd

REPO = Path(__file__).resolve().parent.parent


def _topics(tmp_path: Path, n: int = 2) -> list[ev.Topic]:
    lines = ["topics:"]
    for i in range(n):
        lines += [f"  - id: topic-{i}", f"    channel: {MINI.as_posix()}",
                  f"    title: Falling into Jupiter, take {i}"]
    path = tmp_path / "topics.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return ev.load_topics(path)


def _llm(**overrides) -> FakeLLM:
    frames = [{"number": i, "narration": t, "visual": "clouds"}
              for i, t in enumerate(BASELINE_FRAMES, 1)]
    queues = dict(research=[research_text()],
                  script=[script_reply(dirty_chapters()), script_reply(CLEAN_CHAPTERS)],
                  baseline=[{"frames": frames}])
    return FakeLLM(**{**queues, **overrides})


# --- metrics -----------------------------------------------------------------

def test_metrics_of_the_slot_filled_fixture(mini):
    m = ev.measure(BASELINE_FRAMES, mini, FACTS, BASELINE_FRAMES[-1])
    assert m.frames == 10
    assert m.duplicate_frames == 3
    assert m.repeated_phrases > 0
    assert m.numbers == 1 and m.unsourced_numbers == 1
    assert m.overstated_claims == 1
    assert m.problems >= 5


def test_metrics_of_the_clean_fixture(mini):
    frames = sentences_of(CLEAN_CHAPTERS) + [CLEAN_CTA]
    m = ev.measure(frames, mini, FACTS, CLEAN_CTA)
    assert (m.duplicate_frames, m.repeated_phrases, m.paraphrased_neighbours) == (0, 0, 0)
    assert (m.unsourced_numbers, m.overstated_claims, m.problems) == (0, 0, 0)
    assert abs(m.runtime_error) < 0.08


def test_broken_sentences_are_read_from_the_text_when_not_given(mini):
    m = ev.measure(["You fall and fall,", "and nothing stops you."], mini, [], "q?")
    assert m.broken_sentences == 1


def test_drift_grows_when_slots_are_overfilled(mini):
    line = "The wind keeps pushing you sideways while the clouds roll past below."
    over = ev.measure([f"{line} {line}"] * 10, mini, [], "q?")
    fit = ev.measure([line] * 10, mini, [], "q?")
    assert over.worst_drift_s > fit.worst_drift_s
    assert over.runtime_error > fit.runtime_error


# --- the whole run, offline --------------------------------------------------

def test_run_eval_end_to_end_with_a_fake_model(tmp_path):
    topics = _topics(tmp_path)
    llm = _llm()
    results, run_dir = ev.run_eval(topics, llm, tmp_path, stamp="test")

    assert len(results) == 2 and all("error" not in r for r in results)
    first = results[0]
    assert first["pipeline"]["script"]["repairs"] == 1
    assert results[1]["pipeline"]["script"]["repairs"] == 0
    b, p = first["baseline"]["metrics"], first["pipeline"]["metrics"]
    assert b["duplicate_frames"] == 3 and p["duplicate_frames"] == 0
    assert b["unsourced_numbers"] == 1 and p["unsourced_numbers"] == 0
    assert p["frames"] == 11 and p["broken_sentences"] == 0
    assert p["usd"] == pytest.approx(0.006 + 2 * 0.0062)
    assert b["usd"] == pytest.approx(0.0062)
    assert first["facts"]["facts"][0]["status"] == "confirmed"
    saved = json.loads((run_dir / "topic-0.json").read_text(encoding="utf-8"))
    assert saved["baseline"]["frames"] == BASELINE_FRAMES
    assert llm.count("research") == 2 and llm.count("baseline") == 2


def test_a_failed_topic_is_recorded_and_the_run_goes_on(tmp_path):
    topics = _topics(tmp_path)
    calls = {"n": 0}

    def flaky(user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("search refused")
        return research_text()

    results, run_dir = ev.run_eval(topics, _llm(research=[flaky]), tmp_path, stamp="x")
    assert results[0]["error"] == "search refused"
    assert "error" not in results[1]
    assert (run_dir / "topic-0.json").exists()


def test_report_is_ready_for_the_readme(tmp_path):
    results, run_dir = ev.run_eval(_topics(tmp_path), _llm(), tmp_path, stamp="t")
    text = ev.report(results, model=DEFAULT_MODEL, research_model="perplexity/sonar",
                     run_dir="eval/runs/t", date="2026-10-01")
    assert text.startswith("Eval run 2026-10-01: 2 topics (mini). 1:00 videos cut "
                           "into 10 frames.")
    table = text.split("\n\n")[1].splitlines()
    assert table[0].startswith("| Per script, mean over 2 topics |")
    assert table[1] == "|---|---:|---:|"
    rows = {line.split(" | ")[0].strip("| "): line.split(" | ")[1:] for line in table[2:]}
    assert rows["Verbatim-duplicate frames"] == ["3.0", "0.0 |"]
    assert rows["Numbers not found in the collected facts"] == ["1.0", "0.0 |"]
    assert "| Falling into Jupiter, take 0 | 3 / 0 |" in text
    assert "`eval/runs/t/`" in text


def test_report_marks_failed_topics(tmp_path):
    results = [{"topic": {"id": "a", "channel": "space", "title": "A"}, "error": "boom"}]
    text = ev.report(results, model="m", research_model="r", run_dir="d", date="d")
    assert "| A | failed: boom |" in text
    assert "| Frames | - | - |" in text


def test_summary_averages_absolute_runtime_error():
    def result(err: float) -> dict:
        m = {"runtime_error": err, "usd": 0.0}
        return {"topic": {"channel": "c"}, "baseline": {"metrics": m},
                "pipeline": {"metrics": m}}
    agg = ev.summarise([result(0.1), result(-0.3)])
    assert agg["baseline"]["runtime_error"] == pytest.approx(0.2)


def test_estimate_has_an_expected_value_and_an_upper_bound():
    expected, upper = ev.estimate(9, DEFAULT_MODEL, max_repairs=2)
    assert 0 < expected < upper
    assert upper == pytest.approx(estimate_usd(
        DEFAULT_MODEL, {"script": 9, "repair": 18, "baseline": 9}, research_queries=9))
    assert ev.estimate(9, "unknown/model") == (None, None)


def test_update_readme_inserts_after_the_marker_and_replaces_on_rerun(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\nnumbers\n\n<!-- EVAL RESULTS -->\n\n## Next\n", encoding="utf-8")
    ev.update_readme(readme, "| a | b |\n")
    ev.update_readme(readme, "| c | d |\n")
    text = readme.read_text(encoding="utf-8")
    assert text.count("<!-- EVAL RESULTS -->") == 1
    assert "| a | b |" not in text and "| c | d |" in text
    assert text.index("| c | d |") < text.index("## Next")


def test_update_readme_needs_the_marker(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# T\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ev.update_readme(readme, "x")


def test_bundled_topics_are_valid():
    topics = ev.load_topics(REPO / "eval" / "topics.yaml")
    assert 8 <= len(topics) <= 10
    assert {t.channel for t in topics} == {"history", "space", "science"}
    assert len({t.id for t in topics}) == len(topics)
