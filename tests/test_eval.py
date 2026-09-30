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
FIRST_RUN = REPO / "eval" / "runs" / "20260930T164030Z"
API_401 = ('OpenRouter refused: HTTP 401 {"error":{"message":"API key expired.",'
           '"code":401}}')


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


def _run(tmp_path: Path, n: int = 2, llm: FakeLLM | None = None,
         settings: ev.Settings | None = None) -> tuple[list[dict], Path]:
    run_dir = tmp_path / "runs" / "20261001T000000Z"
    results = ev.run_eval(_topics(tmp_path, n), llm or _llm(), run_dir,
                          settings or ev.Settings())
    return results, run_dir


# --- metrics -----------------------------------------------------------------

def test_metrics_of_the_slot_filled_fixture(mini):
    m = ev.measure(BASELINE_FRAMES, mini, FACTS, BASELINE_FRAMES[-1])
    assert m.frames == 10
    assert m.duplicate_frames == 3
    assert m.repeated_phrases > 0
    assert m.figures == 1 and m.unsourced_numbers == 1
    assert m.overstated_claims == 1
    assert m.problems >= 5


def test_metrics_of_the_clean_fixture(mini):
    frames = sentences_of(CLEAN_CHAPTERS) + [CLEAN_CTA]
    m = ev.measure(frames, mini, FACTS, CLEAN_CTA)
    assert (m.duplicate_frames, m.repeated_phrases, m.paraphrased_neighbours) == (0, 0, 0)
    assert (m.unsourced_numbers, m.overstated_claims, m.problems) == (0, 0, 0)
    assert abs(m.runtime_error) < 0.08


def test_spelled_out_figures_are_counted_and_traced(mini):
    frames = ["You fall two hundred fifty miles.", "Then about fifty-eight more."]
    facts = [FACTS[0], FACTS[1]] + [type(FACTS[0])("It fell 250 miles.", "confirmed", "s")]
    m = ev.measure(frames, mini, facts, "q?")
    assert m.figures == 2 and m.unsourced_numbers == 1


@pytest.mark.parametrize("text,kind", [
    ("It ends here.", "sentence"),
    ('He said "stop!"', "sentence"),
    ("Lower water exposed more ground,", "clause"),
    ("notice the bargain your eyes are making:", "clause"),
    ("It fell —", "clause"),
    ("The star's scale, density", "mid"),
])
def test_frame_endings(text, kind):
    assert ev.frame_ending(text) == kind


def test_clause_breaks_and_mid_clause_breaks_are_separate(mini):
    m = ev.measure(["You fall and fall,", "and nothing", "stops you."], mini, None, "q?")
    assert (m.clause_breaks, m.mid_clause_breaks) == (1, 1)


def test_without_facts_the_fact_metrics_are_empty(mini):
    m = ev.measure(BASELINE_FRAMES, mini, None, BASELINE_FRAMES[-1])
    assert m.unsourced_numbers is None and m.overstated_claims is None
    assert m.figures == 1


def test_drift_grows_when_slots_are_overfilled(mini):
    line = "The wind keeps pushing you sideways while the clouds roll past below."
    over = ev.measure([f"{line} {line}"] * 10, mini, [], "q?")
    fit = ev.measure([line] * 10, mini, [], "q?")
    assert over.worst_drift_s > fit.worst_drift_s
    assert over.runtime_error > fit.runtime_error


def test_frame_count_changes_keep_the_frame_length():
    card = ev.channel_card("space", 80)
    assert (card.frames, card.target_seconds, card.frame_seconds) == (80, 480, 6.0)
    assert card.words > ev.channel_card("space").words
    assert ev.channel_card("space", 60) == ev.channel_card("space")


# --- running offline ---------------------------------------------------------

def test_run_eval_end_to_end_with_a_fake_model(tmp_path):
    llm = _llm()
    results, run_dir = _run(tmp_path, llm=llm)

    assert len(results) == 2 and all("error" not in r for r in results)
    first = results[0]
    assert first["pipeline"]["script"]["repairs"] == 1
    assert results[1]["pipeline"]["script"]["repairs"] == 0
    b, p = first["baseline"]["metrics"], first["pipeline"]["metrics"]
    assert b["duplicate_frames"] == 3 and p["duplicate_frames"] == 0
    assert b["unsourced_numbers"] == 1 and p["unsourced_numbers"] == 0
    assert p["frames"] == 11 and p["mid_clause_breaks"] == 0
    assert p["usd"] == pytest.approx(0.006 + 2 * 0.0062)
    assert b["usd"] == pytest.approx(0.0062)
    saved = json.loads((run_dir / "topic-0.json").read_text(encoding="utf-8"))
    assert saved["baseline"]["frames"] == BASELINE_FRAMES
    assert ev.read_settings(run_dir).mode == ev.FULL
    assert llm.count("research") == 2 and llm.count("baseline") == 2


def test_a_failed_topic_is_recorded_and_the_run_goes_on(tmp_path):
    calls = {"n": 0}

    def flaky(user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError(API_401)
        return research_text()

    results, run_dir = _run(tmp_path, llm=_llm(research=[flaky]))
    assert results[0]["error"] == API_401
    assert "error" not in results[1]
    assert ev.needs_run(run_dir, "topic-0") and not ev.needs_run(run_dir, "topic-1")
    assert ev.needs_run(run_dir, "never-run")


def test_a_script_too_short_to_cut_is_recorded_as_a_failure(tmp_path):
    short = script_reply([{"title": "t", "narration": "You fall. It's dark."}])
    results, _ = _run(tmp_path, 1, _llm(script=[short]), ev.Settings(max_repairs=0))
    assert ev.failure_note(results[0]["error"]) == "failed: script too short to cut"


def test_baseline_only_runs_one_call_per_topic_at_any_frame_count(tmp_path):
    llm = _llm()
    results, run_dir = _run(tmp_path, llm=llm,
                            settings=ev.Settings(mode=ev.BASELINE_ONLY, frames=20))
    assert [n for n, _ in llm.calls] == ["baseline", "baseline"]
    assert "facts" not in results[0] and "pipeline" not in results[0]
    assert results[0]["setup"] == {"channel": "mini", "target_seconds": 120, "frames": 20}
    assert "20 fixed slots" in llm.calls[0][1] or "exactly 20 frames" in llm.calls[0][1]
    assert results[0]["baseline"]["metrics"]["unsourced_numbers"] is None


def test_merging_replaces_failed_topics_and_records_the_merge(tmp_path):
    def refuse(user: str) -> str:
        raise LLMError(API_401)

    _, run_dir = _run(tmp_path, llm=_llm(research=[refuse]))
    assert all(ev.needs_run(run_dir, f"topic-{i}") for i in range(2))
    retry = [t for t in _topics(tmp_path) if ev.needs_run(run_dir, t.id)][:1]
    ev.run_eval(retry, _llm(), run_dir, ev.Settings())

    results = ev.load_results(run_dir, ["topic-1", "topic-0"])
    assert [r["topic"]["id"] for r in results] == ["topic-1", "topic-0"]
    assert "error" in results[0] and "error" not in results[1]
    settings = ev.read_settings(run_dir)
    assert settings.merges[0]["topics"] == ["topic-0"]
    assert settings.started.startswith("20")


def test_merging_into_a_run_without_run_json_dates_it_by_its_name(tmp_path):
    run_dir = tmp_path / "runs" / "20260930T164030Z"
    run_dir.mkdir(parents=True)
    (run_dir / "topic-0.json").write_text(json.dumps(
        {"topic": {"id": "topic-0", "channel": "x", "title": "T"}, "error": API_401}),
        encoding="utf-8")
    ev.run_eval(_topics(tmp_path, 1), _llm(), run_dir, ev.Settings())
    assert ev.read_settings(run_dir).started == "2026-09-30"


# --- rescoring and reports ---------------------------------------------------

def _legacy(result: dict) -> dict:
    """The shape of the first committed run: cost inside the metrics."""
    old = json.loads(json.dumps(result))
    for v in ("baseline", "pipeline"):
        cost = old[v].pop("cost")
        old[v]["metrics"] = {"broken_sentences": 0, "numbers": 0, **cost}
    return old


def test_rescore_recomputes_metrics_and_keeps_the_cost(tmp_path):
    results, _ = _run(tmp_path, 1)
    fresh = ev.rescore(_legacy(results[0]))
    assert fresh["baseline"]["metrics"]["duplicate_frames"] == 3
    assert fresh["baseline"]["metrics"]["usd"] == pytest.approx(0.0062)
    assert "broken_sentences" not in fresh["pipeline"]["metrics"]
    assert fresh["pipeline"]["metrics"] == results[0]["pipeline"]["metrics"]


def test_rescore_leaves_failures_alone():
    failed = {"topic": {"id": "a", "channel": "space", "title": "A"}, "error": API_401}
    assert ev.rescore(failed) is failed


@pytest.mark.skipif(not FIRST_RUN.is_dir(), reason="first eval run not present")
def test_rescoring_the_first_run_sees_spelled_out_figures():
    results = [ev.rescore(r) for r in ev.load_results(FIRST_RUN)]
    ok = [r for r in results if "error" not in r]
    assert len(ok) == 6 and len(results) == 9
    pipe = ev.summarise(ok)["pipeline"]
    assert pipe["figures"] > 1 and pipe["unsourced_numbers"] == 0
    assert pipe["mid_clause_breaks"] == 0 and pipe["clause_breaks"] > 0


def test_report_is_ready_for_the_readme(tmp_path):
    results, _ = _run(tmp_path)
    text = ev.report(results, ev.Settings(), run_dir="eval/runs/t", date="2026-10-01")
    assert text.startswith("Eval run 2026-10-01: 2 topics (mini), 2 finished. "
                           "1:00 videos, 10 frames.")
    table = text.split("\n\n")[1].splitlines()
    assert table[0] == ("| Per script, mean over 2 topics | Baseline: one request, "
                        "10 fixed slots | Pipeline |")
    assert table[1] == "|---|---:|---:|"
    rows = {line.split(" | ")[0].strip("| "): line.split(" | ")[1:] for line in table[2:]}
    assert rows["Frames, call to action included"] == ["10", "11 |"]
    assert rows["Verbatim-duplicate frames"] == ["3.0", "0.0 |"]
    assert rows["Figures not found in the collected facts"] == ["1.0", "0.0 |"]
    assert "Frames ending mid-clause" in rows
    assert "call to action included" in text.split("\n\n")[2]
    assert "| Falling into Jupiter, take 0 | 3 / 0 |" in text
    assert "`eval/runs/t/`" in text


def test_failed_topics_render_as_a_short_note_never_raw_json(tmp_path):
    results, _ = _run(tmp_path)
    results[1] = {"topic": results[1]["topic"], "error": API_401}
    text = ev.report(results, ev.Settings(), run_dir="d", date="2026-10-01")
    assert "2 topics (mini), 1 finished, 1 not run: API error." in text
    assert "| Falling into Jupiter, take 1 | not run: API error | | | | | | |" in text
    assert "{" not in text and "401" not in text


def test_report_of_a_merged_run_says_when_topics_were_rerun(tmp_path):
    results, _ = _run(tmp_path)
    settings = ev.Settings(started="2026-09-30",
                           merges=[{"at": "2026-10-02T10:00:00Z", "topics": ["a", "b"]}])
    text = ev.report(results, settings, run_dir="d", date="2026-09-30")
    assert text.startswith("Eval run 2026-09-30, 2 rerun 2026-10-02:")


def test_baseline_only_report_has_one_column_and_no_fact_rows(tmp_path):
    settings = ev.Settings(mode=ev.BASELINE_ONLY, frames=20)
    results, _ = _run(tmp_path, settings=settings)
    text = ev.report(results, settings, run_dir="d", date="2026-10-01")
    table = text.split("\n\n")[1].splitlines()
    assert table[0].endswith("| Baseline: one request, 20 fixed slots |")
    assert table[1] == "|---|---:|"
    assert "collected facts" not in text and "stated as fact" not in text
    assert "Baseline only" in text


@pytest.mark.parametrize("error,note", [
    (API_401, "not run: API error"),
    ("openai/x failed after 3 attempts: HTTP 502", "not run: API error"),
    ("the sections split into at most 2 pieces but 10 frames are needed", "failed: script too short to cut"),
    ("something odd", "failed: see the run file"),
])
def test_failure_notes(error, note):
    assert ev.failure_note(error) == note


def test_summary_averages_absolute_runtime_error_and_skips_missing_values():
    def result(err: float) -> dict:
        m = {"runtime_error": err, "usd": 0.0, "unsourced_numbers": None}
        return {"topic": {"channel": "c"}, "baseline": {"metrics": m},
                "pipeline": {"metrics": m}}
    agg = ev.summarise([result(0.1), result(-0.3)])
    assert agg["baseline"]["runtime_error"] == pytest.approx(0.2)
    assert agg["baseline"]["unsourced_numbers"] is None


# --- estimates and README ----------------------------------------------------

def test_estimate_has_an_expected_value_and_a_high_one():
    expected, high = ev.estimate(9, DEFAULT_MODEL, max_repairs=2)
    assert 0 < expected < high
    assert high == pytest.approx(estimate_usd(
        DEFAULT_MODEL, {"writing": 27, "baseline": 9}, research_queries=9, high=True))
    assert ev.estimate(9, "unknown/model") == (None, None)


def test_estimate_matches_the_first_eval_run():
    # Six finished topics cost $0.975 with 5 repairs over 6 scripts; the
    # estimate before that run said $0.78 for nine, about half the real rate.
    expected, high = ev.estimate(6, DEFAULT_MODEL)
    assert expected == pytest.approx(0.975, rel=0.03)
    assert high > 0.975


def test_estimate_scales_with_frames_and_can_skip_the_pipeline():
    base60, _ = ev.estimate(9, DEFAULT_MODEL, baseline_only=True)
    base80, high80 = ev.estimate(9, DEFAULT_MODEL, baseline_only=True, frames=80)
    assert base80 == pytest.approx(base60 * 80 / 60, rel=1e-3)
    assert base80 < high80 < ev.estimate(9, DEFAULT_MODEL)[0]


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
