from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CLEAN_CHAPTERS, MINI
from fakes import BASELINE_FRAMES, FakeLLM, dirty_chapters, research_text, script_reply
from narration import cli


def _text_script(tmp_path: Path, chapters: list[dict]) -> Path:
    path = tmp_path / "script.txt"
    path.write_text("\n\n".join(c["narration"] for c in chapters), encoding="utf-8")
    return path


def _use(monkeypatch, llm: FakeLLM) -> None:
    monkeypatch.setattr(cli, "_client", lambda args: llm)


def test_eval_without_yes_prints_the_estimate_and_spends_nothing(monkeypatch, capsys):
    def no_client(args):
        raise AssertionError("a client was created without --yes")
    monkeypatch.setattr(cli, "_client", no_client)
    topics = Path(__file__).resolve().parent.parent / "eval" / "topics.yaml"
    assert cli.main(["eval", "--topics", str(topics)]) == 1
    out = capsys.readouterr().out
    assert "Estimated cost: about $" in out and "up to $" in out
    assert "Nothing was spent" in out


def test_eval_with_yes_writes_results_and_fills_the_readme(monkeypatch, tmp_path, capsys):
    topics = tmp_path / "topics.yaml"
    topics.write_text(f"topics:\n  - id: t1\n    channel: {MINI.as_posix()}\n"
                      f"    title: Falling into Jupiter\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# X\n\n<!-- EVAL RESULTS -->\n\n## Rest\n", encoding="utf-8")
    frames = [{"number": i, "narration": t, "visual": ""} for i, t in enumerate(BASELINE_FRAMES, 1)]
    _use(monkeypatch, FakeLLM(research=[research_text()],
                              script=[script_reply(CLEAN_CHAPTERS)],
                              baseline=[{"frames": frames}]))
    rc = cli.main(["eval", "--topics", str(topics), "--out", str(tmp_path / "ev"),
                   "--readme", str(readme), "--yes"])
    assert rc == 0
    results = (tmp_path / "ev" / "results.md").read_text(encoding="utf-8")
    assert "| Verbatim-duplicate frames | 3.0 | 0.0 |" in results
    assert "| Verbatim-duplicate frames | 3.0 | 0.0 |" in readme.read_text(encoding="utf-8")
    runs = list((tmp_path / "ev" / "runs").iterdir())
    assert (runs[0] / "t1.json").exists() and (runs[0] / "summary.json").exists()


def test_eval_filters_topics(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda args: None)
    topics = Path(__file__).resolve().parent.parent / "eval" / "topics.yaml"
    assert cli.main(["eval", "--topics", str(topics), "--only", "fire-before-matches"]) == 1
    assert "1 topics (fire-before-matches)" in capsys.readouterr().out
    assert cli.main(["eval", "--topics", str(topics), "--only", "nope"]) == 1
    assert "unknown topic ids: nope" in capsys.readouterr().out


def test_eval_estimates_a_baseline_only_run_at_eighty_frames(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_client", lambda args: None)
    topics = Path(__file__).resolve().parent.parent / "eval" / "topics.yaml"
    assert cli.main(["eval", "--topics", str(topics), "--baseline-only",
                     "--frames", "80"]) == 1
    out = capsys.readouterr().out
    assert "9 topics" in out and "80 frames" in out and "baseline only" in out
    assert "Estimated cost: about $0.53" in out


def _two_topics(tmp_path: Path) -> Path:
    topics = tmp_path / "topics.yaml"
    topics.write_text("topics:\n" + "".join(
        f"  - id: t{i}\n    channel: {MINI.as_posix()}\n    title: Jupiter {i}\n"
        for i in (1, 2)), encoding="utf-8")
    return topics


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "ev" / "runs" / "20260930T164030Z"
    run_dir.mkdir(parents=True)
    return run_dir


def test_merge_reruns_only_the_failed_topics_into_the_same_run(monkeypatch, tmp_path, capsys):
    topics = _two_topics(tmp_path)
    frames = [{"number": i, "narration": t, "visual": ""} for i, t in enumerate(BASELINE_FRAMES, 1)]
    llm = FakeLLM(research=[research_text()], script=[script_reply(CLEAN_CHAPTERS)],
                  baseline=[{"frames": frames}])
    _use(monkeypatch, llm)
    assert cli.main(["eval", "--topics", str(topics), "--out", str(tmp_path / "ev"),
                     "--yes"]) == 0
    run_dir = next((tmp_path / "ev" / "runs").iterdir())
    failed = {"topic": {"id": "t2", "channel": MINI.as_posix(), "title": "Jupiter 2"},
              "error": 'OpenRouter refused: HTTP 401 {"error": "API key expired."}'}
    (run_dir / "t2.json").write_text(json.dumps(failed), encoding="utf-8")
    llm.calls.clear()
    capsys.readouterr()

    assert cli.main(["eval", "--topics", str(topics), "--out", str(tmp_path / "ev"),
                     "--merge", str(run_dir), "--yes"]) == 0
    assert llm.count("research") == 1 and "Jupiter 2" in llm.calls[0][1]
    text = (tmp_path / "ev" / "results.md").read_text(encoding="utf-8")
    assert "2 topics (mini), 2 finished" in text and "not run" not in text
    assert ", 1 rerun " in text.splitlines()[0]

    capsys.readouterr()
    assert cli.main(["eval", "--topics", str(topics), "--merge", str(run_dir)]) == 0
    assert "no topics to run" in capsys.readouterr().out


def test_merge_refuses_a_run_with_other_settings(monkeypatch, tmp_path, capsys):
    from narration import evaluate as ev
    run_dir = _run_dir(tmp_path)
    ev.write_settings(run_dir, ev.Settings(mode=ev.BASELINE_ONLY, frames=80))
    monkeypatch.setattr(cli, "_client", lambda args: None)
    assert cli.main(["eval", "--topics", str(_two_topics(tmp_path)),
                     "--merge", str(run_dir), "--yes"]) == 1
    assert "other settings" in capsys.readouterr().out


def test_rescore_rewrites_the_reports_without_a_model(monkeypatch, tmp_path, capsys):
    frames = [{"number": i, "narration": t, "visual": ""} for i, t in enumerate(BASELINE_FRAMES, 1)]
    _use(monkeypatch, FakeLLM(research=[research_text()], script=[script_reply(CLEAN_CHAPTERS)],
                              baseline=[{"frames": frames}]))
    topics = _two_topics(tmp_path)
    assert cli.main(["eval", "--topics", str(topics), "--out", str(tmp_path / "ev"),
                     "--yes"]) == 0
    run_dir = next((tmp_path / "ev" / "runs").iterdir())
    (tmp_path / "ev" / "results.md").unlink()
    (run_dir / "summary.json").unlink()

    def no_client(args):
        raise AssertionError("rescoring must not create a client")
    monkeypatch.setattr(cli, "_client", no_client)
    readme = tmp_path / "README.md"
    readme.write_text("# X\n\n<!-- EVAL RESULTS -->\n", encoding="utf-8")
    assert cli.main(["eval", "--topics", str(topics), "--out", str(tmp_path / "ev"),
                     "--rescore", str(run_dir), "--readme", str(readme)]) == 0
    assert (tmp_path / "ev" / "results.md").exists() and (run_dir / "summary.json").exists()
    assert "| Frames ending mid-clause |" in readme.read_text(encoding="utf-8")
    assert cli.main(["eval", "--rescore", str(tmp_path / "missing")]) == 1


def test_an_experiment_run_does_not_overwrite_the_main_results(monkeypatch, tmp_path):
    frames = [{"number": i, "narration": t, "visual": ""} for i, t in enumerate(BASELINE_FRAMES, 1)]
    _use(monkeypatch, FakeLLM(baseline=[{"frames": frames}]))
    assert cli.main(["eval", "--topics", str(_two_topics(tmp_path)), "--out",
                     str(tmp_path / "ev"), "--baseline-only", "--frames", "12", "--yes"]) == 0
    assert not (tmp_path / "ev" / "results.md").exists()
    run_dir = next((tmp_path / "ev" / "runs").iterdir())
    assert "12 fixed slots" in (run_dir / "results.md").read_text(encoding="utf-8")


def test_check_passes_a_clean_script_and_fails_a_dirty_one(tmp_path, capsys):
    assert cli.main(["check", str(_text_script(tmp_path, CLEAN_CHAPTERS)),
                     "--channel", str(MINI)]) == 0
    assert "clean" in capsys.readouterr().out
    assert cli.main(["check", str(_text_script(tmp_path, dirty_chapters())),
                     "--channel", str(MINI)]) == 1
    out = capsys.readouterr().out
    assert "[duplicate]" in out and "[scaffold]" in out


def test_check_treats_lines_as_frames_on_request(tmp_path, capsys):
    path = tmp_path / "frames.txt"
    path.write_text("\n".join(BASELINE_FRAMES), encoding="utf-8")
    assert cli.main(["check", str(path), "--channel", str(MINI), "--frames"]) == 1
    assert "(at 2, 4, 6, 9)" in capsys.readouterr().out


def test_cut_prints_the_timeline(tmp_path, capsys):
    assert cli.main(["cut", str(_text_script(tmp_path, CLEAN_CHAPTERS)),
                     "--channel", str(MINI)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("001 ") and "10 frames" in out and "broken sentences 0" in out


def test_cut_reports_a_text_that_is_too_short(tmp_path, capsys):
    path = tmp_path / "short.txt"
    path.write_text("Too short.", encoding="utf-8")
    assert cli.main(["cut", str(path), "--channel", str(MINI)]) == 1
    assert "cannot cut" in capsys.readouterr().out


def test_research_writes_facts(monkeypatch, tmp_path, capsys):
    _use(monkeypatch, FakeLLM(research=[research_text()]))
    assert cli.main(["research", "Falling into Jupiter", "--channel", str(MINI),
                     "--out", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "falling-into-jupiter.facts.json").read_text(encoding="utf-8"))
    assert len(saved["facts"]) == 2


def test_write_runs_every_stage_and_saves_the_script_first(monkeypatch, tmp_path, capsys):
    visuals = [{"number": n, "scene_concept": f"scene {n}", "character_present": n in (3, 7),
                "shot": "wide", "character_action": "Kip tilts its lens" if n in (3, 7) else "",
                "object_placement": "", "visible_text": "", "main_motion": "clouds roll",
                "secondary_motion": "", "camera": "locked camera"} for n in range(1, 11)]
    llm = FakeLLM(research=[research_text()],
                  script=[script_reply(dirty_chapters()), script_reply(CLEAN_CHAPTERS)],
                  visuals=[{"frames": visuals}])
    _use(monkeypatch, llm)
    rc = cli.main(["write", "What if you fell into Jupiter?", "--channel", str(MINI),
                   "--id", "jupiter", "--visuals", "--out", str(tmp_path)])
    assert rc == 0
    out_dir = tmp_path / "jupiter"
    assert len(json.loads((out_dir / "facts.json").read_text(encoding="utf-8"))["facts"]) == 2
    script = json.loads((out_dir / "script.json").read_text(encoding="utf-8"))
    assert script["repairs"] == 1
    board = json.loads((out_dir / "jupiter.json").read_text(encoding="utf-8"))
    assert len(board["frames"]) == 11 and board["frames"][0]["image_prompt"]
    assert (out_dir / "jupiter.md").exists()
    assert "Estimated cost: about $" in capsys.readouterr().out
    assert [n for n, _ in llm.calls] == ["research", "script", "script", "visuals"]


def test_write_can_reuse_saved_facts(monkeypatch, tmp_path):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"topic": "t", "facts": [
        {"claim": "Winds reach about 400 miles per hour.", "status": "confirmed",
         "source": "s"}]}), encoding="utf-8")
    llm = FakeLLM(script=[script_reply(CLEAN_CHAPTERS)])
    _use(monkeypatch, llm)
    assert cli.main(["write", "Jupiter", "--channel", str(MINI), "--facts", str(facts),
                     "--out", str(tmp_path)]) == 0
    assert llm.count("research") == 0


def test_a_missing_key_is_a_clear_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["write", "Jupiter", "--channel", str(MINI), "--out", str(tmp_path)]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_unknown_channel_lists_the_bundled_ones():
    from narration.cards import CardError
    with pytest.raises(CardError, match="history, science, space"):
        cli.main(["cut", "x.txt", "--channel", "nope"])


def test_slugify():
    assert cli.slugify("Why can't you tickle yourself?") == "why-can-t-you-tickle-yourself"
    assert cli.slugify("???") == "video"
