from __future__ import annotations

import pytest

from fakes import FACTS, FakeLLM, research_text
from narration.llm import LLMError
from narration.research import (Fact, Research, collect, extract_facts, overstated,
                                research_prompt, unsourced_numbers)


def _facts(*claims: str) -> list[Fact]:
    return [Fact(claim=c, status="confirmed", source="s") for c in claims]


# --- parsing a free-text reply -----------------------------------------------

ONE = '{"claim":"a","status":"confirmed","source":"s"}'


@pytest.mark.parametrize("raw,count", [
    ('{"facts":[' + ONE + ']}', 1),
    ("[" + ONE + "," + ONE + "]", 2),
    ('```json\n{"facts":[' + ONE + ']}\n```', 1),
    ('Here it is:\n{"facts":[' + ONE + ']}', 1),
    ('{"items":[' + ONE + ']}', 1),
    (ONE, 1),
])
def test_parser_handles_every_shape_the_search_model_returns(raw, count):
    assert len(extract_facts(raw)) == count


def test_latex_inside_a_string_does_not_destroy_the_whole_reply():
    raw = (r'{"items":[{"claim":"An effective population of about '
           r'\(N_e \sim 200\) individuals.","status":"confirmed","source":"Science"}]}')
    [fact] = extract_facts(raw)
    assert "200" in fact["claim"]


def test_valid_escapes_survive_the_repair():
    raw = (r'{"facts":[{"claim":"They called it \"the wall\" — about 3 km wide.",'
           r'"status":"reported","source":"s"}]}')
    assert extract_facts(raw)[0]["claim"] == 'They called it "the wall" — about 3 km wide.'
    dash = '{"facts":[{"claim":"a wall \\u2014 3 km wide.","status":"reported","source":"s"}]}'
    assert extract_facts(dash)[0]["claim"] == "a wall — 3 km wide."


@pytest.mark.parametrize("bad", ["", "I could not find anything.", '{"facts":[]}', "{not json"])
def test_parser_fails_loudly_instead_of_returning_nothing(bad):
    with pytest.raises(LLMError):
        extract_facts(bad)


# --- facts and statuses ------------------------------------------------------

def test_fact_with_unknown_status_is_downgraded_not_trusted():
    assert Fact("a", "definitely-true", "s").status == "speculation"


def test_status_carries_its_language_rule_into_the_prompt():
    line = Fact("Something happened", "disputed", "Some body").as_prompt_line()
    assert "[disputed]" in line and "never pick a side" in line


def test_research_round_trips_through_json(tmp_path):
    res = Research(topic="t", facts=list(FACTS), sources=["https://x"], usd=0.006)
    back = Research.load(res.save(tmp_path / "f.json"))
    assert back.facts == res.facts and back.sources == res.sources
    assert back.by_status == {"confirmed": 1, "disputed": 1}


def test_empty_research_prompts_as_none():
    assert Research(topic="t").as_prompt_block() == "- none supplied"


def test_research_prompt_asks_for_status_and_plain_strings(mini):
    p = research_prompt("Why is Jupiter striped?", mini, 6)
    assert "Collect 6" in p and "speculation" in p and "No LaTeX" in p
    assert mini.role in p


def test_collect_uses_the_search_model_and_keeps_sources(mini):
    llm = FakeLLM(research=[research_text()])
    res = collect("Falling into Jupiter", mini, llm)
    assert [f.status for f in res.facts] == ["confirmed", "disputed"]
    assert res.sources and res.usd == pytest.approx(0.006)
    assert llm.count("research") == 1


def test_collect_never_returns_an_empty_fact_list(mini):
    llm = FakeLLM(research=['{"facts": [{"note": "nothing"}]}'])
    with pytest.raises(LLMError):
        collect("Falling into Jupiter", mini, llm)


# --- overstated claims -------------------------------------------------------

def test_a_disputed_fact_stated_flatly_is_caught():
    fact = Fact("Jupiter has a diluted fuzzy core rather than a compact rocky one.",
                "disputed", "s")
    [msg] = overstated("Deep down, Jupiter has a diluted fuzzy core instead.", [fact])
    assert "disputed" in msg


def test_an_openly_hedged_disputed_fact_is_not_called_a_claim():
    fact = Fact(claim="Membership: associated with the Stephenson 2 cluster, but later "
                      "work questions that membership", status="disputed", source="s")
    text = ("That question has no settled answer, because the star is associated "
            "with the Stephenson 2 cluster, while later work has questioned whether "
            "it's truly a member.")
    assert overstated(text, [fact]) == []


def test_a_proposal_is_an_attribution_not_a_claim():
    fact = Fact("The population estimate was about two hundred breeding adults.",
                "speculation", "s")
    assert overstated("Researchers have proposed a population estimate of two "
                      "hundred breeding adults.", [fact]) == []


def test_confirmed_facts_may_be_stated_plainly():
    fact = Fact("Winds in the upper atmosphere reach very high speeds.", "confirmed", "s")
    assert overstated("Winds in the upper atmosphere reach very high speeds.", [fact]) == []


# --- numbers must trace to a source ------------------------------------------

def test_a_figure_absent_from_the_material_is_flagged():
    facts = _facts("The survey examined 97 nearby galaxies.")
    [msg] = unsourced_numbers("Another search examined 1,519 events.", facts)
    assert "1519" in msg


def test_a_figure_present_in_the_material_passes_through_commas():
    facts = _facts("Among 1519 events that passed its filters, nothing held up.")
    assert unsourced_numbers("It found 1,519 events.", facts) == []


def test_a_figure_named_only_in_the_source_line_still_counts():
    facts = [Fact(claim="The search found no candidate signals.", status="confirmed",
                  source="Technosignature Search of 97 Nearby Galaxies")]
    assert unsourced_numbers("They examined 97 galaxies.", facts) == []


def test_small_counts_about_the_script_itself_are_not_claims():
    assert unsourced_numbers("There are 3 ways to read this. Two of them fail.",
                             _facts("Nothing numeric here.")) == []


def test_the_same_stray_figure_is_reported_once():
    text = "It found 1,519 events. Later, 1,519 events again."
    assert len(unsourced_numbers(text, _facts("no digits"))) == 1


def test_a_rounded_figure_from_the_material_is_not_invented():
    facts = _facts("Revenue was 1.181654 billion dollars across 2,896 sites.")
    assert unsourced_numbers("About 1.2 billion dollars came in.", facts) == []
    assert unsourced_numbers("It runs nearly 2,900 sites.", facts) == []


def test_a_more_precise_figure_than_the_material_is_still_flagged():
    facts = _facts("The network runs about 2,900 sites.")
    assert len(unsourced_numbers("It runs 2,896 sites.", facts)) == 1
    assert len(unsourced_numbers("It runs 3,400 sites.", facts)) == 1


def test_a_year_is_never_accepted_as_a_rounding():
    facts = _facts("The comet was first recorded in 1947.")
    assert unsourced_numbers("It happened in 1947.", facts) == []
    assert len(unsourced_numbers("It happened in 1960.", facts)) == 1
    assert len(unsourced_numbers("It happened in 1900.", facts)) == 1
