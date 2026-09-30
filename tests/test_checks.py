from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import CLEAN_CHAPTERS, CLEAN_CTA
from narration import checks
from narration.cards import ChannelCard
from narration.checks import (Problem, abstract_density, banned_scaffolds, budget,
                              character_balance, check_narration, check_visuals,
                              concreteness, cta_follows_card, duplicate_frames,
                              ending_repeats_itself, finale_returns_to_viewer,
                              hook_quality, missing_grounding, ngram_repeats,
                              paraphrased_neighbours, production_leak, questions,
                              questions_on_the_seams, repeated_ngrams,
                              repeated_openings, sentence_rhythm, sentences_of,
                              spoken_register, unspeakable_syntax, viewer_address,
                              visual_variety)


def test_the_clean_fixture_passes_every_check(mini):
    assert check_narration(sentences_of(CLEAN_CHAPTERS), mini) == []
    assert cta_follows_card(CLEAN_CTA, mini) == []


def test_problem_prints_kind_message_and_positions():
    assert str(Problem("duplicate", "x", at=[3, 7])) == "[duplicate] x (at 3, 7)"
    assert str(Problem("grounding", "y", advisory=True)) == "~[grounding] y"


# --- degeneration ------------------------------------------------------------

def test_duplicates_are_found_with_every_position():
    texts = ["The clouds are thick.", "Wind.", "the clouds  are thick.", "The clouds are thick."]
    [p] = duplicate_frames(texts)
    assert p.at == [1, 3, 4]


def test_no_duplicates_no_problem():
    assert duplicate_frames(["a b", "c d"]) == []


def test_banned_scaffold_is_caught_case_insensitively():
    [p] = banned_scaffolds(["Fine.", "THIS MATTERS BECAUSE nothing else does."])
    assert p.at == [2] and "this matters because" in p.message


def test_retention_plea_is_banned():
    assert banned_scaffolds(["Stay with me: the answer leads to monsters."])


def test_same_opening_is_allowed_twice_not_three_times():
    two = ["One fee seems small here.", "One fee seems small there.", "Then more."]
    assert repeated_openings(two) == []
    three = two + ["One fee seems small again."]
    [p] = repeated_openings(three)
    assert p.at == [1, 2, 4]


def test_repeated_phrase_counts_and_exempts_proper_nouns():
    texts = ["the clouds keep rolling past", "and the clouds keep rolling on",
             "so the clouds keep rolling"] + ["Great Red Spot storms"] * 3
    found = dict(ngram_repeats(texts))
    assert found["the clouds keep rolling"] == 3
    assert not any("red spot" in g for g in found)


def test_repeated_ngrams_reports_at_most_ten():
    words = ("able baker charlie delta echo foxtrot golf hotel india juliet kilo "
             "lima mike november oscar papa quebec romeo").split()
    texts = [" ".join(words)] * 3
    assert len(repeated_ngrams(texts)) == 10
    assert len(ngram_repeats(texts)) > 10


def test_short_echo_after_a_long_sentence_is_not_a_paraphrase():
    echo = ["Even a dramatic mystery still has to fit the paper, including blank "
            "spaces where proof should have been.",
            "Blank spaces don't become proof."]
    assert not paraphrased_neighbours(echo)
    real = ["The museum destroyed most of the files in nineteen seventy three.",
            "Most of the museum files were destroyed in nineteen seventy three."]
    [p] = paraphrased_neighbours(real)
    assert p.at == [1, 2]


# --- hook --------------------------------------------------------------------

def test_citation_in_the_first_seconds_is_caught():
    bad = ["You could look straight past it.",
           "For six years Hubble watched its gravity bend a star's light."]
    assert any("hubble" in str(p) for p in hook_quality(bad, [6.0, 6.0]))


def test_citation_after_the_window_is_fine():
    texts = ["You fall."] * 6 + ["Later, NASA measured the winds."]
    assert not [p for p in hook_quality(texts, [6.0] * 7) if "source" in p.message]


def test_citation_marker_matches_whole_words_only():
    ok = ["You breathe through a nasal passage, and the air is thin.",
          "Your ears pop, yet the climb has barely started."]
    assert not [p for p in hook_quality(ok, [6.0, 6.0]) if "source" in p.message]
    bad = ["You are watching this.", "NASA kept the file for forty years."]
    assert [p for p in hook_quality(bad, [6.0, 6.0]) if "source" in p.message]


def test_long_first_sentence_is_caught():
    long_open = ["You are looking at a wall of galaxies stretching more than one "
                 "billion light-years across the sky above you."]
    assert any("first sentence" in p.message for p in hook_quality(long_open, [6.0]))


def test_hedge_in_the_opening_is_caught():
    texts = ["You wake up. This is hypothetical, of course."]
    assert any("caveat" in p.message for p in hook_quality(texts, [6.0]))


def test_opening_without_the_viewer_is_caught():
    assert any("address" in p.message
               for p in hook_quality(["The planet is large. It is far."], [6.0]))


def test_empty_hook_is_silent():
    assert hook_quality([], []) == []


# --- register and rhythm -----------------------------------------------------

def test_written_register_is_caught():
    written = ["You are looking at a wall of galaxies. That is the largest thing we "
               "have found, and it does not look orderly. You cannot see it from here."]
    spoken = ["You're looking at a wall of galaxies. That's the largest thing we've "
              "found, and it doesn't look orderly. You can't see it from here."]
    assert spoken_register(written)
    assert not spoken_register(spoken)


def test_curly_apostrophes_count_as_contractions():
    assert not spoken_register(["You’re here. It’s cold. Don’t move. We’ve got time."])


def test_flat_rhythm_is_caught():
    flat = ["Every sentence here runs to about eleven words in total."] * 25
    assert sentence_rhythm(flat)
    varied = flat[:20] + ["Gone.", "Cut to black.", "It worked.", "Nobody knew.", "Not once."]
    assert not sentence_rhythm(varied)


def test_rhythm_needs_enough_material():
    assert not sentence_rhythm(["One long sentence about something or other."])


def test_abstract_nouns_are_caught():
    heavy = ["The decision produced extinction and a reduction in visibility for "
             "the population, and the situation required intervention."]
    [p] = abstract_density(heavy)
    assert "decision" in p.message
    plain = ["He chose to leave. The fire went out. It got colder, and nobody could "
             "see the far wall of the cave any more."]
    assert not abstract_density(plain)


def test_semicolon_and_long_list_are_caught():
    bad = ["Leaks hide in payouts, rent, labor, repairs, and missing customers; "
           "capital covers equipment and deposits."]
    assert len(unspeakable_syntax(bad)) == 2
    assert not unspeakable_syntax(["It's simple. You pay rent, wages and tax."])


def test_an_introductory_phrase_is_not_a_list_item():
    assert not unspeakable_syntax(
        ["Without it, warmth, light, and cooked food become harder to replace."])
    assert unspeakable_syntax(
        ["Without it, warmth, light, shelter, and cooked food disappear."])


# --- ending ------------------------------------------------------------------

def test_duplicated_ending_is_caught():
    doubled = ["Which object or mystery out there should Kip visit next?",
               "Tell Kip which object or mystery out there it should visit next."]
    assert ending_repeats_itself(doubled)
    fine = ["The sky above you is unusually crowded tonight.",
            "Tell Kip where it should point the dish tomorrow."]
    assert not ending_repeats_itself(fine)


def test_dead_ending_about_the_past_is_caught():
    past = ["A shell or pigment could make that invisible map easier to read.",
            "So how did early humans find partners without villages or cities?",
            "They likely found them through connected bands, meeting places, "
            "exchange and relatives.",
            "The world was mobile, but human relationships could be durable."]
    assert finale_returns_to_viewer(past, [6.0] * 4)


def test_ending_that_turns_onto_the_viewer_passes():
    turned = ["You are the direct heir of that harsh history.",
              "So every time your hand reaches for another sweet, don't blame yourself.",
              "That isn't your weakness.",
              "That's two and a half million years of evolution speaking in you."]
    assert finale_returns_to_viewer(turned, [6.0] * 4) == []


def test_only_the_last_thirty_seconds_count_for_the_finale():
    texts = ["You were there once."] + ["The rocks cooled slowly."] * 6
    assert finale_returns_to_viewer(texts, [6.0] * 7)


# --- call to action ----------------------------------------------------------

def test_final_cta_must_stay_the_channel_question(mini):
    advice = ("Today, notice where your body looks for relief from heat and glare. "
              "Then ask whether you need more machinery.")
    assert cta_follows_card(advice, mini)
    assert not cta_follows_card("Tell us which place out there Kip should visit next.", mini)
    assert cta_follows_card("", mini)


def test_a_card_that_names_no_character_needs_a_question(mini):
    card = replace(mini, cta="Ask which question the viewer wants answered next.")
    assert not cta_follows_card("Which question should we answer next?", card)
    assert not cta_follows_card("Tell us which question we should answer next.", card)
    assert cta_follows_card("Carry that boundary into your next purchase today.", card)
    assert cta_follows_card("Share this video and subscribe for more.", card)


def test_character_name_is_matched_as_a_whole_word():
    kip = ChannelCard.load("space")
    assert kip.character_name == "Kip"
    assert cta_follows_card("Skip the intro and tell us what you'd explore?", kip)
    mira = ChannelCard.load("science")
    assert cta_follows_card("Which admiration-worthy discovery comes next?", mira)
    assert not cta_follows_card("Which question should Mira explore next?", mira)


# --- questions and address ---------------------------------------------------

def test_questions_bunched_at_the_end_are_caught():
    assert questions_on_the_seams(["a."] * 20 + ["and then what?"])


def test_questions_spread_across_the_script_pass():
    spread = ["a?"] + ["b."] * 9 + ["c?"] + ["d."] * 8 + ["e?"]
    assert questions_on_the_seams(spread) == []


def test_all_questions_in_the_last_quarter_is_not_enough():
    [p] = questions_on_the_seams(["a."] * 16 + ["q1?", "q2?", "q3?", "q4?"])
    assert "last quarter" in p.message


def test_no_question_mark_at_all_is_caught():
    assert questions(["A flat statement."])
    assert not questions(["Really?"])


def test_viewer_address_density():
    assert viewer_address(["The planet turns slowly in the dark and nobody sees it."] * 5)
    assert not viewer_address(["You feel the planet turn under your feet."])
    assert viewer_address([]) == []


# --- numbers and production words --------------------------------------------

def test_a_script_read_like_a_report_is_sent_back():
    body = " ".join(["word"] * 100) + " " + " ".join(f"{i + 1}00 units" for i in range(40))
    [p] = concreteness([body])
    assert p.kind == "overload" and not p.advisory


def test_a_few_numbers_are_fine():
    text = "You fall 400 miles in an hour. " + "The clouds keep rolling past you. " * 12
    assert not concreteness([text])


def test_production_words_are_caught():
    [p] = production_leak(["In this script we look at Jupiter.", "Fine."])
    assert p.at == [1]


def test_a_spoken_rule_about_presenting_facts_is_caught():
    bad = ["Neither side can be presented as settled fact here, so the "
           "disagreement must remain visible."]
    assert production_leak(bad)
    assert not production_leak(["Later work has questioned whether the star belongs there."])
    assert production_leak(["Separate confirmed fact from possibility."])
    assert production_leak(["Those possibilities remain open, because the material "
                            "here doesn't give you a public test."])
    assert not production_leak(["The hull material held up under pressure."])


# --- budget ------------------------------------------------------------------

def test_budget_passes_within_tolerance_and_fails_outside(mini):
    assert budget(sentences_of(CLEAN_CHAPTERS), mini) == []
    [p] = budget(["You fall. It's dark."], mini)
    assert p.kind == "budget" and "-" in p.message


def test_budget_catches_an_overlong_script(mini):
    long = [c["narration"] for c in CLEAN_CHAPTERS] * 2
    [p] = budget(long, mini)
    assert "+" in p.message


# --- visuals -----------------------------------------------------------------

def test_repeated_scenes_are_caught():
    same = ["a campfire at night in a cave"] * 5
    kinds = [p.message for p in visual_variety(same)]
    assert any("unique scenes" in k for k in kinds)
    assert any("one scene" in k for k in kinds)


def test_distinct_scenes_pass():
    assert visual_variety(["a cave at dawn with smoke", "a river crossing in winter",
                           "a hand holding a flint blade"]) == []


def test_character_share_and_clumping(mini):
    lo, hi = mini.character_frames
    assert (lo, hi) == (2, 4)
    assert character_balance([True] * 1 + [False] * 9, mini)
    assert not character_balance([True, False] * 2 + [False] * 6, mini)
    assert character_balance([True] * 5 + [False] * 5, mini)


def test_character_run_longer_than_four_is_caught():
    card = replace(ChannelCard.load("history"), frames=20)
    present = [True] * 5 + [False] * 5 + [True] * 2 + [False] * 8
    [p] = character_balance(present, card)
    assert "in a row" in p.message


def test_check_visuals_combines_both(mini):
    assert check_visuals(["same scene here now"] * 10, [False] * 10, mini)


# --- grounding ---------------------------------------------------------------

def test_missing_facts_are_a_note_not_a_paid_repair():
    [p] = missing_grounding([])
    assert p.advisory
    assert missing_grounding(["a fact"]) == []


@pytest.mark.parametrize("name", [n for n in dir(checks) if n.isupper()
                                  and isinstance(getattr(checks, n), tuple)])
def test_phrase_lists_are_lowercase(name):
    # Matching lowercases the text, so an uppercase entry would never match.
    assert all(x == x.lower() for x in getattr(checks, name))


def test_spelled_out_figures_count_toward_the_ceiling():
    report = ("It fell two hundred fifty miles in forty-seven minutes at eighty-two "
              "degrees under sixty-one bars. ") * 4
    assert concreteness([report])
    assert not concreteness(["You fall. One probe, three layers, a thousand years, "
                             "and nobody sees you go. " * 3])
