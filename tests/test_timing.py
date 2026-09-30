from __future__ import annotations

import itertools
import random

import pytest

from narration.timing import (PRESPLIT_RATIO, SOFT_BREAK_PENALTY, SYLLABLES_PER_SENTENCE,
                              SYLLABLES_PER_WORD, Frame, Segmentation, TimingError, Unit,
                              allocate_frames, build_units, count_syllables,
                              estimate_seconds, max_units, partition, segment,
                              segment_sections, split_sentences, words_budget)

# --- syllables ---------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("a", 1), ("the", 1), ("money", 2), ("make", 1), ("simple", 2),
    ("extraordinarily", 6), ("rhythm", 1), ("free", 1), ("purchase", 2),
])
def test_syllable_counting(word, expected):
    assert count_syllables(word) == expected


def test_syllable_counting_is_a_heuristic_not_a_dictionary():
    # "idea" has three syllables but one vowel group for "ea". The contract
    # is statistical: single-word errors average out over a frame.
    assert count_syllables("idea") == 2


def test_real_words_always_have_a_syllable():
    for w in ("x", "hmm", "shh", "rhythms"):
        assert count_syllables(w) >= 1


def test_punctuation_only_token_has_no_syllables():
    assert count_syllables("'") == 0
    assert count_syllables("") == 0


# --- sentences ---------------------------------------------------------------

def test_split_keeps_plain_text_intact():
    text = "One fee seems harmless. One subscription feels tiny! Why? Because."
    assert split_sentences(text) == [
        "One fee seems harmless.", "One subscription feels tiny!", "Why?", "Because."]


def test_abbreviation_is_not_a_sentence_end():
    got = split_sentences("Interest looks busy while principal barely changes. "
                          "U.S. Bank says some formulas use a flat percentage.")
    assert got == ["Interest looks busy while principal barely changes.",
                   "U.S. Bank says some formulas use a flat percentage."]
    assert len(split_sentences("It happened at 9 a.m. Then everyone left.")) == 1
    assert len(split_sentences("Dr. Hale spoke first and nobody argued.")) == 1


def test_a_lone_abbreviation_joins_the_next_sentence():
    assert split_sentences("The rate rose. Etc. Then it fell again.") == [
        "The rate rose.", "Etc. Then it fell again."]
    assert split_sentences("Money is tight. It resets.") == ["Money is tight.", "It resets."]


def test_an_exclamation_inside_a_name_is_not_a_sentence_end():
    assert split_sentences(
        "The event became known as the Wow! Signal, a name that sounds cheerful. "
        "It made the observer write Wow! on the paper."
    ) == ["The event became known as the Wow! Signal, a name that sounds cheerful.",
          "It made the observer write Wow! on the paper."]
    assert split_sentences("Stop! The door is open.") == ["Stop!", "The door is open."]


def test_military_ranks_are_not_sentence_ends():
    assert split_sentences(
        "Another aircraft carried a different sensor. Lt. Cmdr. Jane Hale "
        "recorded the video. Capt. Smith agreed."
    ) == ["Another aircraft carried a different sensor.",
          "Lt. Cmdr. Jane Hale recorded the video.", "Capt. Smith agreed."]


def test_split_survives_closing_quote():
    assert len(split_sentences('He said "no." Then he left.')) == 2


def test_splitting_never_changes_the_text():
    text = ("You fall. U.S. teams watched it! The Wow! Signal stayed. Etc. "
            "Why? Lt. Hale knew.  Spaces   collapse.")
    assert " ".join(split_sentences(text)) == " ".join(text.split())


# --- duration ----------------------------------------------------------------

def test_longer_text_takes_longer():
    assert estimate_seconds("Money is tight.") < estimate_seconds(
        "Money is tight because fixed bills consume most of the income before "
        "any real choice appears.")


def test_faster_voice_reads_the_same_text_quicker():
    text = "Bulk buying saves money only when the unit price is lower."
    assert (estimate_seconds(text, rate=6.30, pause=0.695)
            < estimate_seconds(text, rate=3.66, pause=0.472))


def test_sentence_pause_is_modelled_separately_from_rate():
    # One voice read 91 syllables in 21 sentences in 29.0 s, where the rate
    # alone predicts 14.4 s: on chopped text half the time is pauses.
    chopped = "Money runs out. Bills arrive first. Rent takes a share."
    flowing = "Money runs out because bills arrive first and rent takes a share."
    assert (estimate_seconds(chopped, rate=6.30, pause=0.695)
            > estimate_seconds(flowing, rate=6.30, pause=0.695))


def test_empty_text_is_free():
    assert estimate_seconds("   ") == 0.0


# --- partition ---------------------------------------------------------------

def _units(*durations: float, hard: bool = True) -> list[Unit]:
    return [Unit(f"u{i}", d, hard) for i, d in enumerate(durations)]


def _cost(groups: list[list[Unit]], target: float) -> float:
    return sum((sum(u.seconds for u in g) - target) ** 2
               + (0 if g[-1].hard else SOFT_BREAK_PENALTY) for g in groups)


def test_partition_is_optimal_on_a_case_greedy_gets_wrong():
    # Greedy takes 5+1 and 5, leaving 2; the quadratic cost prefers 5 / 6 / 2.
    groups = partition(_units(5, 1, 5, 2), 3, 6.0)
    assert [sum(u.seconds for u in g) for g in groups] == [5, 6, 2]


def test_partition_matches_brute_force_on_random_inputs():
    rng = random.Random(7)
    for _ in range(60):
        m, k = rng.randint(3, 9), rng.randint(1, 4)
        if k > m:
            continue
        units = [Unit(f"u{i}", round(rng.uniform(0.5, 9), 2), rng.random() > 0.3)
                 for i in range(m)]
        units[-1].hard = True
        best = min(
            _cost([units[a:b] for a, b in zip((0, *cuts), (*cuts, m))], 6.0)
            for cuts in itertools.combinations(range(1, m), k - 1))
        assert _cost(partition(units, k, 6.0), 6.0) == pytest.approx(best)


def test_partition_uses_every_frame():
    groups = partition(_units(*([3.0] * 10)), 5, 6.0)
    assert len(groups) == 5 and all(groups)


def test_partition_preserves_order_and_content():
    units = _units(1, 2, 3, 4, 5)
    assert [u for g in partition(units, 3, 4.0) for u in g] == units


def test_partition_refuses_when_pieces_are_fewer_than_frames():
    with pytest.raises(TimingError):
        partition(_units(1, 2), 5, 6.0)


def test_partition_refuses_zero_frames():
    with pytest.raises(TimingError):
        partition(_units(1, 2), 0, 6.0)


def test_partition_prefers_sentence_boundaries():
    soft = [Unit("a", 3.0, False), Unit("b", 3.0, True),
            Unit("c", 3.0, False), Unit("d", 3.0, True)]
    assert all(g[-1].hard for g in partition(soft, 2, 6.0))


def test_partition_breaks_a_sentence_only_when_it_saves_enough():
    # A sentence of two clauses (a, b) followed by sentence c, two frames.
    # Breaking after a costs the penalty (4); keeping a+b whole costs 9 + 9.
    worth_it = [Unit("a", 6.0, False), Unit("b", 3.0, True), Unit("c", 3.0, True)]
    assert not partition(worth_it, 2, 6.0)[0][-1].hard
    # Here keeping it whole costs 1 + 1, less than the penalty.
    not_worth_it = [Unit("a", 6.0, False), Unit("b", 1.0, True), Unit("c", 5.0, True)]
    assert partition(not_worth_it, 2, 6.0)[0][-1].hard


# --- building units ----------------------------------------------------------

LONG = ("Bulk buying saves money only when the unit price is genuinely lower, but a "
        "pallet of yogurt at half price is a full-price loss if a third of it "
        "expires in the fridge before anyone eats it.")


def test_long_sentence_is_split_at_a_clause():
    units = build_units(LONG, max_seconds=6.0)
    assert len(units) > 1
    assert not units[0].hard
    assert units[-1].hard


def test_clause_split_keeps_the_punctuation_on_the_left():
    units = build_units(LONG, max_seconds=6.0)
    assert units[0].text.endswith(",")
    assert " ".join(u.text for u in units) == LONG


def test_split_prefers_a_semicolon_over_a_comma():
    text = ("The hull held for a while, the crew said later; the engines failed "
            "long before anyone reached the surface of the moon.")
    units = build_units(text, max_seconds=1.0)
    assert units[0].text.endswith(";") or any(u.text.endswith(";") for u in units)


def test_bare_as_is_never_a_break_point():
    text = "The storm was as big as a city and as old as the oldest maps anyone kept."
    for u in build_units(text, max_seconds=0.1):
        assert not u.text.startswith("as ")


def test_short_sentences_are_never_split():
    units = build_units("Money is tight. Bills come first. Saving feels impossible.",
                        max_seconds=6.0 * PRESPLIT_RATIO)
    assert len(units) == 3 and all(u.hard for u in units)


def test_build_units_splits_further_when_frames_outnumber_sentences():
    text = ("Money is tight because bills come first, and saving feels impossible "
            "when the paycheck is gone by the tenth.")
    assert len(build_units(text, max_seconds=60.0, min_units=3)) >= 3


def test_build_units_gives_up_loudly_on_impossible_text():
    with pytest.raises(TimingError, match="too short"):
        build_units("Too short.", max_seconds=60.0, min_units=10)


def test_splitter_does_not_give_up_on_an_unsplittable_longest_piece():
    # The longest sentence has no clause boundary; shorter ones do.
    text = ("Prehistoric partner selection remained deeply constrained by seasonal "
            "mobility patterns across wide territories. The camps were not islands, "
            "and people moved between them. Bones can say a great deal, because "
            "isotopes record where a person grew up.")
    assert len(build_units(text, max_seconds=6.0, min_units=5)) >= 5


def test_max_units_counts_what_the_text_really_yields():
    text = ("Money is tight. The fixed bills arrive first, and any real saving "
            "feels impossible. The month resets.")
    assert max_units(text) == 4


def test_max_units_refuses_to_make_a_stub():
    # "Bills come first," is three words, below the minimum chunk.
    assert max_units("Money is tight. Bills come first, and saving fails. It resets.") == 3


# --- whole frames ------------------------------------------------------------

SCRIPT = " ".join([
    "You can earn more money and still feel permanently broke.",
    "Income alone does not decide whether wealth grows.",
    "The real damage comes from habits so normal they stop looking expensive.",
    "One fee seems harmless.",
    "One subscription feels tiny.",
    "One upgrade looks deserved.",
    "But repeated every month, small leaks consume thousands of dollars.",
    "The goal is not guilt, and the goal is not shame either.",
    "The goal is to find one leak you can close this week.",
    "That single change compounds far more than most people expect.",
])


def test_segment_produces_exactly_the_frames_asked_for():
    seg = segment(SCRIPT, 6, frame_seconds=6.0)
    assert [f.number for f in seg.frames] == [1, 2, 3, 4, 5, 6]


def test_segment_loses_no_text():
    seg = segment(SCRIPT, 6, frame_seconds=6.0)
    assert " ".join(f.text for f in seg.frames).split() == SCRIPT.split()


def test_segment_keeps_sentences_whole_when_it_can():
    assert segment(SCRIPT, 6, frame_seconds=6.0).broken_sentences == 0


def test_uniform_timeline_is_a_regular_grid():
    seg = segment(SCRIPT, 6, frame_seconds=6.0, uniform=True)
    assert [f.start for f in seg.frames] == [0, 6, 12, 18, 24, 30]


def test_measured_timeline_has_no_drift_by_construction():
    seg = segment(SCRIPT, 6, frame_seconds=6.0, uniform=False)
    for prev, cur in zip(seg.frames, seg.frames[1:]):
        assert cur.start == pytest.approx(prev.start + prev.seconds)


def test_worst_drift_accumulates_against_the_grid():
    seg = Segmentation(frames=[Frame(1, "a", 8.0, 0), Frame(2, "b", 8.0, 6),
                               Frame(3, "c", 2.0, 12)], target_seconds=6.0)
    assert seg.worst_drift == pytest.approx(4.0)
    assert seg.total_seconds == pytest.approx(18.0)


# --- sections ----------------------------------------------------------------

def test_allocation_is_proportional_and_exact():
    quota = allocate_frames([60.0, 30.0, 30.0], 80)
    assert sum(quota) == 80 and quota[0] > quota[1]


def test_every_section_gets_at_least_one_frame():
    quota = allocate_frames([100.0, 1.0, 1.0], 5)
    assert sum(quota) == 5 and min(quota) >= 1


def test_allocation_refuses_when_frames_are_fewer_than_sections():
    with pytest.raises(TimingError):
        allocate_frames([1.0] * 10, 5)


def test_allocation_never_exceeds_what_a_section_can_split_into():
    assert allocate_frames([100.0, 10.0], 10, capacity=[8, 2]) == [8, 2]


def test_allocation_redistributes_the_surplus_to_sections_with_room():
    quota = allocate_frames([100.0, 10.0, 10.0], 12, capacity=[4, 20, 20])
    assert quota[0] <= 4 and sum(quota) == 12


def test_allocation_refuses_when_total_capacity_is_short():
    with pytest.raises(TimingError, match="pieces"):
        allocate_frames([50.0, 50.0], 20, capacity=[3, 3])


def test_no_frame_spans_two_sections():
    sections = [
        ("THE TRAP", "One fee seems harmless. One subscription feels tiny. "
                     "One upgrade looks deserved every single time."),
        ("THE FIX", "The goal is not guilt. The goal is one leak closed. "
                    "That single change compounds more than people expect."),
    ]
    seg, owner = segment_sections(sections, 4, frame_seconds=6.0)
    assert len(seg.frames) == 4
    assert owner == sorted(owner)
    for frame, idx in zip(seg.frames, owner):
        assert frame.text in " ".join(sections[idx][1].split())


def test_section_segmentation_survives_an_uneven_script():
    sections = [
        ("LONG ONE", " ".join(f"A short sentence number {i} lands here." for i in range(1, 13))),
        ("SHORT ONE", "One idea only. Then a second one."),
        ("LAST ONE", "A closing thought. And the question you leave with."),
    ]
    seg, owner = segment_sections(sections, 12, frame_seconds=6.0)
    assert len(seg.frames) == 12 and owner == sorted(owner) and len(set(owner)) == 3


def test_a_clause_broken_between_frames_keeps_its_comma():
    sections = [("Route", "A single licence covers all machines operated by that person, "
                          "because that operator stocks them and collects the cash; the "
                          "paperwork follows the route rather than the address, and that "
                          "distinction decides who answers when a machine goes dark.")]
    seg, _ = segment_sections(sections, 4, frame_seconds=3.0, uniform=False)
    broken = [f.text for f in seg.frames if not f.ends_sentence]
    assert broken
    assert all(t.endswith((",", ";")) for t in broken)
    assert " ".join(f.text for f in seg.frames) == sections[0][1]


def test_frame_numbers_run_across_sections():
    seg, _ = segment_sections([("a", "One two three four. Five six seven eight."),
                               ("b", "Nine ten eleven twelve. Thirteen fourteen fifteen.")],
                              4, frame_seconds=3.0)
    assert [f.number for f in seg.frames] == [1, 2, 3, 4]


# --- budget ------------------------------------------------------------------

def test_budget_differs_by_voice_as_measured():
    fast = words_budget(80, 6.0, 6.30, 0.695)
    slow = words_budget(80, 6.0, 3.66, 0.472)
    assert 1560 <= fast <= 1720
    assert 1000 <= slow <= 1130


def test_budget_scales_with_runtime():
    assert words_budget(80, 6.0, 4.75) < words_budget(120, 6.0, 4.75)


def test_timing_constants_are_consistent():
    # The pause is charged per sentence, so syllables per sentence must be
    # syllables per word times the words per sentence of the same scripts.
    assert abs(SYLLABLES_PER_SENTENCE - SYLLABLES_PER_WORD * 13.6) < 0.6
    heavy = words_budget(60, 6.0, 4.46, 0.506, syllables_per_word=1.828,
                         syllables_per_sentence=26.3)
    assert words_budget(60, 6.0, 4.46, 0.506) > heavy


def test_a_script_written_to_budget_fills_the_runtime():
    budget = words_budget(10, 6.0, 4.46, 0.506)
    sentence = "The wind pushes you sideways and the clouds keep rolling past you."
    words_per = len(sentence.split())
    text = " ".join([sentence] * round(budget / words_per))
    secs = estimate_seconds(text, rate=4.46, pause=0.506)
    assert 50 <= secs <= 70
