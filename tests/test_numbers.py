from __future__ import annotations

import pytest

from narration.numbers import HARMLESS, figures, find_numbers


def values(text: str) -> list[str]:
    return [n.value for n in find_numbers(text)]


# Sentences taken from scripts the pipeline wrote in the first eval run.
@pytest.mark.parametrize("sentence,expected", [
    ("The atmosphere pushed against it with a braking force equal to two hundred "
     "fifty times Earth's gravity.", ["250"]),
    ("It measured Jupiter from nanobar pressure levels down to roughly twenty-four "
     "bars.", ["24"]),
    ("The probe was roughly one hundred twelve miles below its entry point.", ["112"]),
    ("NASA says pressure there could reach six hundred fifty million pounds on every "
     "square inch of your body.", ["650000000"]),
    ("Early self-igniting matches in China, around five seventy-seven A.D., used "
     "pinewood sticks coated in sulphur.", ["577"]),
    ("A neutron star packs about one point four Suns into a sphere about twenty "
     "kilometers across,", ["1.4", "20"]),
    ("with a density of about seven times ten to the fourteenth grams per cubic "
     "centimeter.", ["7", "10", "14"]),
    ("On Earth, a sugar cube of neutron-star material would weigh about one trillion "
     "kilograms.", ["1000000000000"]),
    ("That change happens at about zero point six of nuclear saturation density.", ["0.6"]),
    ("They place those crossings when sea levels were about one hundred twenty-five "
     "metres lower than today.", ["125"]),
])
def test_spoken_numbers_from_real_scripts(sentence, expected):
    assert values(sentence) == expected


@pytest.mark.parametrize("text,expected", [
    ("1,519 events", ["1519"]),
    ("about 22.7 atmospheres", ["22.7"]),
    ("650 million pounds", ["650000000"]),
    ("1.4 billion people", ["1400000000"]),
    ("7 x 10 to the 14 grams", ["7", "10", "14"]),
    ("between 10 and 20 kilometers", ["10", "20"]),
    ("a hundred and twelve people", ["112"]),
    ("a thousand years", ["1000"]),
    ("two million three thousand", ["2003000"]),
    ("nineteen hundred", ["1900"]),
    ("in nineteen forty-seven", ["1947"]),
    ("by twenty twenty-four", ["2024"]),
    ("the twenty-first time", ["21"]),
    ("two point five times", ["2.5"]),
    ("one point four million tonnes", ["1400000"]),
    ("about two point five billion years", ["2500000000"]),
    ("ten-20 range", ["10", "20"]),
])
def test_digits_scales_years_decimals_and_ordinals(text, expected):
    assert values(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("a sugar cube", []),
    ("hundreds of years", []),
    ("thousands of stars", []),
    ("one two three", ["1", "2", "3"]),
    ("three and a half", ["3"]),
    ("twenty, thirty", ["20", "30"]),
    ("a one-way trip", ["1"]),
    ("one twenty-kilometre leg", ["1", "20"]),
    ("wait a second", []),
])
def test_what_is_not_one_number(text, expected):
    assert values(text) == expected


def test_surface_text_is_kept():
    [n] = find_numbers("roughly one hundred twelve miles")
    assert n.text == "one hundred twelve"


def test_figures_skip_small_counts():
    text = "One probe, three layers, two hundred fifty times, ten miles, 1,000 years."
    assert [n.value for n in figures(text)] == ["250"]
    assert {"0", "1", "10", "100", "1000"} <= HARMLESS
