from __future__ import annotations

import pytest

from narration.cards import CardError, ChannelCard, available


def test_bundled_channels():
    assert available() == ["history", "science", "space"]
    for name in available():
        card = ChannelCard.load(name)
        assert card.id == name and card.character_name in card.cta
        assert card.frame_seconds == pytest.approx(6.0)


def test_word_budget_follows_the_voice():
    # Same runtime, different narrators: the slow voice gets fewer words.
    history, space = ChannelCard.load("history"), ChannelCard.load("space")
    assert history.voice.syllables_per_second < space.voice.syllables_per_second
    assert history.words < space.words
    assert 760 <= history.words <= 840 and 910 <= space.words <= 990


def test_sentence_target_leaves_headroom_over_frames(mini):
    assert mini.sentences >= round(mini.frames * 1.1)
    assert mini.words_per_sentence == pytest.approx(mini.words / mini.sentences, abs=0.05)


def test_character_frame_range(mini):
    assert mini.character_frames == (2, 4)


def test_a_card_without_measured_pace_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("character_name: A\ncharacter_bible: b\nrole: r\n"
                    "narrative_direction: n\ncta: c\nvoice:\n  label: x\n", encoding="utf-8")
    with pytest.raises(CardError, match="syllables_per_second"):
        ChannelCard.from_file(path)


def test_a_card_with_an_empty_field_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("character_name: A\ncharacter_bible: ''\nrole: r\n"
                    "narrative_direction: n\ncta: c\n", encoding="utf-8")
    with pytest.raises(CardError, match="character_bible"):
        ChannelCard.from_file(path)


def test_video_summary_shows_the_budget(video):
    text = video.summary()
    assert "jupiter-fall" in text and f"{video.channel.words} words" in text
