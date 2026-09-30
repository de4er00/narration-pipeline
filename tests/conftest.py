from __future__ import annotations

from pathlib import Path

import pytest

from narration.cards import ChannelCard, VideoCard

DATA = Path(__file__).resolve().parent / "data"
MINI = DATA / "mini.yaml"

# A one-minute script for the mini channel that passes every check.
CLEAN_CHAPTERS = [
    {"title": "Falling Without a Floor", "narration":
        "You're falling, and there's no ground. The sky below you is a churning "
        "wall of yellow and brown. Wind shoves you sideways at about 400 miles per "
        "hour. Lightning flickers somewhere far beneath your feet. Your stomach "
        "lifts, and it doesn't come back. You can't see where the clouds end. Why "
        "doesn't anything catch you?"},
    {"title": "The Air Gets Heavy", "narration":
        "Jupiter has no surface to land on. The gas just gets thicker as you drop. "
        "Soon it squeezes you like the bottom of a deep ocean. Your ears would have "
        "given up long ago. Then it gets hotter than a kitchen oven, and it keeps "
        "climbing. What happens to a machine sent down here?"},
    {"title": "What's Waiting at the Bottom", "narration":
        "One probe lasted about an hour before the heat won. It melted on the way "
        "down. Nobody knows exactly what sits at the center. The core may be fuzzy "
        "and spread out, or it may be a hard lump. Down there, even the word bottom "
        "stops meaning much. Where would you stop falling, if anywhere? Maybe you'd "
        "never stop at all."},
]
CLEAN_CTA = "Which place out there should Kip visit next?"


@pytest.fixture
def mini() -> ChannelCard:
    return ChannelCard.from_file(MINI)


@pytest.fixture
def video(mini: ChannelCard) -> VideoCard:
    return VideoCard(video_id="jupiter-fall",
                     title="What would happen if you fell into Jupiter?",
                     channel=mini)
