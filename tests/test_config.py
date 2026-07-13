"""Tests for src/config.py — static configuration values."""
from src import config as cfg


class TestChannelLayout:
    def test_shape_is_13x5(self):
        assert len(cfg.CHANNEL_LAYOUT) == 13
        assert all(len(row) == 5 for row in cfg.CHANNEL_LAYOUT)

    def test_bottom_right_corner_is_cut(self):
        assert cfg.CHANNEL_LAYOUT[12][4] is None

    def test_contains_every_channel_1_to_64_exactly_once(self):
        seen = [
            v for row in cfg.CHANNEL_LAYOUT for v in row if v is not None
        ]
        assert sorted(seen) == list(range(1, 65))

    def test_corner_values_match_reference_image(self):
        # Top two rows (r=1,2), from docs/sơ đồ channel.png:
        # row1: 64, 39, 38, 13, 12 | row2: 63, 40, 37, 14, 11
        assert cfg.CHANNEL_LAYOUT[0] == [64, 39, 38, 13, 12]
        assert cfg.CHANNEL_LAYOUT[1] == [63, 40, 37, 14, 11]
        # Bottom two rows (r=12,13):
        # row12: 53, 50, 27, 24, 1 | row13: 52, 51, 26, 25, None
        assert cfg.CHANNEL_LAYOUT[11] == [53, 50, 27, 24, 1]
        assert cfg.CHANNEL_LAYOUT[12] == [52, 51, 26, 25, None]
