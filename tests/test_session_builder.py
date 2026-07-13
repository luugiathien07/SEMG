"""Tests for src/session_builder.py — Usecase 1 (v2) session assembly."""
from pathlib import Path

import numpy as np
import pytest

from src import data_loader as dl
from src.session_builder import (
    parse_mvc, is_post_fatigue, list_ordered_segments, common_valid_channels,
    build_session_signal, build_session_signal_avg,
)


def _fake_file(subject: int, condition: str, label: int) -> dl.FileInfo:
    return dl.FileInfo(
        path=Path(f"Sujet_{subject}_{condition}_emg.csv"),
        subject=subject, condition=condition, label=label,
    )


class TestParseMvc:
    def test_plain_number(self):
        assert parse_mvc("10") == 10

    def test_ap_fatigue_suffix(self):
        assert parse_mvc("10_ap_fatigue") == 10

    def test_fatigue_prefix(self):
        assert parse_mvc("fatigue_70") == 70


class TestIsPostFatigue:
    def test_ap_fatigue_condition_is_post_fatigue(self):
        assert is_post_fatigue("10_ap_fatigue") is True

    def test_fatigue_prefix_condition_is_not_post_fatigue(self):
        # "fatigue_70" is the fatiguing bout itself (chronologically part of
        # the ascending-intensity ramp), not a post-fatigue retest.
        assert is_post_fatigue("fatigue_70") is False

    def test_plain_number_condition_is_not_post_fatigue(self):
        assert is_post_fatigue("60") is False


class TestListOrderedSegments:
    def test_sorts_ascending_mvc_with_post_fatigue_retest_last(self, monkeypatch):
        files = [
            _fake_file(9, "60", 0),
            _fake_file(9, "10_ap_fatigue", 0),
            _fake_file(9, "10", 0),
            _fake_file(9, "fatigue_70", 1),
            _fake_file(9, "90", 1),
        ]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        ordered = list_ordered_segments(9)
        # "10_ap_fatigue" is a post-fatigue retest — measured chronologically
        # AFTER the fatiguing bout and the 90% MVC peak, despite its low
        # %MVC number, so it must sort last, not second.
        assert [f.condition for f in ordered] == [
            "10", "60", "fatigue_70", "90", "10_ap_fatigue"]

    def test_filters_to_subject(self, monkeypatch):
        files = [_fake_file(9, "10", 0), _fake_file(5, "20", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        ordered = list_ordered_segments(9)
        assert len(ordered) == 1
        assert ordered[0].subject == 9


class TestCommonValidChannels:
    def test_intersects_masks(self, monkeypatch):
        files = [_fake_file(9, "10", 0), _fake_file(9, "20", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        channels_by_file = {
            files[0].path: np.array([[1.0, 0.0, 1.0]]),
            files[1].path: np.array([[1.0, 1.0, 0.0]]),
        }
        monkeypatch.setattr(dl, "load_channels", lambda path: channels_by_file[path])
        result = common_valid_channels(9)
        assert result == [0]


class TestBuildSessionSignal:
    def test_concatenates_in_mvc_order_with_segment_bounds(self, monkeypatch):
        files = [_fake_file(9, "20", 0), _fake_file(9, "10", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        channels_by_file = {
            files[0].path: np.array([[10.0], [10.0], [10.0]]),  # "20", 3 samples
            files[1].path: np.array([[1.0], [1.0]]),            # "10", 2 samples
        }
        monkeypatch.setattr(dl, "load_channels", lambda path: channels_by_file[path])

        signal, segments = build_session_signal(9, channel=0)

        np.testing.assert_allclose(signal, [1.0, 1.0, 10.0, 10.0, 10.0])
        assert [s.file.condition for s in segments] == ["10", "20"]
        assert segments[0].mvc == 10 and segments[1].mvc == 20
        assert segments[0].start_sample == 0 and segments[0].end_sample == 2
        assert segments[1].start_sample == 2 and segments[1].end_sample == 5

    def test_raises_if_fewer_than_two_files(self, monkeypatch):
        files = [_fake_file(9, "10", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        with pytest.raises(ValueError):
            build_session_signal(9, channel=0)


class TestBuildSessionSignalAvg:
    def test_averages_only_valid_channels_and_concatenates_in_mvc_order(self, monkeypatch):
        files = [_fake_file(9, "20", 0), _fake_file(9, "10", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        # 3 channels each; channel 1 (idx 1) is dead (all zero) in file "10"
        # -> common_valid_channels should exclude idx 1 from the average.
        channels_by_file = {
            files[0].path: np.array([  # "20", 2 samples, 3 channels
                [10.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
            ]),
            files[1].path: np.array([  # "10", 2 samples, 3 channels
                [1.0, 0.0, 3.0],
                [1.0, 0.0, 3.0],
            ]),
        }
        monkeypatch.setattr(dl, "load_channels", lambda path: channels_by_file[path])

        signal, segments = build_session_signal_avg(9)

        # valid channels = idx 0 and idx 2 (idx 1 is all-zero in file "10")
        # "10": mean(1.0, 3.0) = 2.0 per sample; "20": mean(10.0, 30.0) = 20.0
        np.testing.assert_allclose(signal, [2.0, 2.0, 20.0, 20.0])
        assert [s.file.condition for s in segments] == ["10", "20"]
        assert segments[0].start_sample == 0 and segments[0].end_sample == 2
        assert segments[1].start_sample == 2 and segments[1].end_sample == 4

    def test_raises_if_fewer_than_two_files(self, monkeypatch):
        files = [_fake_file(9, "10", 0)]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        with pytest.raises(ValueError):
            build_session_signal_avg(9)
