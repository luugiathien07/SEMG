"""Tests for src/session_builder.py — Usecase 1 (v2) session assembly."""
from pathlib import Path

import numpy as np
import pytest

from src import data_loader as dl
from src.session_builder import (
    parse_mvc, list_ordered_segments, common_valid_channels, build_session_signal,
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


class TestListOrderedSegments:
    def test_sorts_by_mvc_then_condition(self, monkeypatch):
        files = [
            _fake_file(9, "60", 0),
            _fake_file(9, "10_ap_fatigue", 0),
            _fake_file(9, "10", 0),
            _fake_file(9, "fatigue_70", 1),
        ]
        monkeypatch.setattr(dl, "list_files", lambda: files)
        ordered = list_ordered_segments(9)
        assert [f.condition for f in ordered] == [
            "10", "10_ap_fatigue", "60", "fatigue_70"]

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
