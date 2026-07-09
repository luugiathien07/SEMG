"""Tests for src/session_builder.py — Usecase 1 (v2) session assembly."""
from pathlib import Path

import numpy as np

from src import data_loader as dl
from src.session_builder import (
    parse_mvc, list_ordered_segments, common_valid_channels,
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
