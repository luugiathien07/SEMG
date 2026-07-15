"""Tests for src/session_builder.py — Usecase 1 (v2) session assembly."""
from pathlib import Path

import numpy as np
import pytest

from src import data_loader as dl
from src.session_builder import (
    parse_mvc, is_post_fatigue, list_ordered_segments, common_valid_channels,
    build_session_signal, build_session_signal_avg, build_session_column, trim_for_display,
    SegmentInfo,
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


class TestTrimForDisplay:
    FS = 100  # small fs keeps the synthetic arrays short and readable

    def _seg(self, condition: str, start: int, end: int) -> SegmentInfo:
        return SegmentInfo(
            file=_fake_file(9, condition, 0), mvc=parse_mvc(condition),
            start_sample=start, end_sample=end,
        )

    def test_trims_ramp_in_and_ramp_out(self):
        fs = self.FS
        # 1s near-silent ramp-in + 3s loud plateau + 1s near-silent ramp-out.
        ramp = np.full(fs, 0.01)
        plateau = np.full(3 * fs, 1.0)
        x = np.concatenate([ramp, plateau, ramp])
        segments = [self._seg("60", 0, len(x))]

        trimmed, new_segments = trim_for_display(x, segments, fs)

        assert len(trimmed) < len(x)
        # plateau (all 1.0) must survive the trim
        assert np.all(trimmed[new_segments[0].start_sample:new_segments[0].end_sample] > 0)
        assert trimmed.max() == pytest.approx(1.0)
        assert new_segments[0].file.condition == "60"
        assert new_segments[0].mvc == 60

    def test_flat_segment_is_not_trimmed(self):
        fs = self.FS
        x = np.full(2 * fs, 0.5)
        segments = [self._seg("10", 0, len(x))]

        trimmed, new_segments = trim_for_display(x, segments, fs)

        np.testing.assert_allclose(trimmed, x)
        assert new_segments[0].start_sample == 0
        assert new_segments[0].end_sample == len(x)

    def test_segment_shorter_than_one_chunk_does_not_crash(self):
        fs = self.FS
        x = np.array([0.1, 0.2, 0.3])
        segments = [self._seg("10", 0, len(x))]

        trimmed, new_segments = trim_for_display(x, segments, fs)

        np.testing.assert_allclose(trimmed, x)
        assert new_segments[0].end_sample - new_segments[0].start_sample == len(x)

    def test_concatenated_segments_have_contiguous_bounds(self):
        fs = self.FS
        ramp = np.full(fs, 0.01)
        plateau = np.full(2 * fs, 1.0)
        seg_a = np.concatenate([ramp, plateau, ramp])
        seg_b = np.concatenate([ramp, plateau * 2, ramp])
        x = np.concatenate([seg_a, seg_b])
        segments = [
            self._seg("60", 0, len(seg_a)),
            self._seg("70", len(seg_a), len(seg_a) + len(seg_b)),
        ]

        trimmed, new_segments = trim_for_display(x, segments, fs)

        assert new_segments[0].start_sample == 0
        assert new_segments[0].end_sample == new_segments[1].start_sample
        assert new_segments[1].end_sample == len(trimmed)


class TestBuildSessionColumn:
    def test_returns_matrix_shaped_by_requested_channels(self, monkeypatch):
        f1 = dl.FileInfo(path=Path("Sujet_9_10_emg.csv"), subject=9, condition="10", label=0)
        f2 = dl.FileInfo(path=Path("Sujet_9_20_emg.csv"), subject=9, condition="20", label=0)
        monkeypatch.setattr(
            "src.session_builder.dl.list_files", lambda: [f1, f2])
        fake_channels = {
            f1.path: np.arange(300).reshape(100, 3).astype(float),
            f2.path: np.arange(300, 600).reshape(100, 3).astype(float),
        }
        monkeypatch.setattr(
            "src.session_builder.dl.load_channels", lambda p: fake_channels[p])

        signal, segments = build_session_column(9, physical_channels=[1, 3])

        assert signal.shape == (2, 200)  # 2 requested channels, 100+100 samples
        assert len(segments) == 2
        assert segments[0].start_sample == 0 and segments[0].end_sample == 100
        assert segments[1].start_sample == 100 and segments[1].end_sample == 200
        # physical channel 1 -> array index 0; verify it's really column 0, not 1
        expected_first_col = fake_channels[f1.path][:, 0]
        np.testing.assert_array_equal(signal[0, :100], expected_first_col)

    def test_raises_on_fewer_than_two_files(self, monkeypatch):
        f1 = dl.FileInfo(path=Path("Sujet_9_10_emg.csv"), subject=9, condition="10", label=0)
        monkeypatch.setattr(
            "src.session_builder.dl.list_files", lambda: [f1])
        with pytest.raises(ValueError):
            build_session_column(9, physical_channels=[1, 2])
