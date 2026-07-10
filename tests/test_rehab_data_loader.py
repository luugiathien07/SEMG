"""Tests for src/rehab/data_loader.py — Usecase 2 (physiomio) data access."""
from src.rehab import config as rc
from src.rehab import data_loader as rdl


class TestListPatients:
    def test_finds_generated_patients(self):
        patients = rdl.list_patients()
        assert patients == sorted(patients)
        assert set(patients) >= {"patient1", "patient2"}


class TestListSessions:
    def test_healthy_arm_has_two_sessions(self):
        sessions = rdl.list_sessions("patient1", "healthy_arm")
        assert [s.session_no for s in sessions] == [1, 2]

    def test_impaired_arm_has_six_sessions_in_order(self):
        sessions = rdl.list_sessions("patient1", "impaired_arm")
        assert [s.session_no for s in sessions] == [1, 2, 3, 4, 5, 6]

    def test_unknown_arm_returns_empty(self):
        assert rdl.list_sessions("patient1", "no_such_arm") == []


class TestLoadSession:
    def test_columns_match_physiomio_schema(self):
        sessions = rdl.list_sessions("patient1", "healthy_arm")
        df = rdl.load_session(sessions[0].path)
        assert set(rc.CHANNEL_COLUMNS) <= set(df.columns)
        assert "movement_type" in df.columns
        assert "Rest" in set(df["movement_type"].unique())


class TestSessionRmsMdf:
    def test_returns_finite_positive_values(self):
        sessions = rdl.list_sessions("patient1", "impaired_arm")
        df = rdl.load_session(sessions[0].path)
        rms, mdf = rdl.session_rms_mdf(df)
        assert rms > 0
        assert mdf > 0

    def test_impaired_arm_recovers_toward_healthy_baseline(self):
        healthy = rdl.list_sessions("patient1", "healthy_arm")
        impaired = rdl.list_sessions("patient1", "impaired_arm")
        healthy_rms, _ = rdl.session_rms_mdf(rdl.load_session(healthy[0].path))
        first_rms, _ = rdl.session_rms_mdf(rdl.load_session(impaired[0].path))
        last_rms, _ = rdl.session_rms_mdf(rdl.load_session(impaired[-1].path))
        assert first_rms < last_rms < healthy_rms * 1.5
