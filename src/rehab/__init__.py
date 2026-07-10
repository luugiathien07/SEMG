"""Usecase 2 — rehab recovery tracking on the synthetic "physiomio" dataset.

Kept fully separate from the Usecase 1 fatigue-classification pipeline
(`src/config.py`, `src/data_loader.py`, ...): different dataset, different
sampling rate, different question (recovery trend across sessions, not a
binary Normal/Fatigue label per channel).
"""
