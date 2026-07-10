import os
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# Scenarios directories
s1_dir = DATASET_DIR / "scenario01"
s2_dir = DATASET_DIR / "scenario02"
s3_dir = DATASET_DIR / "scenario03"

for d in [s1_dir, s2_dir, s3_dir]:
    d.mkdir(parents=True, exist_ok=True)

raw_files = [f for f in DATASET_DIR.glob("*.csv") if f.is_file()]

def split_and_save(src_path: Path, dst_dir: Path, part1_name: str, part3_name: str):
    print(f"Splitting {src_path.name} -> {dst_dir.name}")
    df = pd.read_csv(src_path, sep=";", header=None, decimal=".")
    n_rows = len(df)
    
    # Part 1: 0 - 30%
    df_part1 = df.iloc[:int(n_rows*0.3)]
    df_part1.to_csv(dst_dir / part1_name, sep=";", header=False, index=False)
    print(f"  Saved {part1_name} ({len(df_part1)} rows)")
    
    # Part 3: 70 - 100%
    df_part3 = df.iloc[int(n_rows*0.7):]
    df_part3.to_csv(dst_dir / part3_name, sep=";", header=False, index=False)
    print(f"  Saved {part3_name} ({len(df_part3)} rows)")

def create_symlink(src_path: Path, dst_dir: Path, new_name: str = None):
    new_name = new_name or src_path.name
    dst_path = dst_dir / new_name
    if dst_path.exists():
        dst_path.unlink()
    os.symlink(src_path.absolute(), dst_path.absolute())

for f in sorted(raw_files):
    print(f"\nProcessing {f.name}...")
    
    # -------------------------------------------------------------
    # Scenario 01: Time Splitting (60 and 70% MVC split)
    # -------------------------------------------------------------
    if "60_emg" in f.name:
        part1 = f.name.replace("60_emg", "60p1_emg")
        part3 = f.name.replace("60_emg", "71p3_emg")
        split_and_save(f, s1_dir, part1, part3)
    elif "fatigue_70_emg" in f.name:
        part1 = f.name.replace("fatigue_70_emg", "59p1_emg")
        part3 = f.name.replace("fatigue_70_emg", "70p3_emg")
        split_and_save(f, s1_dir, part1, part3)
    else:
        create_symlink(f, s1_dir)

    # -------------------------------------------------------------
    # Scenario 02: Fix 10_ap_fatigue
    # -------------------------------------------------------------
    if "10_ap_fatigue" in f.name:
        new_name = f.name.replace("10_ap_fatigue", "71_ap_fatigue")
        create_symlink(f, s2_dir, new_name)
    else:
        create_symlink(f, s2_dir)

    # -------------------------------------------------------------
    # Scenario 03: Combined (Time Splitting + Fix AP Fatigue)
    # -------------------------------------------------------------
    if "10_ap_fatigue" in f.name:
        new_name = f.name.replace("10_ap_fatigue", "71_ap_fatigue")
        create_symlink(f, s3_dir, new_name)
    elif "60_emg" in f.name:
        part1 = f.name.replace("60_emg", "60p1_emg")
        part3 = f.name.replace("60_emg", "71p3_emg")
        split_and_save(f, s3_dir, part1, part3)
    elif "fatigue_70_emg" in f.name:
        part1 = f.name.replace("fatigue_70_emg", "59p1_emg")
        part3 = f.name.replace("fatigue_70_emg", "70p3_emg")
        split_and_save(f, s3_dir, part1, part3)
    else:
        create_symlink(f, s3_dir)

print("\nScenarios created successfully!")
