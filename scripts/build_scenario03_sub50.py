import os
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
dst_dir = DATASET_DIR / "scenario03_sub50"
dst_dir.mkdir(parents=True, exist_ok=True)

raw_files = [f for f in DATASET_DIR.glob("*.csv") if f.is_file()]

def split_and_save(src_path: Path, part1_name: str, part3_name: str, split_ratio: float):
    df = pd.read_csv(src_path, sep=";", header=None, decimal=".")
    n_rows = len(df)
    
    # Part 1: Normal
    df_part1 = df.iloc[:int(n_rows * split_ratio)]
    if not df_part1.empty:
        df_part1.to_csv(dst_dir / part1_name, sep=";", header=False, index=False)
    
    # Part 3: Fatigue
    df_part3 = df.iloc[int(n_rows * split_ratio):]
    if not df_part3.empty:
        df_part3.to_csv(dst_dir / part3_name, sep=";", header=False, index=False)

def create_symlink(src_path: Path, new_name: str = None):
    new_name = new_name or src_path.name
    dst_path = dst_dir / new_name
    if dst_path.exists():
        dst_path.unlink()
    os.symlink(src_path.absolute(), dst_path.absolute())

print("Generating scenario03_sub50 (50:50 split + 71_ap_fatigue)...")
for f in sorted(raw_files):
    if "10_ap_fatigue" in f.name:
        new_name = f.name.replace("10_ap_fatigue", "71_ap_fatigue")
        create_symlink(f, new_name)
    elif "60_emg" in f.name:
        part1 = f.name.replace("60_emg", "60p1_emg")
        part3 = f.name.replace("60_emg", "71p3_emg")
        split_and_save(f, part1, part3, 0.5)
    elif "fatigue_70_emg" in f.name:
        part1 = f.name.replace("fatigue_70_emg", "59p1_emg")
        part3 = f.name.replace("fatigue_70_emg", "70p3_emg")
        split_and_save(f, part1, part3, 0.5)
    else:
        create_symlink(f)

print("Done generating. Run evaluation pipeline.")
