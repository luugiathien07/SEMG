import os
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

# Splitting configs
# Format: name: (normal_end_pct, fatigue_start_pct)
SPLIT_CONFIGS = {
    "scenario01_sub20": (0.2, 0.8),
    "scenario01_sub30": (0.3, 0.7),
    "scenario01_sub40": (0.4, 0.6),
    "scenario01_sub50": (0.5, 0.5),
}

raw_files = [f for f in DATASET_DIR.glob("*.csv") if f.is_file()]

def split_and_save(src_path: Path, dst_dir: Path, part1_name: str, part3_name: str, normal_end: float, fatigue_start: float):
    print(f"Splitting {src_path.name} -> {dst_dir.name} ({normal_end*100}% - {fatigue_start*100}%)")
    df = pd.read_csv(src_path, sep=";", header=None, decimal=".")
    n_rows = len(df)
    
    # Part 1: Normal
    df_part1 = df.iloc[:int(n_rows * normal_end)]
    if not df_part1.empty:
        df_part1.to_csv(dst_dir / part1_name, sep=";", header=False, index=False)
    
    # Part 3: Fatigue
    df_part3 = df.iloc[int(n_rows * fatigue_start):]
    if not df_part3.empty:
        df_part3.to_csv(dst_dir / part3_name, sep=";", header=False, index=False)

def create_symlink(src_path: Path, dst_dir: Path, new_name: str = None):
    new_name = new_name or src_path.name
    dst_path = dst_dir / new_name
    if dst_path.exists():
        dst_path.unlink()
    os.symlink(src_path.absolute(), dst_path.absolute())

for sub_name, (norm_pct, fat_pct) in SPLIT_CONFIGS.items():
    dst_dir = DATASET_DIR / sub_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating {sub_name} ---")
    for f in sorted(raw_files):
        if "60_emg" in f.name:
            part1 = f.name.replace("60_emg", "60p1_emg")
            part3 = f.name.replace("60_emg", "71p3_emg")
            split_and_save(f, dst_dir, part1, part3, norm_pct, fat_pct)
        elif "fatigue_70_emg" in f.name:
            part1 = f.name.replace("fatigue_70_emg", "59p1_emg")
            part3 = f.name.replace("fatigue_70_emg", "70p3_emg")
            split_and_save(f, dst_dir, part1, part3, norm_pct, fat_pct)
        else:
            # 10_ap_fatigue and all others are just symlinked exactly as they are
            create_symlink(f, dst_dir)

print("\nSub-scenarios created successfully!")
