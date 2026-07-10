import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src import pipeline as pl
from src import config
from src import data_loader

results = pl.run_pipeline(use_cache=True).results
df = data_loader.load_features(config.CACHE_DIR / "features.parquet")
df_test = df[df["subject"] == config.TEST_SUBJECT]

print("Test Subject 9 Segments:")
for file_name in df_test["file"].unique():
    subset = df_test[df_test["file"] == file_name]
    true_label = subset["label"].iloc[0]
    print(f"File: {file_name} - True Label: {true_label}")
    # predict using SVM
    for r in results:
        X = subset[r.features]
        preds = r.model.predict(X)
        pred_label = int(sum(preds) > len(preds)/2) # majority vote
        print(f"  {r.model_name}: {pred_label}")
