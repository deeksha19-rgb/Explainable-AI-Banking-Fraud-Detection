"""Inspect a small sample of the downloaded PaySim CSV without loading it all."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
preferred = RAW_DIR / "PS_20174392719_1491204439457_log.csv"
if preferred.exists():
    path = preferred
else:
    files = sorted(RAW_DIR.glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Put the PaySim CSV in {RAW_DIR}; expected its standard name or exactly one CSV file.")
    path = files[0]
df = pd.read_csv(path, nrows=5)
print(f"File: {path}")
print("Columns:")
print(df.columns.tolist())
print("\nFirst five rows:")
print(df.to_string(index=False))
print("\nColumn data types:")
print(df.dtypes)
