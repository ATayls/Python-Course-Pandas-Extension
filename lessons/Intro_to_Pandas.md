---
layout: lessons
title: Intro to Pandas
---

## Lesson Aims

* Load tabular data into pandas with clear, reliable labels.
* Select and filter data safely using `.loc` and `.iloc`.
* Reproduce earlier NumPy statistics and Matplotlib plots using `pandas.DataFrame.plot`.
* Create new columns using vectorised operations and simple functions.
* Combine many CSVs into one table with `concat`.
* Summarise data with `groupby`/`agg`, handle missing values explicitly.
* Export clean results and (optionally) wrap the workflow in a small CLI.

---

## What is pandas?

* Pandas provides the `DataFrame`
* A DataFrame is a labeled, table-like data structure that simplifies data manipulation and analysis.
* NumPy gives you fast arrays,
* Pandas adds labels, missing-data handling, reshaping, and grouping—ideal for day-to-day data analysis.

See [pandas docs](https://pandas.pydata.org/docs/getting_started/overview.html) for more.

* Assumes prior comfort with: Python basics, NumPy arrays, Matplotlib plotting, simple functions, and reading multiple files.
* Data: `inflammation-*.csv` (each file is patients × days) placed in your working directory.
* Environment: Python ≥3.8; install `pandas`, `numpy`, `matplotlib`. Jupyter recommended.

---

## From Arrays to DataFrames

```python
import pandas as pd

df = pd.read_csv("inflammation-01.csv", header=None)
df.columns = [f"day_{i}" for i in range(df.shape[1])]
df.head()
````

```python
df.info()
df.describe()
```

### Quick Tips

* If no header row, use `header=None` then set `df.columns`.
* `info()` quickly shows wrong dtypes or unexpected missing values.
* Clear, human-readable names avoid off-by-one mistakes.

### Exercise 1 (Inspect)

How many **patients** and **days** are in `inflammation-01.csv`?

**Solution:**

```python
n_patients, n_days = df.shape
print(n_patients, n_days)
```

---

## Selecting Data Safely

Use `.iloc` (by position) and `.loc` (by label). Avoid chained indexing when writing.

```python
df.iloc[0, :5]  # First patient, first 5 days
df.loc[0, ["day_0", "day_1", "day_2", "day_3", "day_4"]]
```

```python
first_week = [f"day_{i}" for i in range(7)]
df[first_week].mean(axis=1)  # Patient-wise mean over first week
```

```python
mask = df["day_0"] > 0
df.loc[mask, "day_0"].head()
```

### Quick Tips

* **Read:** use attribute or `[]`.
* **Write:** use `.loc[...] = value`.
* `axis=0` → down rows; `axis=1` → across columns.
* Avoid chaining like `df[df["day_0"]>0]["day_1"]=...`.

### Exercise 2 (Selection)

Get patient 10’s readings for days 10–19 using both `.iloc` and `.loc`.

**Solution:**

```python
df.iloc[10, 10:20]
df.loc[10, [f"day_{i}" for i in range(10, 20)]]
```

---

## Reproducing Stats & Plots with pandas

Compute per-day mean/min/max, then plot.

```python
day_cols = df.columns[df.columns.str.startswith("day_")]

per_day = pd.DataFrame({
    "mean": df[day_cols].mean(axis=0),
    "min":  df[day_cols].min(axis=0),
    "max":  df[day_cols].max(axis=0),
})
```

```python
per_day.plot(title="Per-day inflammation summary")
```

### Quick Tips

* Methods mirror NumPy: `mean`, `std`, `idxmax`, etc.
* `.plot()` is a thin wrapper around Matplotlib—fast for exploration.

### Exercise 3 (Plot Parity)

Recreate the mean/min/max per-day plot with `.plot()`, adding axis labels.

**Solution:**

```python
ax = per_day.plot()
ax.set_xlabel("day")
ax.set_ylabel("value")
```

---

## Creating New Columns

```python
first_week = [f"day_{i}" for i in range(7)]
df["week1_total"] = df[first_week].sum(axis=1)
df["day_0_centered"] = df["day_0"] - df["day_0"].mean()
df["any_week1_gt5"] = (df[first_week].max(axis=1) > 5).astype(int)
```

### Exercise 4 (Create)

Make `day_0_diff = day_0 - day_1` and `week1_std` as sample SD of days 0–6.

**Solution:**

```python
df["day_0_diff"] = df["day_0"] - df["day_1"]
df["week1_std"] = df[first_week].std(axis=1, ddof=1)
```

---
### Creating New Columns (assign)

We can also use `assign` to create new columns. This is useful for chaining operations.

This is slightly more verbose, but has the advantage of not modifying `df` until the end.
It is preferred for more declarative code.

```python
df = df.assign(
    day_0_diff = df["day_0"] - df["day_1"],
    week1_std = df[first_week].std(axis=1, ddof=1)
)
```

---

## Column Transformations

```python
df["day_0_sq"] = df["day_0"] ** 2
df["day_3_sq"] = df["day_3"].pow(2)
df["day_7_cubed"] = df["day_7"].pow(3)
df["week1_energy"] = (df[first_week] ** 2).sum(axis=1)
```

---

## User-Defined Functions

Vectorised where possible:

```python
def zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=1)
    return (s - mu) / sigma

df["day_0_z"] = zscore(df["day_0"])
```

Row-wise only if necessary:

```python
def week1_status(row: pd.Series) -> str:
    return "ok" if row[first_week].mean() < 5 else "review"

df["week1_status"] = df.apply(week1_status, axis=1)
```

---

## Putting It Together: Combine → Summarise → Export

This workflow loads multiple files, labels them, combines, summarises, and exports.
Here it’s written slowly with excessive comments so each step is clear.

```python
from pathlib import Path
import pandas as pd

# Step 1: Load one file and give it clear column names
def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    # Label columns as day_0, day_1, ...
    df.columns = [f"day_{i}" for i in range(df.shape[1])]
    # Reset the row index to get an explicit patient ID
    df = df.reset_index().rename(columns={"index": "patient"})
    # Add a column for provenance: which file this row came from
    return df.assign(source=path.name)

# Step 2: Find all files matching inflammation-*.csv
files = sorted(Path().glob("inflammation-*.csv"))
if not files:
    raise SystemExit("No files matched 'inflammation-*.csv'")

# Step 3: Load all files into one "wide" DataFrame
#    Each row: one patient from one file
wide = pd.concat([load_one(p) for p in files], ignore_index=True)

# Step 4: Compute per-day summaries (mean, sd, n) within each file
day_cols = [c for c in wide.columns if c.startswith("day_")]
summary = (
    wide.groupby("source")[day_cols]
        .agg(["mean", "std", "count"])  # nested columns: (day, stat)
)

# Step 5: Clean up column structure: flatten multiindex
summary.columns = [f"{day}_{stat}" for day, stat in summary.columns]
summary = summary.reset_index()

# Step 6: Write the summary to disk
Path("out").mkdir(exist_ok=True)
summary.to_csv("out/per_day_summary.csv", index=False)
print("Wrote out/per_day_summary.csv")
```

### Why this pattern?

* `reset_index` ensures each patient has an explicit identifier.
* `source` preserves provenance across files.
* Using `groupby("source")[day_cols].agg(...)` lets you compute many stats in one call.
* Flattening the column MultiIndex is optional but makes the CSV easier to read.
* Exporting with `to_csv` makes results reusable in other tools.

---

# Supplementary Material

## (Optional) Tiny CLI Wrapper

To run the workflow from the shell without touching a notebook, save the below as `scripts/summarise_inflammation.py`:

```python
#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def load_one(path: Path):
    """
    Load one inflammation CSV, label columns, add patient ID and source.
    """
    df = pd.read_csv(path, header=None)
    df.columns = [f"day_{i}" for i in range(df.shape[1])]
    df = df.reset_index().rename(columns={"index": "patient"})
    return df.assign(source=path.name)

def main(pattern: str, out_csv: str):
    """
    Load all files matching pattern, summarise per-day stats, write to out_csv.
    """
    # Find files
    files = sorted(Path().glob(pattern))
    # Exit if none found
    if not files:
        # raise error with pattern shown
        raise SystemExit(f"No files matched {pattern!r}")
    # Load all files into one wide DataFrame
    list_of_loaded_files = [load_one(p) for p in files]
    # Concatenate into one big DataFrame
    wide = pd.concat(
        list_of_loaded_files,
        ignore_index=True
    )
    # Reshape to long format for easier grouping
    tidy = wide.melt(
        id_vars=["source", "patient"],
        var_name="day",
        value_name="inflammation"
    )
    # Compute per-day mean, sd, n within each source file
    summary = (
        tidy.groupby(["source", "day"], as_index=False)
            .agg(mean=("inflammation","mean"),
                 sd=("inflammation", lambda s: s.std(ddof=1)),
                 n=("inflammation","count"))
    )
    # Ensure output directory exists
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    # Write summary to CSV
    summary.to_csv(out_csv, index=False)

# This evaluates to TRUE if the script is run directly (not imported)
# It is best practice to put script-running code in this block
if __name__ == "__main__":
    # This is the approach to parse command-line arguments
    # argparse is in the standard library
    ap = argparse.ArgumentParser()
    # Positional argument: glob pattern for input files
    ap.add_argument("pattern", help="Glob like 'inflammation-*.csv' (use 'data/inflammation-*.csv' if files are in a subdir)")
    # Optional argument: output CSV path
    ap.add_argument("--out", default="out/per_day_summary.csv")
    # Parse args and call main
    args = ap.parse_args()
    # Call our user defined main with parsed arguments
    main(args.pattern, args.out)
```

To Run from the command line call the script with a glob pattern and optional output path:
```bash
python scripts/summarise_inflammation.py "inflammation-*.csv" --out out/per_day_summary.csv
```

---
