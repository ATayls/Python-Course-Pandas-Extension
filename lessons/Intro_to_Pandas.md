---

layout: lessons
lessons title: Intro to pandas
------------------------------

## Lesson Aims

* Load tabular data into pandas with clear, reliable labels.
* Select and filter data safely using `.loc` and `.iloc`.
* Reproduce earlier NumPy statistics and Matplotlib plots using pandas’ APIs.
* Combine many CSVs into one tidy table using `concat` and `melt`.
* Summarise with `groupby`/`agg`, handle missing and odd values explicitly.
* Export clean results and (optionally) wrap the workflow in a small CLI.

## What is pandas?

Pandas is a library that provides the `DataFrame`: a labeled, table-like data structure that makes tabular analysis expressive and robust. If NumPy gives you fast arrays, pandas layers labels, missing-data handling, and rich reshaping and grouping operations on top—ideal for day-to-day data analysis.

## Prerequisites & Setup

* Assumes prior comfort with: Python basics, NumPy arrays, Matplotlib plotting, loops, simple functions, and reading multiple files.
* Data: `inflammation-*.csv` (each file is patients × days).
* Environment: Python ≥3.8; install `pandas`, `numpy`, `matplotlib`. A Jupyter notebook is recommended for the lesson, but a text editor + REPL also works.

---

## From Arrays to DataFrames

We’ll load one inflammation CSV and give columns meaningful names.

```python
import pandas as pd

df = pd.read_csv("data/inflammation-01.csv", header=None)
df.columns = [f"day_{i}" for i in range(df.shape[1])]  # name columns day_0..day_59
df.head()
```

```python
df.info()
```

```python
df.describe()
```

### Quick Tips

* If your CSV has no header row, use `header=None` and then set `df.columns` or pass `names=`.
* `info()` is the fastest way to catch wrong dtypes and unexpected missingness.
* Prefer explicit, human-readable column names to avoid off-by-one mistakes later.

### Exercise 1 (Inspect)

How many **patients** and **days** are in `inflammation-01.csv`? Confirm with both `df.shape` and `df.info()`.

**Solution:**

```python
n_patients, n_days = df.shape
print(n_patients, n_days)
```

---

## Selecting Data Safely

Use **position-based** (`.iloc`) and **label-based** (`.loc`) indexing. Avoid chained indexing when setting values.

```python
# First patient, first 5 days (by position)
df.iloc[0, :5]

# Same using labels
df.loc[0, ["day_0", "day_1", "day_2", "day_3", "day_4"]]
```

```python
# Patient-wise mean over first week
first_week = [f"day_{i}" for i in range(7)]
df[first_week].mean(axis=1)
```

```python
# Filter by a condition, then select a column
mask = df["day_0"] > 0
df.loc[mask, "day_0"].head()
```

### Quick Tips

* **Rule of thumb:** read with either attribute or `[]`, but **write** with `.loc[row_sel, col_sel] = value` to avoid `SettingWithCopyWarning`.
* `axis=0` means “down rows (per column)”; `axis=1` means “across columns (per row)”.
* Don’t chain: `df[df["day_0"]>0]["day_1"]=...` — do `df.loc[df["day_0"]>0, "day_1"] = ...`.

### Exercise 2 (Selection)

Extract the readings for **patient 10** on **days 10–19** inclusive using `.iloc`. Then repeat using `.loc` with your named columns.

**Solution:**

```python
df.iloc[10, 10:20]
df.loc[10, [f"day_{i}" for i in range(10, 20)]]
```

---

## Defining a DataFrame (brief)

A pandas `DataFrame` is defined by **data**, **row labels (index)**, and **column labels**. You can construct one from Python objects directly:

```python
basic = pd.DataFrame(
    {"a": [1, 2], "b": [3.0, 4.5]},   # columns
    index=["row1", "row2"]            # optional row labels
)
basic.dtypes      # per-column data types
basic.index       # row labels
basic.columns     # column labels
```

Clear labels and appropriate dtypes make later selection, grouping, and plotting safer and more readable.

---

## Reproducing Earlier Stats & Plots with pandas

We’ll replicate per-day mean/min/max and plot them using pandas’ plotting (which wraps Matplotlib).

```python
per_day = pd.DataFrame({
    "mean": df.mean(axis=0),
    "min":  df.min(axis=0),
    "max":  df.max(axis=0),
})
per_day.head()
```

```python
ax = per_day.plot(title="Per-day inflammation: mean, min, max")
ax.set_xlabel("day")
ax.set_ylabel("value")
```

```python
# Or single series quickly:
df.mean(axis=0).plot(title="Mean inflammation per day").set_xlabel("day")
```

### Quick Tips

* Many `DataFrame` methods mirror NumPy: `mean`, `std`, `idxmax`, etc.
* Pandas plotting is great for quick EDA; still use `plt.savefig("...")` for reproducibility.
* Prefer vectorised operations to `.apply` for performance and clarity.

### Exercise 3 (Plot Parity)

Recreate the three-line “mean/min/max per day” plot you built earlier with pure Matplotlib, but using pandas. Add axis labels and a title.

**Solution:**

```python
ax = per_day.plot()
ax.set_title("Per-day inflammation summary")
ax.set_xlabel("day")
ax.set_ylabel("value")
```

---

## Creating New Columns

A “new column” is just a labeled `Series` aligned to the existing index. The most common pattern you’ll see is direct column assignment; for conditional writes or chained transforms, prefer `.loc` or `.assign`.

```python
# Common pattern: direct assignment
first_week = [f"day_{i}" for i in range(7)]
df["week1_total"] = df[first_week].sum(axis=1)

# Center a column by subtracting its mean
df["day_0_centered"] = df["day_0"] - df["day_0"].mean()

# Conditional column (boolean mask -> int)
df["any_week1_gt5"] = (df[first_week].max(axis=1) > 5).astype(int)

df.head()
```

```python
# Safe in-place creation with .loc (useful when writing conditionally or to slices)
df.loc[:, "week1_mean"] = df[first_week].mean(axis=1)
```

```python
# Chainable creation with .assign (helps readability in pipelines)
df = df.assign(
    week1_total=df[first_week].sum(axis=1),
    day_0_centered=df["day_0"] - df["day_0"].mean(),
    any_week1_gt5=(df[first_week].max(axis=1) > 5).astype(int),
)
```

### Quick Tips

* Use direct assignment for simple, one-off columns.
* When writing conditionally or to avoid `SettingWithCopyWarning`, use `.loc[row_sel, "col"] = value`.
* Prefer `assign(new_col=...)` when you want readable, chainable transformations.

### Exercise 4 (Create)

Make a column called `day_0_diff` that is `day_0 - day_1`. Then create `week1_std` as the standard deviation across days 0–6 for each patient.

**Solution:**

```python
df = df.assign(
    day_0_diff=df["day_0"] - df["day_1"],
    week1_std=df[first_week].std(axis=1, ddof=0),
)
```

---

### Quick Tips

* Use `assign(new_col=...)` to keep transformations readable and chainable.
* When writing conditionally, use `.loc[row_sel, "col"] = value`.
* Avoid chained indexing when setting values.

### Exercise 4 (Create)

Make a column called `day_0_diff` that is `day_0 - day_1`. Then create `week1_std` as the standard deviation across days 0–6 for each patient.

**Solution:**

```python
df = df.assign(
    day_0_diff=df["day_0"] - df["day_1"],
    week1_std=df[first_week].std(axis=1, ddof=0),
)
```

---

## New Column from an Existing Column (Squared)

You often want a simple transformation of one column.

```python
# Three equivalent, vectorised ways:
df["day_0_sq"] = df["day_0"] ** 2
# or
df["day_0_sq"] = df["day_0"].pow(2)
# or
df = df.assign(day_0_sq=df["day_0"] * df["day_0"])
```

If missing values exist, they propagate (NaNs stay NaN), which is usually the right behaviour.

### Exercise 5 (Power Up)

Create `day_3_sq` and `day_7_cubed`. Then compute `week1_energy` as the sum of squares across the first week.

**Solution:**

```python
df = df.assign(
    day_3_sq=df["day_3"].pow(2),
    day_7_cubed=df["day_7"].pow(3),
    week1_energy=(df[first_week] ** 2).sum(axis=1),
)
```

---

## New Column via a User-Defined Function

When no vectorised operation fits, use a small, pure Python function. Prefer **Series-wise** functions (returning a Series) or **row-wise** `apply(..., axis=1)` as a last resort.

### Option A: Function operating on a Series (preferred when possible)

```python
def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sigma = s.std(ddof=0)
    return (s - mu) / sigma

# Z-score of day_0 across patients
df["day_0_z"] = zscore(df["day_0"])
```

This stays vectorised and fast because it uses pandas ops internally.

### Option B: Row-wise function with `apply(axis=1)` (use sparingly)

```python
# A tiny "flare score" using a weighted sum of the first 3 days:
weights = {"day_0": 0.5, "day_1": 0.3, "day_2": 0.2}

def flare_score(row: pd.Series) -> float:
    return sum(row[k] * w for k, w in weights.items())

df["flare_score"] = df.apply(flare_score, axis=1)
```

Row-wise `apply` is clear when logic mixes multiple columns in nontrivial ways, but it’s slower on large tables.

### Quick Tips

* If your function is elementwise (e.g., square, clip, log), try `Series.map`, `Series.apply` (elementwise), or direct operators before `DataFrame.apply(axis=1)`.
* Keep User Define Functions pure (no side effects) and small; they’re easier to test and reason about.

### Exercise 6 (User Defined Function)

Write a function `week1_status(row)` that returns `"ok"` if the **mean** of week 1 is `< 5`, else `"review"`. Add it as a column.

**Solution:**

```python
def week1_status(row: pd.Series) -> str:
    return "ok" if row[first_week].mean() < 5 else "review"

df["week1_status"] = df.apply(week1_status, axis=1)
```

---

## Putting It Together: Combine → Tidy → Summarise → Export

This mini-workflow reads all `inflammation-*.csv` files, adds clear labels, reshapes to a tidy table, summarises, and writes results to disk.

```python
from pathlib import Path
import pandas as pd

# 1) Load and label many files
def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = [f"day_{i}" for i in range(df.shape[1])]
    df = df.reset_index(names="patient")          # patient id 0..N-1
    return df.assign(source=path.name)            # keep file/source label

files = sorted(Path("data").glob("inflammation-*.csv"))
wide = pd.concat([load_one(p) for p in files], ignore_index=True)

# 2) Reshape to tidy (long) format: one value per row
tidy = wide.melt(
    id_vars=["source", "patient"],
    var_name="day",
    value_name="inflammation",
)

# 3) Handle missing/odd values explicitly (adjust to your context)
# Example policies:
#   - no negatives allowed
#   - replace sentinel -999 with NA (if present in your data)
tidy = tidy.replace(-999, pd.NA)
tidy["inflammation"] = tidy["inflammation"].clip(lower=0)

# Optional quick QA checks
missing_rate_by_file = (
    tidy["inflammation"].isna()
    .groupby(tidy["source"])
    .mean()
    .round(3)
)
print("Missing rate by file:\n", missing_rate_by_file)

# 4) Summarise with groupby/agg (mean, sd, count) per file × day
summary = (
    tidy
    .groupby(["source", "day"], as_index=False)
    .agg(
        mean=("inflammation", "mean"),
        sd=("inflammation", "std"),
        n=("inflammation", "size"),
    )
)

# 5) Export clean results
Path("out").mkdir(exist_ok=True)
summary.to_csv("out/per_day_summary.csv", index=False)
print("Wrote out/per_day_summary.csv")
```

### Why this pattern?

* `reset_index(names="patient")` gives you a stable patient identifier before melting.
* Keeping `source` (the filename) preserves provenance, so summaries can be compared across files.
* Tidy data (`melt`) makes `groupby`/`agg` straightforward and composable.

---

## (Optional) Tiny CLI Wrapper

If you’d like to run the workflow from the shell without touching a notebook, save the below as `scripts/summarise_inflammation.py`:

```python
#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = [f"day_{i}" for i in range(df.shape[1])]
    df = df.reset_index(names="patient")
    return df.assign(source=path.name)

def main(pattern: str, out_csv: str):
    files = sorted(Path().glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern!r}")
    wide = pd.concat([load_one(p) for p in files], ignore_index=True)
    tidy = wide.melt(id_vars=["source", "patient"], var_name="day", value_name="inflammation")
    tidy = tidy.replace(-999, pd.NA)
    tidy["inflammation"] = tidy["inflammation"].clip(lower=0)
    summary = (tidy.groupby(["source", "day"], as_index=False)
                    .agg(mean=("inflammation","mean"),
                         sd=("inflammation","std"),
                         n=("inflammation","size")))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="Glob like 'data/inflammation-*.csv'")
    ap.add_argument("--out", default="out/per_day_summary.csv")
    args = ap.parse_args()
    main(args.pattern, args.out)
```

Run:

```bash
python scripts/summarise_inflammation.py "data/inflammation-*.csv" --out out/per_day_summary.csv
```

---