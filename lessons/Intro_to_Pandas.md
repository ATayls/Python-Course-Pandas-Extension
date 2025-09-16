---
layout: lessons
lessons title: Intro to pandas
---

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

## Many Files → One Tidy Table

Load all `inflammation-*.csv`, keep provenance (`file`, `patient`), then **melt** to tidy format: one row = one observation.

```python
import glob

paths = sorted(glob.glob("data/inflammation-*.csv"))
frames = []
for p in paths:
    tmp = pd.read_csv(p, header=None)
    tmp["file"] = p
    tmp["patient"] = range(len(tmp))
    frames.append(tmp)

wide = pd.concat(frames, ignore_index=True)
wide.columns = [*(f"day_{i}" for i in range(df.shape[1])), "file", "patient"]
```

```python
long = wide.melt(
    id_vars=["file", "patient"],
    var_name="day",
    value_name="inflammation",
)

# Make day numeric
long["day"] = long["day"].str.replace("day_", "", regex=False).astype(int)
long.head()
```

```python
summary = (long
    .groupby("day")["inflammation"]
    .agg(["mean", "min", "max"])
    .reset_index()
)
summary.head()
```

### Quick Tips

* Always carry identifiers (`file`, `patient`) **before** melting so you don’t lose context.
* `concat` + `melt` is a robust pattern: many wide CSVs → one analysis table.
* After reshaping, check types and coerce with `.astype(...)` as needed.

### Exercise 4 (Batch Summary)

Compute per-**file**, per-**day** mean inflammation and plot a small-multiples figure (one panel per file).

**Solution:**

```python
by_file_day = (long
    .groupby(["file", "day"])["inflammation"]
    .mean()
    .unstack("file")
)

axes = by_file_day.plot(subplots=True, sharex=True, legend=False)
for ax in axes:
    ax.set_xlabel("day")
    ax.set_ylabel("mean inflammation")
```

---

## Cleaning & Validation

Detect missing or suspicious values, then handle them explicitly. Bridge to defensive programming with assertions.

```python
# Missingness overview
long.isna().sum()
```

```python
# Domain-specific sanity checks (example thresholds)
suspicious = long.query("inflammation < 0 or inflammation > 100")
suspicious.head()
```

```python
# Replace negatives with NA, then drop rows with NA in the analysis column
long["inflammation"] = long["inflammation"].where(long["inflammation"].ge(0), pd.NA)
clean = long.dropna(subset=["inflammation"])
```

```python
# Invariants
assert clean["inflammation"].ge(0).all(), "Negative readings remain"
assert clean["day"].between(0, 59).all(), "Day out of expected range"
```

### Quick Tips

* Prefer `where`/`mask` for conditional replacement—they preserve index alignment.
* Use `pd.NA` and `.isna()` consistently; mixing `None`/`NaN` can surprise you.
* Avoid `inplace=True` for clarity; assignment is easier to reason about.

### Exercise 5 (Data Hygiene)

Inject a few negative values into `long` and write code to (a) **flag**, (b) **replace with NA**, (c) **drop**, and (d) **report** how many were affected.

**Solution:**

```python
# Example injection (for practice only)
long.loc[long.sample(5, random_state=0).index, "inflammation"] = -1

neg = long["inflammation"] < 0
count_neg = neg.sum()
long.loc[neg, "inflammation"] = pd.NA
clean = long.dropna(subset=["inflammation"])
print("Replaced negatives:", count_neg)
```

---

## Reshaping & Richer Summaries

Pivot to a matrix-like view; compute grouped statistics with counts for honest intervals.

```python
heat = clean.pivot_table(
    index="patient",
    columns="day",
    values="inflammation",
    aggfunc="mean",
)
heat.shape
```

```python
by_file_day = (clean
    .groupby(["file", "day"])["inflammation"]
    .agg(mean="mean", std="std", n="count")
    .reset_index()
)
by_file_day.head()
```

### Quick Tips

* `pivot` reshapes; `pivot_table` can **aggregate** duplicates via `aggfunc=`.
* Named aggregations in `.agg(mean="mean", std="std", ...)` are explicit and readable.
* Always keep `n` alongside summary stats (`mean`, `std`) to interpret variability.

### Exercise 6 (Suspicious Patterns)

Find **runs** of strictly increasing per-file means of length ≥10 (too-perfect trends). Return `file`, `start_day`, `length`.

**Solution:**

```python
def find_runs(s):
    inc = s.diff().gt(0)
    grp = inc.ne(inc.shift()).cumsum()
    runs = (inc.groupby(grp).sum()
            .reset_index(drop=True))
    # runs counts only True segments; need start indices as well
    starts = (inc.groupby(grp).apply(lambda g: g.index[0])).reset_index(drop=True)
    out = []
    for start, length in zip(starts, runs):
        if length >= 10:
            out.append((start - length + 1, length))
    return out

results = []
for f, sub in by_file_day.groupby("file"):
    for start, length in find_runs(sub["mean"].reset_index(drop=True)):
        results.append({"file": f, "start_day": int(start), "length": int(length)})

pd.DataFrame(results)
```

---

## Export & Optional CLI Wrap

Write results to disk and optionally wrap into a small command-line tool with `argparse`.

```python
summary.to_csv("results/per_day_summary.csv", index=False)
```

```python
# analyse.py
import argparse, pandas as pd, glob, sys

def load_summary(paths, stat="mean"):
    frames = []
    for p in paths:
        tmp = pd.read_csv(p, header=None)
        tmp["file"] = p
        tmp["patient"] = range(len(tmp))
        frames.append(tmp)
    wide = pd.concat(frames, ignore_index=True)
    n_days = len(wide.columns) - 2
    wide.columns = [*(f"day_{i}" for i in range(n_days)), "file", "patient"]
    long = wide.melt(id_vars=["file","patient"], var_name="day", value_name="inflammation")
    long["day"] = long["day"].str.replace("day_", "", regex=False).astype(int)
    agg_map = {"mean":"mean","min":"min","max":"max"}[stat]
    return (long.groupby("day")["inflammation"].agg(agg_map).reset_index(name=stat))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarise inflammation CSVs")
    ap.add_argument("pattern", help="Glob like data/inflammation-*.csv")
    ap.add_argument("-o","--out", default="summary.csv")
    ap.add_argument("--stat", choices=["mean","min","max"], default="mean")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        sys.exit("No files matched.")
    out = load_summary(paths, stat=args.stat)
    out.to_csv(args.out, index=False)
```

### Quick Tips

* Keep functions **pure** (inputs → outputs) to make them testable.
* Fail early if `glob` matches nothing; avoid silently writing empty outputs.
* For larger CLIs use `argparse` (or libraries like `click`/`typer` later).

### Exercise 7 (CLI Test)

Run the CLI on `inflammation-*.csv` and confirm the row count equals `n_days`. Then add a flag `--stat mean|min|max` (already shown) and test each option.

**Solution:**

```bash
python analyse.py "data/inflammation-*.csv" -o results/mean.csv --stat mean
python analyse.py "data/inflammation-*.csv" -o results/min.csv  --stat min
python analyse.py "data/inflammation-*.csv" -o results/max.csv  --stat max
```

---

## Practical Exercises

#### Exercise 8: Median vs Mean

Load `inflammation-03.csv` and compute the **median per day**. Does it differ materially from the mean? Briefly justify.

**Solution:**

```python
df3 = pd.read_csv("data/inflammation-03.csv", header=None)
df3.columns = [f"day_{i}" for i in range(df3.shape[1])]
median_vs_mean = pd.DataFrame({
    "mean":   df3.mean(axis=0),
    "median": df3.median(axis=0),
})
median_vs_mean.head()
```

#### Exercise 9: Mean ± SEM

Combine all files, tidy them, and compute per-day **standard error of the mean** (`sem = std / sqrt(n)`). Plot mean ± SEM.

**Solution:**

```python
import numpy as np

sem = (long.groupby("day")["inflammation"]
           .agg(mean="mean", std="std", n="count")
           .assign(sem=lambda d: d["std"] / np.sqrt(d["n"])))
ax = sem["mean"].plot(title="Mean ± SEM by day")
(sem["mean"] + sem["sem"]).plot(ax=ax)
(sem["mean"] - sem["sem"]).plot(ax=ax)
ax.set_xlabel("day"); ax.set_ylabel("inflammation")
```

#### Exercise 10: Outlier Files

Identify any file with ≥1 day where `file_mean(day) > global_mean(day) + 3*global_std(day)`. List file and day.

**Solution:**

```python
global_stats = (long.groupby("day")["inflammation"]
                    .agg(gmean="mean", gstd="std")
                    .reset_index())

file_day = (long.groupby(["file","day"])["inflammation"]
                 .mean()
                 .reset_index(name="fmean"))

merged = file_day.merge(global_stats, on="day", how="left")
outliers = merged.query("fmean > gmean + 3*gstd")[["file","day","fmean"]]
outliers.sort_values(["file","day"])
```

---

## Key Points

* A `DataFrame` is a labeled 2D array—keep identifiers like `file` and `patient` from the start.
* Use `.loc`/`.iloc` for clarity; **never rely on chained indexing** when assigning.
* Pandas mirrors many NumPy reductions (`mean`, `std`, …) and offers convenient plotting.
* To combine many files: `concat`, then `melt` to get **tidy** data; analyse with `groupby`/`agg`.
* Validate assumptions early with `info()`, `describe()`, and simple **assertions**; handle missing data intentionally.
* Export results with `to_csv`; optional: wrap workflows in small, testable CLIs with `argparse`.

#### Lesson Inspired by Software Carpentries’ *python-novice-inflammation* and adapted for a pandas-focused bridge.
