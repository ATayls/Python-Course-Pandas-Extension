---
layout: lessons
title: Intro to StatsModels
---

## Lesson Aims

* Understand what `statsmodels` is and when to use it.
* Fit a minimal Ordinary Least Squares (OLS) regression with the **formula API**.
* Read and interpret the key parts of a model summary.
* Make predictions with confidence intervals.
* Know where to go next: GLMs (e.g., logistic regression)

## What is statsmodels?

`statsmodels` is a Python library for **classical statistical modeling**. It complements pandas/NumPy by providing well-tested implementations of linear models (OLS), generalised linear models (GLM), time series (ARIMA/ETS), and statistical tests, with rich summaries (standard errors, p-values, confidence intervals, diagnostic metrics).

Use `statsmodels` when you need **interpretability**, **inference**, and **statistical diagnostics** beyond what scikit-learn typically exposes.

Link to docs: [Docs](https://www.statsmodels.org/stable/index.html)

Worked Example in docs:

[Regression Examples](https://www.statsmodels.org/stable/examples/index.html#regression-examples)

[Interaction Examples](https://www.statsmodels.org/0.6.1/examples/notebooks/generated/interactions_anova.html)


## Prerequisites & Setup

* Comfort with Python, pandas, and basic plotting (Matplotlib or pandas `.plot`).
* Install:

```bash
pip install statsmodels patsy
```

* We’ll use a built-in dataset to avoid external downloads.

---

## Minimal OLS with the Formula API

The **formula API** (`statsmodels.formula.api` or `smf`) uses R-like model formulas via `patsy`. It integrates smoothly with pandas `DataFrame`s.

```python
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 1) Load a sample dataset
# 'auto' contains mpg and car attributes; if unavailable, use get_rdataset fallback.
try:
    data = sm.datasets.get_rdataset("mtcars").data.rename(columns=str.lower)
except Exception:
    # Fallback: small manual DataFrame (hp: horsepower, wt: weight, mpg: fuel efficiency)
    data = pd.DataFrame({
        "mpg":[21,21,22.8,21.4,18.7,18.1,14.3,24.4,22.8,19.2],
        "hp":[110,110,93,110,175,105,245,62,95,123],
        "wt":[2.62,2.875,2.32,3.215,3.44,3.46,3.57,3.19,3.15,3.44]
    })

# 2) Fit a simple linear model: mpg ~ hp + wt
model = smf.ols("mpg ~ hp + wt", data=data).fit()

# 3) Inspect a compact summary
print(model.summary())
```

### Quick Tips

* Formulas: `y ~ x1 + x2` (additive), `x1:x2` (interaction), `C(cat)` (treat a column as categorical),

* y ~ x1 + x2 → main effects only
* y ~ x1 * x2 → includes x1, x2, and their interaction (x1:x2)
* y ~ x1:x2 → interaction only, no main effects
* y ~ x1 + x2 + x1:x2 → explicit version of x1 * x2

* `ols(...).fit()` returns a results object (`.params`, `.bse`, `.pvalues`, `.conf_int()`).
* Use **clean column names** (snake\_case, no spaces) to avoid quoting hassles in formulas.

---

## Reading the Summary (Essentials)

Key fields from `model.summary()`:

* **coef**: estimated effect per unit change in predictor, holding others fixed.
* **std err**: standard error of the estimate.
* **t, P>|t|**: t-statistic and p-value for the null hypothesis that the coefficient is zero.
* **\[0.025, 0.975]**: 95% confidence interval.
* **R-squared / Adj. R-squared**: variance explained (adjusted penalises extra predictors).
* **F-statistic**: global test that at least one predictor is nonzero.
* **Durbin-Watson**: autocorrelation check (mainly for time-series-like residuals).

Extract the essentials programmatically:

```python
out = pd.DataFrame({
    "coef": model.params,
    "se": model.bse,
    "pval": model.pvalues
})
out["ci_low"], out["ci_high"] = model.conf_int()[0], model.conf_int()[1]
out
```

### Quick Tips

* Large absolute t-statistics and small p-values suggest evidence against a zero coefficient (context matters).
* **Significance** is not **importance**—check effect sizes and domain relevance.

---

## Prediction with Confidence Intervals

```python
new = pd.DataFrame({"hp":[100,150], "wt":[2.5, 3.2]})
pred = model.get_prediction(new).summary_frame(alpha=0.05)  # includes mean_ci & obs_ci
pred
```

* `mean_ci_lower/upper`: CI for the **expected mean** of `y` at these predictors.
* `obs_ci_lower/upper`: PI for a **new observation** (wider).

### Quick Tips

* For reporting, prefer tabular outputs (`summary_frame`) you can save:

```python
pred.to_csv("predictions.csv", index=False)
```

---

## Very Brief Diagnostics

Check linear model assumptions quickly: linearity, homoscedasticity, normal-ish residuals, independence.

```python
import matplotlib.pyplot as plt
import numpy as np

resid = model.resid
fitted = model.fittedvalues

# Residuals vs Fitted (look for funnel patterns)
plt.scatter(fitted, resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted"); plt.ylabel("Residuals"); plt.title("Residuals vs Fitted")
plt.show()

# Q-Q plot (normality check)
sm.qqplot(resid, line="45")
plt.title("Q-Q Plot of Residuals")
plt.show()
```

### Quick Tips

* Patterns in residuals vs fitted suggest model misspecification or heteroscedasticity.
* Consider transformations or alternative models (e.g., GLM) if assumptions are violated.

---

## Where Next? (Signposts)

### Generalised Linear Models (GLM)

For non-Gaussian outcomes (binary counts, rates):

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Logistic regression example: y is 0/1
glm_logit = smf.glm("y ~ x1 + x2", data=df, family=sm.families.Binomial()).fit()
print(glm_logit.summary())
```

* Common families: `Binomial` (logit/probit), `Poisson`, `Gamma` (with appropriate links).

## Practical Mini-Exercises

### Exercise 1 — Add an Interaction

Fit `mpg ~ hp * wt` (which expands to `hp + wt + hp:wt`). Does the interaction appear significant?

**Solution:**

```python
m_int = smf.ols("mpg ~ hp * wt", data=data).fit()
print(m_int.summary().tables[1])  # coefficient table only
```

Explanation: The hp:wt row tests whether the effect of horsepower on mpg depends on weight. A small p-value (e.g., < 0.05) suggests meaningfully different slopes across weights; otherwise, prefer the simpler additive model.

---

### Exercise 2 — Categorical Predictor

Create a binary indicator `heavy = (wt > wt.median())` and fit `mpg ~ hp + C(heavy)`. Interpret the `C(heavy)[T.True]` coefficient.

**Solution:**

```python
data = data.assign(heavy = (data["wt"] > data["wt"].median()))
m_cat = smf.ols("mpg ~ hp + C(heavy)", data=data).fit()
print(m_cat.summary().tables[1])
```

Explanation: C(heavy)[T.True] is the average difference in mpg between heavy and light cars at the same horsepower. A negative, significant coefficient indicates heavier cars get fewer mpg, controlling for hp.

---

### Exercise 3 — Prediction Table

Make predictions for `hp={90, 120, 150}` at `wt=3.0`. Save the full prediction frame to CSV.

**Solution:**

```python
grid = pd.DataFrame({"hp":[90,120,150], "wt":[3.0,3.0,3.0]})
pred = model.get_prediction(grid).summary_frame()
pred.to_csv("pred_grid.csv", index=False)
pred
```

Explanation: The table includes fitted means and intervals. Mean CI reflects uncertainty in the expected mpg at those settings; obs CI is wider, reflecting variability for a single new car.

---

## Key Points

* `statsmodels` focuses on **inference** and **diagnostics** for classical models.
* Use the **formula API** (`smf.ols("y ~ x1 + x2", data=...)`) for concise, readable models.
* Read summaries for coefficients, uncertainty (SE/CI), and goodness-of-fit.
* Use `get_prediction(...).summary_frame()` for CIs/PIs you can report.
* For non-linear means or non-Gaussian outcomes, step up to **GLM**; for temporal dependence, explore **ARIMA/SARIMAX**.

#### Further Reading

* `help(smf.ols)`, `help(sm.GLM)`, `help(ARIMA)`
* Statsmodels documentation: model families, links, diagnostics, and robust standard errors.
