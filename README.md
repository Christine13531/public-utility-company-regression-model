# AFM 244 – Week 10 Quiz: Seasonal Regression Analysis

**Author:** Christine Miao
**Platform:** Google Colab (exported to PDF)

## Overview

This notebook builds and evaluates two Ordinary Least Squares (OLS) regression
models that forecast **revenue** from **production**, using seasonal dummy
variables (and dummy-interaction terms) to test whether the revenue/production
relationship differs by season. One model isolates a **winter** effect, the
other isolates a **fall** effect.

> I'm not fully certain of every step, since the PDF's text layer was
> corrupted/garbled on export and I could only reliably read the portions that
> rendered as images (code cells, computed equations, and printed metrics).
> You may want to verify against the live Colab notebook for the full
> narrative/markdown commentary.

## Data

Two dataframes are used:

- `dt4training` — training data
- `dt4testing` — holdout data for out-of-sample assessment

Columns visible in the data preview:

| Column | Description |
|---|---|
| `type` | dataset/segment label (e.g. `dt4testing`) |
| `date` | monthly date |
| `revenue` | dependent variable |
| `production` | primary independent variable |
| `coolDD` / `heatDD` | cooling/heating degree-days |
| `winter_DV` | dummy variable, 1 if winter month |
| `fall_DV` | dummy variable, 1 if fall month |
| `winter_interaction` | `production × winter_DV` |
| `fall_interaction` | `production × fall_DV` |

## Methodology

Both models are fit with `statsmodels.api.OLS` (`sm.OLS(y, x).fit()`), using
`production`, a seasonal dummy, and the corresponding interaction term as
predictors (plus a constant added via `sm.add_constant`).

### Model 1 — Winter model

```
revenue = 5,629,257.08 + 13.51 × production
                       + 14.16 × production × winter
                       − 201,742.73 × winter
```

Split into two implied equations:

- **Non-winter months:** `revenue = 5,629,257.08 + 13.51 × production`
- **Winter months:** `revenue = 5,427,514.35 + 27.67 × production`

**Performance (on `dt4testing`):**
- MAPE: **15.90%**
- Adjusted R²: **0.7518**

### Model 2 — Fall model

```
revenue = 6,118,386.30 + 18.30 × production
                       − 7.67 × production × fall
                       − 477,240.43 × fall
```

Split into two implied equations:

- **Non-fall months:** `revenue = 6,118,386.30 + 18.30 × production`
- **Fall months:** `revenue = 5,641,145.87 + 10.63 × production`

**Performance (on `dt4testing`):**
- MAPE: **22.02%**
- Adjusted R²: **0.4514**

## Evaluation Approach

For each model, the notebook:
1. Builds the test design matrix (`production`, seasonal dummy, interaction) with a constant.
2. Generates predictions with `model.predict(...)`.
3. Computes **Mean Absolute Percentage Error (MAPE)**:
   `mean(abs((actual − predicted) / actual)) × 100`.
4. Reports the model's **adjusted R²** (`model.rsquared_adj`).

## Visualization

Each model is plotted with `matplotlib`:
- A scatter plot of actual `production` vs. `revenue`.
- Two fitted lines derived from the model coefficients — one for the season
  in question (e.g. winter) and one for the rest of the year — overlaid in
  different colors (e.g. red = non-winter, blue = winter) to visually compare
  slopes/intercepts.

## Key Takeaway

The **winter model fits noticeably better** than the fall model (lower MAPE,
higher adjusted R²), suggesting the winter dummy/interaction captures a more
meaningful shift in the production–revenue relationship than the fall dummy
does — though I'd treat the fall model's weaker R² (0.4514) as a sign that
season alone may not explain much of the variation in fall revenue.

## Tools/Libraries Used

- `pandas` — data handling
- `statsmodels` (`sm.OLS`, `sm.add_constant`) — regression modeling
- `matplotlib` (`plt`) — scatter/line plots

## Suggested Verification

Because of the PDF export issue described above, I'd recommend re-checking
the original Colab notebook directly for:
- Any written interpretation/commentary in markdown cells
- The exact number of rows/months in `dt4training` vs. `dt4testing`
- Any additional quiz questions or written answers not captured here
