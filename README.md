# AFM 244 – Week 11 Quiz: Target Corp. Quarterly Revenue Regression

**Author:** Christine Miao
**Platform:** Google Colab / Jupyter Notebook
**Prepared for (in-notebook memo):** Chapman Wealth Management (via QuantFolio Solutions)

## Overview

This notebook builds a time-trend regression model with two seasonal/event
dummy variables to explain and forecast **Target Corporation's (TGT)**
quarterly revenue. It combines the code/analysis with a client-facing
memorandum summarizing the findings.

## Data

- **Source file:** `qSales_2024.csv`
- Filtered to `tic == 'TGT'` (Target Corp), giving **93 quarterly observations**
  from **2001-01-31 to 2024-01-31** (fiscal quarters).
- Key columns used: `datadate`, `fqtr` (fiscal quarter), `datacqtr` (calendar
  quarter, e.g. `2020Q2`), `saleq` (quarterly revenue, $M).

## Feature Engineering

| Variable | Definition |
|---|---|
| `time` | Sequential index, 1 to 93, representing quarter order |
| `holiday_dv` | 1 if fiscal Q4 (Nov 1–Jan 31, the holiday shopping quarter), else 0 |
| `holiday_interaction` | `time × holiday_dv` |
| `postCovid_dv` | 1 if `datacqtr >= 2020Q2`, else 0 |
| `postCovid_interaction` | `time × postCovid_dv` |

**Note on train/test split:** The notebook explicitly decided *against* a
75/25 train-test split, since only ~16.13% of the data (15 of 93 rows) falls
in the post-COVID period — a split would have left too few post-COVID rows
to train on. The **full dataset** was used to fit the model instead.

## Model

Fit via `statsmodels.api.OLS` with a constant, `time`, both dummies, and both
interaction terms as predictors:

```
TargetRevenue = 9,744.83
              + 123.92 × time
              + 3,955.93 × holiday
              + 16.95   × (time × holiday)
              − 2,659.14 × postCovid
              + 84.20   × (time × postCovid)
```

**Fit quality:** Adjusted R² = **0.953** — the model explains ~95.3% of
quarter-to-quarter revenue variation.

## Prediction & Confidence Intervals

Using `model.get_prediction(x).summary_frame(alpha=0.2)`, the notebook
generates fitted values plus an **80% confidence interval** (`obs_ci_lower`,
`obs_ci_upper`) for each quarter, then plots:
- Actual revenue (solid line, circle markers)
- Predicted revenue (dashed orange line, x markers)
- Shaded 80% CI band around the prediction

The CI band width is roughly **$3,200M** consistently, and actual values
generally fall within it.

## Key Findings (from the in-notebook memo)

- **Holiday effect (`holiday_dv`):** Q4 revenue jumps by **$3,955.93M** versus
  other quarters, attributed to holiday shopping season; the interaction term
  (+16.95/quarter) suggests this seasonal lift has grown only modestly over
  time.
- **Post-COVID effect (`postCovid_dv`):** Captures an elevated revenue level
  starting 2020 Q2, linked to accelerated e-commerce/digital adoption
  (online shopping, curbside pickup). Revenue has **stayed elevated** rather
  than reverting, through the most recent data point (2023 Q4).
- **Overall trend:** Revenue is both cyclical (Q4 spikes) and structurally
  growing over the 23-year window.

## Recommendations (from the memo)

1. **Conduct qualitative analysis** — supplement the numbers with insight
   from investor relations material (earnings calls, quarterly reports).
2. **Analyze other business metrics** — revenue alone doesn't capture
   operational health; compare profitability, liquidity, efficiency, and
   solvency ratios against industry benchmarks.
3. **Forecast cautiously** — the post-COVID lift is a relatively recent,
   unproven-as-permanent shift; don't assume it persists indefinitely without
   monitoring current developments.

## Tools/Libraries Used

- `pandas`, `numpy` — data handling
- `statsmodels` (`sm.OLS`, `get_prediction`) — regression + prediction intervals
- `matplotlib` — actual-vs-predicted visualization with confidence band

## Deliverable Format

The notebook concludes with a formatted **memorandum** (To/From/CC/Date/Re
header, Scope, Methodology, Key Findings, Recommendations) addressed to
Chapman Wealth Management, framing the regression results as investment
due-diligence input.
