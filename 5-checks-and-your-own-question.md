---
title: "Checks, and your own research"
teaching: 10
exercises: 15
---

:::::::::::::::::::::::::::::::::::::: questions

- How strong would an unmeasured confounder need to be to change the answer?
- How much does the estimate depend on the random split into folds?
- Where could DML help in my own research?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Run and interpret DoubleML's sensitivity analysis for unmeasured confounding
- Repeat the cross-fitting with several random splits and report the spread
- Identify a question from your own research where DML could help

::::::::::::::::::::::::::::::::::::::::::::::::

## Set up

Let's start by fitting the random forest model from episode 3 again, written a little more compactly this time:

```python
!pip install doubleml     # once per session
import numpy as np
import doubleml as dml
from doubleml.datasets import fetch_401K
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

data = fetch_401K(return_type='DataFrame')
features_base = ['age', 'inc', 'educ', 'fsize', 'marr',
                 'twoearn', 'db', 'pira', 'hown']
data_dml_base = dml.DoubleMLData(data, y_col='net_tfa', d_cols='e401', x_cols=features_base)

randomForest = RandomForestRegressor(
    n_estimators=500, max_depth=7, max_features=3, min_samples_leaf=3)
randomForest_class = RandomForestClassifier(
    n_estimators=500, max_depth=5, max_features=4, min_samples_leaf=7)

np.random.seed(123)
dml_plr_forest = dml.DoubleMLPLR(data_dml_base, ml_l = randomForest,
                                 ml_m = randomForest_class, n_folds = 3)
dml_plr_forest.fit(store_predictions=True)
print(dml_plr_forest.summary.round(2))
```

```output
         coef  std err     t  P>|t|    2.5 %    97.5 %
e401  8909.63  1321.82  6.74    0.0  6318.91  11500.36
```

## How strong would an unmeasured confounder need to be?

Every estimate so far rests on the assumption that the nine confounders (income, age, education, family size, marital status, two earners, a defined benefit pension, IRA participation and home ownership) capture everything that drives both eligibility and savings. In episode 1 we named the obvious worry, which is a preference for saving. We have no measure of it in the data, so instead we ask how strong it would have to be to change our answer.

DoubleML's sensitivity analysis (Chernozhukov et al., 2026) describes this hypothetical confounder with two numbers:

- `cf_y` is the share of the residual variation in savings that the confounder would explain.
- `cf_d` is how much the confounder would add to the explained variation in eligibility.

Let's try a confounder that explains 4% of the residual variation in savings and adds 3% for eligibility:

```python
dml_plr_forest.sensitivity_analysis(cf_y=0.04, cf_d=0.03)
print(dml_plr_forest.sensitivity_summary)
```

```output
================== Sensitivity Analysis ==================

------------------ Scenario          ------------------
Significance Level: level=0.95
Sensitivity parameters: cf_y=0.04; cf_d=0.03, rho=1.0

------------------ Bounds with CI    ------------------
         CI lower  theta lower        theta   theta upper      CI upper
e401  2359.496777  4610.983759  8909.634078  13208.284397  15430.415812

------------------ Robustness Values ------------------
      H_0    RV (%)   RVa (%)
e401  0.0  7.029209  5.233154
```

The `theta lower` and `theta upper` columns show how far the estimate could move if a confounder that strong existed, and you should see that the whole interval still lies well above zero. `CI lower` and `CI upper` widen those bounds a little further to allow for sampling error.

The last block of the output gives the **robustness value**, `RV`. This is the strength a confounder would need, with the same share for both `cf_y` and `cf_d`, to push the estimate all the way to zero. `RVa` is the same idea with sampling error allowed for.

Whether a confounder that strong is plausible is a judgement about the setting, and the data cannot make that judgement for you. What the analysis does is turn a vague "there might be confounding" into a number that you and a reviewer can argue about.

::::::::::::::::::::::::::::::::::::: challenge

## Challenge 1: a stronger confounder

Rerun the sensitivity analysis with `cf_y=0.08, cf_d=0.08`. Does the lower bound for the estimate stay above zero?

:::::::::::::::::::::::: solution

```python
dml_plr_forest.sensitivity_analysis(cf_y=0.08, cf_d=0.08)
print(dml_plr_forest.sensitivity_summary)
```

```output
================== Sensitivity Analysis ==================

------------------ Scenario          ------------------
Significance Level: level=0.95
Sensitivity parameters: cf_y=0.08; cf_d=0.08, rho=1.0

------------------ Bounds with CI    ------------------
         CI lower  theta lower        theta  theta upper      CI upper
e401 -3820.980088 -1283.868465  8909.634078  19103.13662  21578.115315

------------------ Robustness Values ------------------
      H_0    RV (%)   RVa (%)
e401  0.0  7.029209  5.233154
```

A confounder this strong is above the robustness value, so the lower bound drops below zero. In other words, if such a confounder existed, we could no longer rule out an effect of zero. The robustness value would have told you this already, without trying values one by one.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## How much does the split into folds matter?

The three folds are drawn at random, so it is fair to ask how much our estimate depends on that one draw. Setting `n_rep=5` repeats the whole cross-fitting with five different random splits and reports the median estimate. Expect this to take about five times as long as a single fit:

```python
np.random.seed(123)
dml_plr_forest_rep = dml.DoubleMLPLR(data_dml_base, ml_l = randomForest,
                                     ml_m = randomForest_class, n_folds = 3, n_rep = 5)
dml_plr_forest_rep.fit()
print("Estimate from each split:", dml_plr_forest_rep.all_coef.round(0).ravel().tolist())
print(dml_plr_forest_rep.summary.round(2))
```

```output
Estimate from each split: [8947.0, 9012.0, 9099.0, 9169.0, 8821.0]
        coef  std err     t  P>|t|   2.5 %    97.5 %
e401  9012.1  1317.42  6.84    0.0  6430.0  11594.21
```

The first line lists the five estimates, each from the same model on a different random split. Their spread is small compared with the standard error of about $1,300, so the choice of split is not what drives the result here. With small samples or very flexible learners the spread can be much larger, so it is good practice to repeat the split for any result you report.

## A checklist for before you run DML

The table below gathers the steps from this lesson that you would work through for a question of your own. Most of them are decisions about research design and need little or no code. The right-hand column shows where each step came up.

| Step | In this lesson |
|---|---|
| Write the causal question: treatment, outcome, population | Episode 1 |
| Draw a causal diagram (DAG) and pick the confounders from it | Episode 1 |
| Leave out mediators and colliders (bad controls) | Episode 1, challenge 1 |
| Explain why no major confounder is missing | Episode 1, challenge 2 |
| Check overlap between treated and untreated | Not covered here; see DoubleML's [401(k) example][doubleml-401k] |
| Try several ML models and data splits, and report the spread | Episodes 4 and 5 |
| Run a sensitivity analysis for unmeasured confounding | Episode 5 |
| Report a simple regression alongside, for comparison | Episode 2 |

::::::::::::::::::::::::::::::::::::: discussion

## Discussion: where could DML help in your own research?

DML suits questions that ask what would happen if something changed, where an experiment is not possible but you have data on the things that drive both the treatment and the outcome. The masterclass gave three examples from different fields (slide 5): would a follow-up phone call reduce hospital readmissions, would offering a workplace pension raise savings, and would tutoring help pupils pass? Think of a question like these from your own work, and consider:

- What is the treatment, and what is the outcome?
- What affects both of them, and do your data measure it?
- Which confounder could you not measure, and would it make the estimate too high or too low?
- Would DML add anything over a regression with controls? It helps most when there are many confounders, or when their effects are unlikely to be straight lines.

If you are working through the lesson with others, compare your answers. Questions from quite different fields often turn out to have the same structure.

::::::::::::::::::::::::::::::::::::::::::::::::

## Where to go next

If you want to take DML further, these are good places to start:

- The [DoubleML example][doubleml-401k] that this lesson follows, in both Python and R.
- The [DoubleML documentation][doubleml-docs], which covers other designs, including DML for difference-in-differences (`DoubleMLDID`). Part 2 of the slides introduced DID.
- Chernozhukov et al. (2024), *Applied causal inference powered by ML and AI*, which is free online with notebooks.
- Hünermund et al. (2023), on why handing DML every available variable can make estimates worse.
- The [Reference](../learners/reference.md) page, which has a glossary and the full reference list.

::::::::::::::::::::::::::::::::::::: keypoints

- A sensitivity analysis asks how strong an unmeasured confounder would need to be to change the conclusion, and the robustness value summarises the answer in one number.
- Repeating the cross-fitting with several random splits (`n_rep`) shows how much the estimate depends on the split.
- DML suits questions about what would happen if something changed, when an experiment is not possible but the confounders are measured.
- Most of the work in a credible DML analysis goes into design, which covers the question, the causal diagram, the confounders, overlap and the sensitivity of the result.

::::::::::::::::::::::::::::::::::::::::::::::::
