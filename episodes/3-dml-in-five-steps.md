---
title: "DML in five steps with DoubleML"
teaching: 10
exercises: 5
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I run double machine learning on the 401(k) data with DoubleML?
- How do I read the output from DoubleML?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Run the five steps from the slides (load the data, set the roles, choose two models, fit and read the result)
- Interpret the estimate and its 95% confidence interval
- Connect DoubleML's stored predictions to the residuals from episode 2

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

## From the masterclass

This episode follows the five steps on slides 35 to 39, from "Step 1: load the data" to "Step 5: read the result". The code for steps 1 to 4 is copied unchanged from slides 35 to 38 (if you came to the workshop, it is the code you ran there), and step 5 reads the output shown on slide 39. If you run the blocks in order, you will get the same numbers as the slides.

:::::::::::::::::::::::::::::::::::::::::::::::::

## Step 1: load the data

```python
!pip install doubleml     # once per session
import numpy as np
import doubleml as dml
from doubleml.datasets import fetch_401K

data = fetch_401K(return_type='DataFrame')
print(data[['e401', 'net_tfa']]
      .groupby('e401').mean().diff())
```

```output
          net_tfa
e401             
0             NaN
1     19559.34475
```

You should see the same raw gap as in episode 1, with eligible households holding $19,559 more on average.

## Step 2: tell DoubleML which variable is which

There is no machine learning in this step, yet it matters more than any of the others. We give DoubleML the outcome in `y_col`, the treatment in `d_cols` and the confounders in `x_cols`, and we chose those confounders from the causal diagram in episode 1.

```python
features_base = ['age', 'inc', 'educ',
                 'fsize', 'marr', 'twoearn',
                 'db', 'pira', 'hown']

data_dml_base = dml.DoubleMLData(data,
    y_col='net_tfa',         # outcome
    d_cols='e401',           # treatment
    x_cols=features_base)    # confounders
```

Let's print the object to check what DoubleML will use:

```python
print(data_dml_base)
```

```output
================== DoubleMLData Object ==================

------------------ Data summary      ------------------
Outcome variable: net_tfa
Treatment variable(s): ['e401']
Covariates: ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']
Instrument variable(s): None
No. Observations: 9915
------------------ DataFrame info    ------------------
<class 'pandas.DataFrame'>
RangeIndex: 9915 entries, 0 to 9914
Columns: 14 entries, nifa to hown
dtypes: float32(4), int8(10)
memory usage: 251.9 KB
```

## Step 3: choose the two machine learning models

We need two random forests. The first, `randomForest`, predicts savings (*Y*) from *X* and is model 1 in the diagram on slide 30. The second, `randomForest_class`, predicts eligibility (*D*) from *X* and is model 2. Eligibility is a yes or no variable, so model 2 has to be a classifier. We use the settings from DoubleML's own example, which set the number of trees at 500 and limit how deep each tree may grow, how many variables each split considers and how few households a leaf may hold.

```python
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier)

randomForest = RandomForestRegressor(
    n_estimators=500, max_depth=7,
    max_features=3, min_samples_leaf=3)

randomForest_class = RandomForestClassifier(
    n_estimators=500, max_depth=5,
    max_features=4, min_samples_leaf=7)
```

## Step 4: fit the model

```python
np.random.seed(123)  # same result each run
dml_plr_forest = dml.DoubleMLPLR(
    data_dml_base,
    ml_l = randomForest,
    ml_m = randomForest_class,
    n_folds = 3)

dml_plr_forest.fit(store_predictions=True)

forest_summary = dml_plr_forest.summary
print(forest_summary.round(2))
```

```output
         coef  std err     t  P>|t|    2.5 %    97.5 %
e401  8909.63  1321.82  6.74    0.0  6318.91  11500.36
```

`DoubleMLPLR` is the **partially linear regression** model named on slide 38, *Y* = θ*D* + *g*(*X*) + ε. In other words, eligibility has a single effect θ on savings, while the confounders can affect savings through a function *g* of any shape. (Episode 2 showed why the slope of the residuals estimates θ.) Setting `n_folds = 3` asks for three-fold cross-fitting. When we call `fit()`, DoubleML runs everything we did in episode 2. It trains both forests on two folds and predicts the third, rotates the folds, and then regresses residual savings on residual eligibility.

Notice that we set the seed just before creating the model. DoubleML draws the random split into folds when the model is *created*, and the forests make their own random choices when they are *fitted*. Setting the seed before both steps fixes all of these choices, so you get the same numbers as on the slides. Without the seed the estimate changes from run to run, and in episode 5 we repeat the split five times and get anything from $8,821 to $9,169. The numbers may also differ slightly if you have a different version of scikit-learn or DoubleML.

## Step 5: read the result

Each column of the summary tells us something different:

- `coef` is the estimated effect of eligibility on net financial assets, about $8,910.
- `std err` is its standard error, about $1,322.
- `t` and `P>|t|` test the hypothesis of no effect. The `0.0` is rounding, and *p* is below .001.
- `2.5 %` and `97.5 %` are the two ends of the 95% confidence interval, which runs from $6,319 to $11,500.

Compare this with the raw gap of $19,559. Once we take the nine confounders out of both savings and eligibility, the estimate drops to less than half of the raw gap, so about 54% of the gap was confounding. The income bands in episode 1 suggest that much of it comes through income. The estimate still assumes that there is no confounder outside *X*, and this is the assumption to argue about. We will come back to it in episode 5.

## The residuals DoubleML used

Because we fitted the model with `store_predictions=True`, DoubleML kept its cross-fitted predictions. This means we can redo step C from episode 2 with DoubleML's own residuals:

```python
y_resid = data['net_tfa'] - dml_plr_forest.predictions['ml_l'][:, 0, 0]
d_resid = data['e401'] - dml_plr_forest.predictions['ml_m'][:, 0, 0]
print(f"Slope of the residuals: ${(d_resid @ y_resid) / (d_resid @ d_resid):,.2f}")
print(f"DoubleML's estimate:    ${dml_plr_forest.coef[0]:,.2f}")
```

```output
Slope of the residuals: $8,909.63
DoubleML's estimate:    $8,909.63
```

You should see that the two numbers match to the cent. These are the same residuals plotted on slide 32. The `[:, 0, 0]` is there because `predictions['ml_l']` has three dimensions (households, repetitions of the split and treatments), and we have only one repetition and one treatment.

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: five folds instead of three

Refit the model with `n_folds = 5` and keep the seed as it is. How much does the estimate move, and is the change large compared with the standard error? Replace the `____` with the number of folds before you run the cell:

```python
np.random.seed(123)
dml_plr_forest5 = dml.DoubleMLPLR(data_dml_base, ml_l = randomForest,
                                  ml_m = randomForest_class, n_folds = ____)
dml_plr_forest5.fit()
print(dml_plr_forest5.summary.round(2))
```

:::::::::::::::::::::::: solution

```python
np.random.seed(123)
dml_plr_forest5 = dml.DoubleMLPLR(data_dml_base, ml_l = randomForest,
                                  ml_m = randomForest_class, n_folds = 5)
dml_plr_forest5.fit()
print(dml_plr_forest5.summary.round(2))
```

```output
         coef  std err     t  P>|t|    2.5 %    97.5 %
e401  8915.54  1305.55  6.83    0.0  6356.71  11474.37
```

The estimate barely moves, from $8,910 to $8,916, which is a tiny fraction of the standard error. With five folds each model trains on 80% of the households instead of two thirds. Both choices are reasonable, so pick one before you look at the answer and report which one you used.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: keypoints

- `DoubleMLData` sets the roles of the variables, `DoubleMLPLR` combines the data with two learners, and `fit()` runs the cross-fitting and the final regression.
- With random forests and three folds, we estimate that 401(k) eligibility raised net financial assets by about $8,910 (95% CI $6,319 to $11,500), against a raw gap of $19,559.
- Setting `np.random.seed()` just before creating the model keeps the folds and forests the same on every run.
- DoubleML's estimate is exactly the slope of its stored residuals.

::::::::::::::::::::::::::::::::::::::::::::::::
