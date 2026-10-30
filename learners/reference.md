---
title: 'Reference'
---

## Glossary

ATE (average treatment effect)
:   The average, over the whole population, of the difference between each unit's outcome with and without the treatment.

Bad control
:   A variable that should not be adjusted for, because adjusting for it creates or increases bias. The common cases are mediators and colliders (Cinelli et al., 2024).

Collider
:   A variable caused by two others, for example by both the treatment and the outcome. Adjusting for a collider opens a non-causal path between its causes.

Confounder
:   A common cause of the treatment and the outcome. Left unadjusted, it produces an association between them even when the treatment has no effect.

Cross-fitting
:   Splitting the data into *K* folds, training the models on *K* − 1 folds and predicting the held-out fold. We rotate through the folds until every unit has a prediction from a model that never saw it, and then estimate the effect on all units. This removes overfitting bias without discarding data.

Cross-validation
:   Judging a model's predictions on data it was not trained on, by rotating through folds. We use it to choose tuning settings, such as the lasso penalty.

DAG (directed acyclic graph)
:   A causal diagram in which arrows show what is believed to cause what, with no loops. We use it to decide which variables to adjust for.

Decision tree
:   A model that predicts by splitting the data into groups with a sequence of yes/no questions about the variables, and predicting the group average.

DML (double, or debiased, machine learning)
:   A family of estimators that use machine learning for the predictions of the outcome and the treatment, and combine them with an orthogonal score and cross-fitting, so that the effect estimate has a valid confidence interval (Chernozhukov et al., 2018).

FWL (Frisch–Waugh–Lovell) theorem
:   The coefficient on *D* in a multiple regression of *Y* on *D* and *X* equals the slope from regressing the residuals of *Y* (after a regression on *X*) on the residuals of *D* (after a regression on *X*). We check it on the 401(k) data in episode 2.

Gradient boosting (XGBoost)
:   A model built from many small decision trees, each fitted to the errors of the ones before it. XGBoost is a widely used implementation.

Lasso
:   Linear regression with a penalty on the sum of the absolute coefficients, which shrinks coefficients towards zero and sets some exactly to zero.

Learner
:   In DoubleML, the machine learning model used for one of the predictions. In the partially linear model the learners are `ml_l` (the outcome) and `ml_m` (the treatment).

Mediator
:   A variable on the causal path from the treatment to the outcome, such as taking part in the 401(k) example. Adjusting for it removes part of the effect.

Neyman orthogonality
:   A property of the estimating equation for the effect, under which small errors in the predictions have no first-order effect on the estimate. DML gets this property by using both residuals (rather than just one).

Overlap (positivity)
:   The assumption that every kind of unit has some chance of receiving each treatment level, so there is someone to compare with. We check it by looking at the estimated propensity scores.

PLR (partially linear regression) model
:   *Y* = θ*D* + *g*(*X*) + ε. In words, there is one effect θ for the treatment, and the confounders can affect the outcome in any shape *g*.

Potential outcomes
:   The outcomes a unit would have under each treatment level, *Y*(1) and *Y*(0). Only one is ever observed.

Propensity score
:   The probability of receiving the treatment given the confounders, Pr(*D* = 1 | *X*).

Random forest
:   An average of many decision trees, each grown on a random sample of the data and considering a random subset of the variables at each split.

Residual (leftover)
:   What is left of a variable after we subtract its prediction from the confounders, in other words $Y - \hat{Y}$ or $D - \hat{D}$. DML regresses the outcome's residuals on the treatment's residuals.

Robustness value
:   In DoubleML's sensitivity analysis, the strength (as a share of residual variation in both the outcome and the treatment) that an unmeasured confounder would need to push the estimate to zero.

Sensitivity analysis
:   Asking how far an estimate could move if there were an unmeasured confounder of a given strength.

## Further reading

- DoubleML's example [Impact of 401(k) on Financial Wealth][doubleml-401k], in Python and R: the example this lesson follows.
- Chernozhukov et al. (2024), *Applied causal inference powered by ML and AI* ([causalml-book.org][causalml-book]), a free textbook with Python and R notebooks.
- Chernozhukov et al. (2018): the original DML paper, including the 401(k) application.
- Bach et al. (2022) and the [DoubleML documentation][doubleml-docs]: the Python package used here, which also covers difference-in-differences.
- Hernán and Robins (2020), *Causal inference: What if* ([free online][what-if]): causal inference foundations.
- Cinelli et al. (2024): good and bad controls.
- Hünermund et al. (2023): why handing DML every available variable can make estimates worse.
- Brand et al. (2023): causal inference and machine learning in sociology.
- Knaus (2021): a DML study in education (musical practice and students' skills).

## References

Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML: An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, *23*(53), 1–6. https://jmlr.org/papers/v23/21-0862.html

Brand, J. E., Zhou, X., & Xie, Y. (2023). Recent developments in causal inference and machine learning. *Annual Review of Sociology*, *49*, 81–110. https://doi.org/10.1146/annurev-soc-030420-015345

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, *21*(1), C1–C68. https://doi.org/10.1111/ectj.12097

Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A., & Syrgkanis, V. (2026). Long story short: Omitted variable bias in causal machine learning. *The Review of Economics and Statistics*. Advance online publication. https://doi.org/10.1162/rest.a.1705

Chernozhukov, V., Hansen, C., Kallus, N., Spindler, M., & Syrgkanis, V. (2024). *Applied causal inference powered by ML and AI*. arXiv. https://doi.org/10.48550/arXiv.2403.02467

Cinelli, C., Forney, A., & Pearl, J. (2024). A crash course in good and bad controls. *Sociological Methods & Research*, *53*(3), 1071–1104. https://doi.org/10.1177/00491241221099552

DoubleML. (n.d.). *Python: Impact of 401(k) on financial wealth* [Example notebook]. DoubleML documentation. https://docs.doubleml.org/stable/examples/py_double_ml_pension.html

Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends. *Econometrica*, *1*(4), 387–401. https://doi.org/10.2307/1907330

Hernán, M. A., & Robins, J. M. (2020). *Causal inference: What if*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hünermund, P., Louw, B., & Caspi, I. (2023). Double machine learning and automated confounder selection: A cautionary tale. *Journal of Causal Inference*, *11*(1), Article 20220078. https://doi.org/10.1515/jci-2022-0078

Knaus, M. C. (2021). A double machine learning approach to estimate the effects of musical practice on student's skills. *Journal of the Royal Statistical Society Series A: Statistics in Society*, *184*(1), 282–300. https://doi.org/10.1111/rssa.12623

Poterba, J. M., Venti, S. F., & Wise, D. A. (1995). Do 401(k) contributions crowd out other personal saving? *Journal of Public Economics*, *58*(1), 1–32. https://doi.org/10.1016/0047-2727(94)01462-w
