---
site: sandpaper::sandpaper_site
---

This self-paced lesson follows on from the QMEG (Quantitative Methods in Education Group) masterclass *Introduction to Causal Inference with Machine Learning*, delivered on Microsoft Teams on 13 January 2027 at the University of Southampton. In this lesson we pick up where Part 4 of the slides left off and work through its example in Python, step by step.

The example asks whether eligibility for a 401(k), a US workplace pension, raised households' savings. We use data on 9,915 households from 1991 (Chernozhukov et al., 2018), and the code follows DoubleML's own example, [Impact of 401(k) on Financial Wealth][doubleml-401k].

:::::::::::::::::::::::::::::::::::::::::: prereq

## Prerequisites

- The masterclass, or its slides. Each episode tells you which slides it builds on.
- Basic regression. You should know what a regression coefficient, a standard error and a 95% confidence interval are.
- Some experience of running code in a Jupyter notebook. You do not need to know Python well, because every exercise gives you most of the code.
- Python and the lesson's packages, installed by following the [Setup](learners/setup.md) instructions (about 15 minutes).

::::::::::::::::::::::::::::::::::::::::::::::::::

We do not cover difference-in-differences, panel data or the theory behind DML in any depth. The [Reference](learners/reference.md) page has a glossary and suggestions for further reading.
