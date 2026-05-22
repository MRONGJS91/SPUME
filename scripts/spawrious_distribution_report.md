# Spawrious Dataset: Class-Background Distribution Analysis

## 1. Overview

- **spawrious_o2o_easy**: 60,816 train / 7,632 test, classes=['bulldog', 'corgi', 'dachshund', 'labrador'], backgrounds=['beach', 'desert', 'dirt', 'jungle', 'snow']
  - Marginal background JS divergence: 0.0000
  - Per-class JS: bulldog=0.0000, corgi=0.0000, dachshund=0.0000, labrador=0.0000

- **spawrious_o2o_medium**: 60,816 train / 7,632 test, classes=['bulldog', 'corgi', 'dachshund', 'labrador'], backgrounds=['beach', 'desert', 'dirt', 'jungle', 'mountain', 'snow']
  - Marginal background JS divergence: 0.0000
  - Per-class JS: bulldog=0.0000, corgi=0.0000, dachshund=0.0000, labrador=0.0000

## 2. Interpretation

- **JS divergence** measures how different the train and test distributions are.
  - 0 = identical distributions; larger values = more shift.
- **Marginal background JS** tells whether the overall background mix changes between train and test.
- **Per-class JS** tells whether a specific class sees its background distribution shift.

## 3. Key Findings

### spawrious_o2o_easy
- Marginal background shift (JS): **0.0000**
- Largest per-class shift: **bulldog** (JS=0.0000)
- Smallest per-class shift: **bulldog** (JS=0.0000)
- **Conclusion**: Train and test class-background distributions are nearly identical. The correlation structure does NOT change between splits.

### spawrious_o2o_medium
- Marginal background shift (JS): **0.0000**
- Largest per-class shift: **bulldog** (JS=0.0000)
- Smallest per-class shift: **bulldog** (JS=0.0000)
- **Conclusion**: Train and test class-background distributions are nearly identical. The correlation structure does NOT change between splits.
