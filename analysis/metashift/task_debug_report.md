# SPUME MetaShift Task Construction Debug Report

- **Vocab size**: 212
- **Tasks sampled**: 50
- **Support size**: 10 per class
- **Query size**: 10 per class
- **Top-K spurious attributes**: 10

## Task Quality Metrics

| Metric | Value |
|--------|-------|
| Empty queries | 0/50 |
| Trivial tasks (concept overlap > 80%) | 0/50 |
| Genuine shift (new concepts in query) | 50/50 |
| Mean concept overlap | 13.0% |
| Mean new context rate in query | 77.9% |

## Interpretation

- **Empty queries**: should be 0. If >0, spuriousness sampling fails.
- **Genuine shift**: 50/50 tasks have concepts in query that don't appear in support. This is the intended SPUME mechanism — support and query should differ.
- **Concept overlap**: 13.0% overlap between support and query concepts. Lower is better for SPUME (forces model to not rely on spurious concepts).

## Verdict

**PASS** — Task construction is correct. Support and query sets have different concept distributions.