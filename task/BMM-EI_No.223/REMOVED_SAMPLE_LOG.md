# Removed Sample Log

## BMM-EI No. 223 - Application_Failure Sample Removal

### Reason
The `System Issue Type` target column originally contained 5 classes:
- `Normal_Operation`           : 7,620 samples
- `Performance_Degradation`    : 2,183 samples
- `Resource_Exhaustion`        : 1,245 samples
- `Network_Failure`            :   951 samples
- `Application_Failure`        :     **1 sample**  <-- problematic

A single-sample class is incompatible with `StratifiedKFold(n_splits=5)`,
which requires at least `n_splits` (5) samples per class. The single
`Application_Failure` row also cannot contribute meaningfully to model
training, statistical analysis, or sensitivity/uncertainty computations.

### Action Taken
- The single `Application_Failure` row was removed from the `Encoded_Data` sheet.
- Remaining classes were relabeled to contiguous integers `0..3`:

| New label | Original label              | Count |
|-----------|-----------------------------|-------|
| 0         | Network_Failure             |   951 |
| 1         | Normal_Operation            | 7,620 |
| 2         | Performance_Degradation     | 2,183 |
| 3         | Resource_Exhaustion         | 1,245 |

### Result
- Encoded_Data shape: **11,999 x 32** (down from 12,000 x 32)
- All downstream steps (Chi-Square, VIF, K-Fold CV, model exports,
  Brier decomposition, t-test, Friedman test, ANOVA sensitivity,
  Entropy uncertainty) will operate on the 11,999-row, 4-class dataset.

### Files affected
- `Data.xlsx` sheet `Encoded_Data` (overwritten)
- This log file (`REMOVED_SAMPLE_LOG.md`)
