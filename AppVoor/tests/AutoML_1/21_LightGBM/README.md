# Summary of 21_LightGBM

## LightGBM
- **objective**: binary
- **metric**: binary_logloss
- **num_leaves**: 31
- **learning_rate**: 0.1
- **feature_fraction**: 1.0
- **bagging_fraction**: 0.9
- **min_data_in_leaf**: 5
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **k_folds**: 10
 - **shuffle**: False

## Optimized metric
logloss

## Training time

1.8 seconds

## Metric details
|           |    score |    threshold |
|:----------|---------:|-------------:|
| logloss   | 0.484824 | nan          |
| auc       | 0.822381 | nan          |
| f1        | 0.686767 |   0.313004   |
| accuracy  | 0.778646 |   0.492712   |
| precision | 0.966667 |   0.839317   |
| recall    | 1        |   0.00820802 |
| mcc       | 0.497909 |   0.313004   |


## Confusion matrix (at threshold=0.313004)
|                     |   Predicted as negative |   Predicted as positive |
|:--------------------|------------------------:|------------------------:|
| Labeled as negative |                     376 |                     124 |
| Labeled as positive |                      63 |                     205 |

## Learning curves
![Learning curves](learning_curves.png)