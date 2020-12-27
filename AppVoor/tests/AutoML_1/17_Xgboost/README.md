# Summary of 17_Xgboost

## Extreme Gradient Boosting (Xgboost)
- **objective**: binary:logistic
- **eval_metric**: logloss
- **eta**: 0.025
- **max_depth**: 8
- **min_child_weight**: 9
- **subsample**: 0.9
- **colsample_bytree**: 0.5
- **explain_level**: 0

## Validation
 - **validation_type**: kfold
 - **k_folds**: 10
 - **shuffle**: False

## Optimized metric
logloss

## Training time

2.5 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.471584 | nan         |
| auc       | 0.835918 | nan         |
| f1        | 0.695507 |   0.360968  |
| accuracy  | 0.768229 |   0.506371  |
| precision | 0.873418 |   0.745017  |
| recall    | 1        |   0.0116265 |
| mcc       | 0.513792 |   0.373362  |


## Confusion matrix (at threshold=0.360968)
|                     |   Predicted as negative |   Predicted as positive |
|:--------------------|------------------------:|------------------------:|
| Labeled as negative |                     376 |                     124 |
| Labeled as positive |                      59 |                     209 |

## Learning curves
![Learning curves](learning_curves.png)