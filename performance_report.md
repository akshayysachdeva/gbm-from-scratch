# Performance Report

## Model

Gradient Boosting Regressor implemented from scratch using Decision
Trees.

## Training Behaviour

During training, the loss decreases gradually as the number of trees
increases. This shows that each new tree is learning from the residual
errors of previous trees, which is expected behaviour in gradient
boosting.

## Training Curve

The training loss curve shows a downward trend, indicating that the
model is learning patterns from the dataset effectively.

## Test Performance

The model was evaluated using RMSE (Root Mean Squared Error) on a
held-out test dataset.

## Final RMSE

2.753317331832554

## Model Observations

-   Predictions follow actual values reasonably well.
-   Residual errors are distributed around zero.
-   No major signs of overfitting observed from training curve.

## Conclusion

The implemented Gradient Boosting model successfully learns housing
price patterns and generalizes well to unseen data.
