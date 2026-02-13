import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from gbm_model import GradientBoostingRegressor


try:
    from sklearn.datasets import load_boston

    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    print("Loaded Boston dataset using sklearn load_boston")

except:
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    X = boston.data
    y = boston.target.astype(float)
    print("Loaded Boston dataset using openml fallback")


print("Feature shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Final RMSE:", rmse)

plt.plot(model.train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Trees")
plt.ylabel("Loss")
plt.show()
