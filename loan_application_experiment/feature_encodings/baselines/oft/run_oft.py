import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

oft_out_file = "data/BPI17/feature_encodings/OFT/application_features.csv"

application_features = pd.read_csv(oft_out_file)

# make train test split
X, y = (
    application_features.drop("object_lifecycle_duration", axis=1),
    application_features.loc[:, "object_lifecycle_duration"],
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

params = {
    "objective": "regression",
    "metric": ["mse", "mae", "mape", "rmse"],
}
bst = lgb.train(
    params,
    train_data,
    # num_boost_round=5000,
    valid_sets=[valid_data],
    # callbacks=[lgb.early_stopping(50)],
)

y_train_preds = bst.predict(X_train)
y_valid_preds = bst.predict(X_valid)
train_mse_loss = mean_squared_error(y_train, y_train_preds)
valid_mse_loss = mean_squared_error(y_valid, y_valid_preds)

print(f"Training loss (MSE): {train_mse_loss}")
print(f"Validation loss (MSE): {valid_mse_loss}")
