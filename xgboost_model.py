import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("dataset/clean_preprocessed_dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["job_id", "salary", "min_salary", "max_salary", "listingDate"])

# Separate target and features
X = df.drop(columns=["mean_salary"])
y = df["mean_salary"]

# For models that need numeric input (XGBoost, RF)
X_encoded = pd.get_dummies(X)

# Split
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# For CatBoost (can handle raw categorical)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBRegressor

model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(X_train_enc, y_train)

# Predict & Evaluate
y_pred_xgb = model_xgb.predict(X_test_enc)
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))
# print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb) ** 0.5)

# Feature importance
feat_importances_xgb = model_xgb.feature_importances_
features_xgb = X_encoded.columns

# Plot
importances_df = pd.DataFrame({'Feature': features_xgb, 'Importance': feat_importances_xgb})
importances_df = importances_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x="Importance", y="Feature")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()
