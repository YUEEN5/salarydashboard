import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("dataset/clean_preprocessed_dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["job_id", "salary", "min_salary", "max_salary", "listingDate"])

# # Separate target and features
# selected_features = [
#     'role',              # High importance
#     'subcategory',       # More specific than category/broad_category
#     # 'company',           # Important based on result
#     'type',              # Employment type
#     'location'           # More specific than state
# ]
# X = df[selected_features]
# then same training split and model as before
X = df.drop(columns=["mean_salary"])
y = df["mean_salary"]

# For models that need numeric input (XGBoost, RF)
X_encoded = pd.get_dummies(X)

# Split
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# For CatBoost (can handle raw categorical)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

from catboost import CatBoostRegressor, Pool

# Identify categorical columns
cat_features = [col for col in X.columns if X[col].dtype == 'object']

# Train
model_cb = CatBoostRegressor(verbose=0, random_state=42)
model_cb.fit(X_train_cat, y_train_cat, cat_features=cat_features)

# Predict & Evaluate
y_pred_cb = model_cb.predict(X_test_cat)
print("CatBoost R²:", r2_score(y_test_cat, y_pred_cb))
print("CatBoost RMSE:", mean_squared_error(y_test_cat, y_pred_cb) ** 0.5)

# Feature importance
feat_importances_cb = model_cb.get_feature_importance()
features_cb = X.columns

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances_cb, y=features_cb)
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()
