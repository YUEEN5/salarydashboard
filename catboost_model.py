import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool

# Load dataset
df = pd.read_csv("dataset/clean_preprocessed_dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["job_id", "salary", "min_salary", "max_salary", "listingDate"])

# Define features and target
selected_features = ['category', 'role', 'location', 'type']
X = df[selected_features]
y = df["mean_salary"]

# Split data
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
cat_features = [col for col in X.columns if X[col].dtype == 'object']

# Initialize and train model
model_cb = CatBoostRegressor(verbose=0, random_state=42)
model_cb.fit(X_train_cat, y_train_cat, cat_features=cat_features)

# # Save the model
# model_cb.save_model("catboost_salary_model2.cbm")

# Evaluate the model
y_pred_cb = model_cb.predict(X_test_cat)
print("CatBoost MAE:", mean_absolute_error(y_test_cat, y_pred_cb))
print("CatBoost R²:", r2_score(y_test_cat, y_pred_cb))
print("CatBoost RMSE:", mean_squared_error(y_test_cat, y_pred_cb) ** 0.5)

# Plot feature importance
feat_importances_cb = model_cb.get_feature_importance()
features_cb = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importances_cb, y=features_cb)
plt.title("CatBoost Feature Importance")
plt.tight_layout()
plt.show()
